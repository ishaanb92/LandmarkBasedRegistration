"""

Script to predict landmark correspondences for COPD patients using models trained in a leave-one-out fashion

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
from monai.inferers import sliding_window_inference
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
import os, sys
from lesionmatching.analysis.visualize import *
from lesionmatching.arch.model import LesionMatchingModel
from lesionmatching.data.deformations import *
from lesionmatching.data.datapipeline import *
from lesionmatching.arch.loss import create_ground_truth_correspondences
from lesionmatching.analysis.metrics import get_match_statistics
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import save_ras_as_itk
import shutil
import numpy as np
import random
from math import ceil
from math import floor

COPD_DIR = '/home/ishaan/COPDGene/mha'
NUM_WORKERS=4

def test(args):

    if args.soft_masking is False:
        proceed = input('Softmasking is set to False. Are you sure you want to continue? (y/n)')
        if proceed == 'n':
            return
        elif proceed == 'y':
            pass

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    checkpoint_dir = args.checkpoint_dir

    if os.path.exists(args.out_dir) is True:
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_determinism(seed=args.seed)

    rescaling_stats = joblib.load(os.path.join(COPD_DIR,
                                               'rescaling_stats_copd.pkl'))

    print(rescaling_stats)

    n_patients = len([f.name for f in os.scandir(COPD_DIR) if f.is_dir()])
    pids = ['copd{}'.format(idx+1) for idx in range(n_patients)] # Ensure that the patients are in asc. order of ID

    model = LesionMatchingModel(W=args.window_size,
                                K=args.kpts_per_batch,
                                descriptor_length=args.desc_length)

    # Loop over patient
    for idx, pid in enumerate(pids):

        # Define dataset and dataloader
        data_dicts = create_data_dicts_dir_lab_paired(patient_dir_list=[os.path.join(COPD_DIR, pid)],
                                                      dataset='copd',
                                                      soft_masking=args.soft_masking)

        data_loader = create_dataloader_dir_lab_paired(data_dicts=data_dicts,
                                                       batch_size=args.batch_size,
                                                       num_workers=NUM_WORKERS,
                                                       rescaling_stats=rescaling_stats,
                                                       patch_size=(128, 128, 96),
                                                       overlap=0.0)

        # Load the model that was trained without the current PID
        pat_checkpoint_dir = os.path.join(args.checkpoint_dir, 'holdout_{}'.format(idx+1))
        load_dict = load_model(model=model,
                               checkpoint_dir=pat_checkpoint_dir,
                               training=False)

        model = load_dict['model']
        out_dir = os.path.join(args.out_dir, pid)
        os.makedirs(out_dir)
        model.to(device)

        # Perform inference
        with torch.no_grad():
            # Loop over batches
            for batch_idx, batch_data in enumerate(data_loader):

                patient_id = batch_data['patient_id'][0]
                images, images_hat, mask, mask_hat = (batch_data['moving_image'], batch_data['fixed_image'],\
                                                      batch_data['moving_lung_mask'], batch_data['fixed_lung_mask'])

                fixed_metadata_list = detensorize_metadata(metadata=batch_data['fixed_metadata'],
                                                           batchsz=images.shape[0])

                moving_metadata_list = detensorize_metadata(metadata=batch_data['moving_metadata'],
                                                            batchsz=images.shape[0])

                assert(images.shape == images_hat.shape)

                b, c, i, j, k = images.shape
                n_patches = len(batch_data['patch_dict']['origins'])
                fixed_landmarks_np = None
                moving_landmarks_np = None

                # Loop over patches
                for idx, (patch_origin, patch_end) in \
                enumerate(zip(batch_data['patch_dict']['origins'], batch_data['patch_dict']['ends'])):
                    x_start = patch_origin[0]
                    x_end = patch_end[0]
                    y_start = patch_origin[1]
                    y_end = patch_end[1]
                    z_start = patch_origin[2]
                    z_end = patch_end[2]

                    # Convert (dummy) tensors to scalars
                    origin_list = (patch_origin[0].item(), patch_origin[1].item(), patch_origin[2].item())

                    patch = images[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                    patch_hat = images_hat[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                    mask_patch = mask[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                    mask_patch_hat = mask_hat[:, :, x_start:x_end, y_start:y_end, z_start:z_end]

                    _, _, px, py, pz = patch.shape
                    excess_x = 128-px
                    excess_y = 128-py
                    excess_z = 128-pz
                    patch = F.pad(patch,
                                  pad=(0, excess_z,
                                       0, excess_y,
                                       0, excess_x),
                                   mode='constant')

                    patch_hat = F.pad(patch_hat,
                                     pad=(0, excess_z,
                                          0, excess_y,
                                          0, excess_x),
                                     mode='constant')

                    patches_cat = torch.cat([patch, patch_hat], dim=1)


                    print('Processing {}/{} patches'.format(idx+1, n_patches))

                    unet_outputs = model.get_unet_outputs(patches_cat.to(device))

                    kpts_logits_1 = unet_outputs['kpts_logits_1']
                    kpts_logits_2 = unet_outputs['kpts_logits_2']
                    features_1_low = unet_outputs['features_1_low']
                    features_1_high = unet_outputs['features_1_high']
                    features_2_low = unet_outputs['features_2_low']
                    features_2_high = unet_outputs['features_2_high']



                    features_1 = (features_1_low.to(device), features_1_high.to(device))
                    features_2 = (features_2_low.to(device), features_2_high.to(device))
                    try:
                        if args.soft_masking is False:
                            outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                                      kpts_2=kpts_logits_2.to(device),
                                                      features_1=features_1,
                                                      features_2=features_2,
                                                      conf_thresh=args.conf_threshold,
                                                      num_pts=args.kpts_per_batch,
                                                      mask=mask_patch.to(device),
                                                      mask2=mask_patch_hat.to(device),
                                                      soft_masking=False)
                        else:
                            outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                                      kpts_2=kpts_logits_2.to(device),
                                                      features_1=features_1,
                                                      features_2=features_2,
                                                      conf_thresh=args.conf_threshold,
                                                      num_pts=args.kpts_per_batch,
                                                      mask=None,
                                                      mask2=None,
                                                      soft_masking=True)

                    except RuntimeError as e:
                        print(e)
                        continue
                    patch_matches = outputs['matches'][0]
                    patch_landmarks_fixed = outputs['landmarks_2'][0]
                    patch_landmarks_moving = outputs['landmarks_1'][0]

                    patch_landmarks_fixed_corrected, patch_landmarks_moving_corrected = \
                            create_corresponding_landmark_arrays(landmarks_fixed=patch_landmarks_fixed,
                                                                 landmarks_moving=patch_landmarks_moving,
                                                                 matches=patch_matches,
                                                                 origin=origin_list)

                    if patch_landmarks_moving_corrected is None: # No pairs found
                        continue

                    if fixed_landmarks_np is None:
                        fixed_landmarks_np = patch_landmarks_fixed_corrected
                    else:
                        fixed_landmarks_np = np.concatenate((fixed_landmarks_np,
                                                             patch_landmarks_fixed_corrected),
                                                            axis=0)

                    if moving_landmarks_np is None:
                        moving_landmarks_np = patch_landmarks_moving_corrected
                    else:
                        moving_landmarks_np = np.concatenate((moving_landmarks_np,
                                                             patch_landmarks_moving_corrected),
                                                             axis=0)



                print('Patient {} :: Number of predicted corresponding landmarks = {}'.format(batch_data['patient_id'][0],
                                                                                              fixed_landmarks_np.shape[0]))

                # Save the landmarks in an elastix compliant fashion (in world coordinates)
                save_landmark_predictions_in_elastix_format(landmarks_fixed=fixed_landmarks_np,
                                                            landmarks_moving=moving_landmarks_np,
                                                            metadata_fixed=fixed_metadata_list[0],
                                                            metadata_moving=moving_metadata_list[0],
                                                            matches=None,
                                                            save_dir=out_dir)
                save_ras_as_itk(img=images[0, ...].float(),
                                metadata=moving_metadata_list[0],
                                fname=os.path.join(out_dir, 'moving_image.mha'))

                save_ras_as_itk(img=images_hat[0, ...].float(),
                                metadata=fixed_metadata_list[0],
                                fname=os.path.join(out_dir, 'fixed_image.mha'))








if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--loss_mode', type=str, default='aux')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--soft_masking', action='store_true')
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--desc_length', type=int, default=-1)
    parser.add_argument('--kpts_per_batch', type=int, default=512)

    args = parser.parse_args()

    test(args)
