"""

Script to evaluate trained models (with visualization)

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
from utils.utils import *
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'arch'))
from model import LesionMatchingModel
from deformations import *
from datapipeline import *


def test(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    checkpoint_dir = args.checkpoint_dir

    # Set up data pipeline
    if args.mode == 'val':
        patients = joblib.load('val_patients.pkl')
    elif args.mode == 'test':
        patients = joblib.load('test_patients.pkl')


    if args.synthetic is True:
        data_dicts = create_data_dicts_lesion_matching(patients)
        data_loader, _ = create_dataloader_lesion_matching(data_dicts=data_dicts,
                                                          train=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=4,
                                                          data_aug=False)
    else: # "Real" data
        data_dicts = create_data_dicts_lesion_matching_inference(patients)
        data_loader, _ = create_dataloader_lesion_matching_inference(data_dicts=data_dicts,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=4)


    # Define the model
    model = LesionMatchingModel(W=4)

    # Load the model
    load_dict = load_model(model=model,
                           checkpoint_dir=checkpoint_dir,
                           training=False)

    model = load_dict['model']

    model.to(device)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if args.synthetic is True:

                images, liver_mask, vessel_mask = (batch_data['image'], batch_data['liver_mask'], batch_data['vessel_mask'])

                batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                       device=images.device,
                                                                       dummy=True)

                if batch_deformation_grid is None:
                    continue

                # FIXME: Change mode when debug done
                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="nearest")



                # DEBUG: REMOVEEEE!!!!
                assert(torch.equal(images, images_hat))

                # Concatenate along channel axis so that sliding_window_inference can
                # be used
                assert(images_hat.shape == images.shape)
                images_cat = torch.cat([images, images_hat], dim=1)


                # Pad so that sliding window inference does not complain
                # about non-integer output shapes
                depth = images_cat.shape[-1]
                pad = 256-depth

                images_cat = F.pad(images_cat, (0, pad), "constant", 0)
                liver_mask = F.pad(liver_mask, (0, pad), "constant", 0)


                kpts_1, kpts_2 = sliding_window_inference(inputs=images_cat.to(device),
                                                          roi_size=(128, 128, 64),
                                                          sw_batch_size=2,
                                                          predictor=model.get_patch_keypoint_scores,
                                                          overlap=0.5)


                # Mask using liver mask
                kpts_1 = kpts_1*liver_mask.to(kpts_1.device)
                kpts_2 = kpts_2*liver_mask.to(kpts_2.device)

                features_1_low, features_1_high, features_2_low, features_2_high =\
                                                        sliding_window_inference(inputs=images_cat.to(device),
                                                                                 roi_size=(128, 128, 64),
                                                                                 sw_batch_size=2,
                                                                                 predictor=model.get_patch_feature_descriptors,
                                                                                 overlap=0.0)

                features_1 = (features_1_low, features_1_high)
                features_2 = (features_2_low, features_1_high)

                # Get landmarks and matches on the full image
                landmarks_1, landmarks_2, matches = model.inference(kpts_1=kpts_1,
                                                                    kpts_2=kpts_2,
                                                                    features_1=features_1,
                                                                    features_2=features_2)

                match_idxs = torch.nonzero(matches)
                print(match_idxs.shape)

            else: #TODO
                pass


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--synthetic', action='store_true')

    args = parser.parse_args()

    test(args)
