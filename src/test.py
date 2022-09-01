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
from loss import create_ground_truth_correspondences
from metrics import get_match_statistics

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
                                                                       dummy=args.dummy,
                                                                       coarse_displacements=(6, 3, 3),
                                                                       fine_displacements=(2, 2, 2))

                if batch_deformation_grid is None:
                    continue

                if args.dummy is True:
                    images_hat = F.grid_sample(input=images,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="nearest")

                    assert(torch.equal(images, images_hat))
                else:
                    images_hat = F.grid_sample(input=images,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="bilinear")


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


                # Keypoint logits
                kpts_1, kpts_2 = sliding_window_inference(inputs=images_cat.to(device),
                                                          roi_size=(128, 128, 64),
                                                          sw_batch_size=2,
                                                          predictor=model.get_patch_keypoint_scores,
                                                          overlap=0.5)


                # Mask using liver mask
                kpts_1 = kpts_1*liver_mask.to(kpts_1.device)
                kpts_2 = kpts_2*liver_mask.to(kpts_2.device)

                # Feature maps
                features_1_low, features_1_high, features_2_low, features_2_high =\
                                                        sliding_window_inference(inputs=images_cat.to(device),
                                                                                 roi_size=(128, 128, 64),
                                                                                 sw_batch_size=2,
                                                                                 predictor=model.get_patch_feature_descriptors,
                                                                                 overlap=0.5)

                features_1 = (features_1_low, features_1_high)
                features_2 = (features_2_low, features_1_high)



                # Get (predicted) landmarks and matches on the full image
                # These landmarks are predicted based on L2-norm between feature descriptors
                # and predicted matching probability
                outputs = model.inference(kpts_1=kpts_1,
                                          kpts_2=kpts_2,
                                          features_1=features_1,
                                          features_2=features_2,
                                          conf_thresh=0.5)

                # Get ground truth matches based on projecting keypoints using the deformation grid
                gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                           kpts2=outputs['kpt_sampling_grid_2'],
                                                                                           deformation=batch_deformation_grid,
                                                                                           pixel_thresh=5)

                print('Number of ground truth matches (based on projecting keypoints) = {}'.format(num_gt_matches))
                print('Number of matches based on feature descriptor distance '
                      '& matching probability = {}'.format(torch.nonzero(outputs['matches']).shape[0]))

                # Get TP, FP, FN matches
                for batch_id in range(gt_matches.shape[0]):
                    batch_gt_matches = gt_matches[batch_id, ...] # Shape: (K, K)
                    batch_pred_matches = outputs['matches'][batch_id, ...] # Shape (K, K)
                    batch_pred_matches_norm = outputs['matches_norm'][batch_id, ...] # Shape (K, K)
                    batch_pred_matches_prob = outputs['matches_prob'][batch_id, ...] # Shape (K, K)

                    stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                                 pred=batch_pred_matches.cpu())
                    print('Matches :: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                              stats['False Positives'],
                                                                              stats['False Negatives']))

                    stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                                 pred=batch_pred_matches_norm.cpu())
                    print('Matches w.r.t L2-Norm:: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                                           stats['False Positives'],
                                                                                           stats['False Negatives']))

                    stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                                 pred=batch_pred_matches_prob.cpu())
                    print('Matches w.r.t match probability:: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                                                     stats['False Positives'],
                                                                                                     stats['False Negatives']))

                    # TODO: Save outputs (images, landmarks points) for visualization


            else: #TODO
                pass


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dummy', action='store_true')

    args = parser.parse_args()

    test(args)
