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
from visualize import *
from model import LesionMatchingModel
from deformations import *
from datapipeline import *
from loss import create_ground_truth_correspondences
from metrics import get_match_statistics
import shutil
import numpy as np
import random

ROI_SIZE = (128, 128, 64)


def test(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    checkpoint_dir = args.checkpoint_dir

    save_dir  = os.path.join(checkpoint_dir, args.out_dir)
    if os.path.exists(save_dir) is True:
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up data pipeline
    if args.mode == 'val':
        patients = joblib.load('val_patients.pkl')
    elif args.mode == 'test':
        patients = joblib.load('test_patients.pkl')
    elif args.mode == 'train':
        patient = joblib.load('train_patients.pkl')

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
    model = LesionMatchingModel(W=args.window_size,
                                K=args.kpts_per_batch)

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


                # Pad image in the k-direction to make the shape [256, 256, 256]
                # Makes it easier to scale deformation grid size to ensure the same
                # control point spacing for patches and full images
                b, c, i, j, k = images.shape
                pad = 256 - k
                images = F.pad(images, (0, pad), "constant", 0)
                liver_mask = F.pad(liver_mask, (0, pad), "constant", 0)

                deform_grid_multiplier = [i//ROI_SIZE[0], j//ROI_SIZE[1], k//ROI_SIZE[2]]


                batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                       non_rigid=True,
                                                                       coarse=True,
                                                                       fine=args.fine_deform,
                                                                       coarse_displacements=(2, 4, 4),
                                                                       fine_displacements=(1, 2, 2),
                                                                       coarse_grid_resolution=(4*deform_grid_multiplier[2],
                                                                                               4*deform_grid_multiplier[1],
                                                                                               4*deform_grid_multiplier[0]),
                                                                       fine_grid_resolution=(8*deform_grid_multiplier[2],
                                                                                             8*deform_grid_multiplier[1],
                                                                                             8*deform_grid_multiplier[0]))

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

                liver_mask_hat = F.grid_sample(input=liver_mask,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="nearest")


                # Concatenate along channel axis so that sliding_window_inference can
                # be used
                assert(images_hat.shape == images.shape)
                images_cat = torch.cat([images, images_hat], dim=1)


                # Keypoint logits
                kpts_logits_1, kpts_logits_2 = sliding_window_inference(inputs=images_cat.to(device),
                                                                        roi_size=ROI_SIZE,
                                                                        sw_device=device,
                                                                        device='cpu',
                                                                        sw_batch_size=4,
                                                                        predictor=model.get_patch_keypoint_scores,
                                                                        overlap=0.5)

                # Feature maps
                features_1_low, features_1_high, features_2_low, features_2_high =\
                                                        sliding_window_inference(inputs=images_cat.to(device),
                                                                                 roi_size=ROI_SIZE,
                                                                                 sw_batch_size=4,
                                                                                 sw_device=device,
                                                                                 device='cpu',
                                                                                 predictor=model.get_patch_feature_descriptors,
                                                                                 overlap=0.5)

                features_1 = (features_1_low.to(device), features_1_high.to(device))
                features_2 = (features_2_low.to(device), features_1_high.to(device))

                # Get (predicted) landmarks and matches on the full image
                # These landmarks are predicted based on L2-norm between feature descriptors
                # and predicted matching probability
                outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                          kpts_2=kpts_logits_2.to(device),
                                          features_1=features_1,
                                          features_2=features_2,
                                          conf_thresh=0.5,
                                          num_pts=args.kpts_per_batch,
                                          mask=liver_mask.to(device),
                                          mask2=liver_mask_hat.to(device))

                # Get ground truth matches based on projecting keypoints using the deformation grid
                gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                           kpts2=outputs['kpt_sampling_grid_2'],
                                                                                           deformation=batch_deformation_grid,
                                                                                           pixel_thresh=(2, 4, 4))

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

                    patient_id = batch_data['patient_id'][batch_id]
                    scan_id = batch_data['scan_id'][batch_id]
                    dump_dir = os.path.join(save_dir, patient_id, scan_id)
                    os.makedirs(dump_dir)

                    # Save the matrices/tensors for later analysis
                    np.save(file=os.path.join(dump_dir, 'gt_matches'),
                            arr=maybe_convert_tensor_to_numpy(batch_gt_matches))

                    np.save(file=os.path.join(dump_dir, 'pred_matches'),
                            arr=maybe_convert_tensor_to_numpy(batch_pred_matches))

                    np.save(file=os.path.join(dump_dir, 'pred_matches_norm'),
                            arr=maybe_convert_tensor_to_numpy(batch_pred_matches_norm))

                    np.save(file=os.path.join(dump_dir, 'pred_matches_prob'),
                            arr=maybe_convert_tensor_to_numpy(batch_pred_matches_prob))

                    np.save(file=os.path.join(dump_dir, 'landmarks_original'),
                            arr=maybe_convert_tensor_to_numpy(outputs['landmarks_1'][batch_id, ...]))

                    np.save(file=os.path.join(dump_dir, 'landmarks_deformed'),
                            arr=maybe_convert_tensor_to_numpy(outputs['landmarks_2'][batch_id, ...]))

                    # Visualize keypoint matches
                    visualize_keypoints_3d(im1=batch_data['image'][batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches'),
                                           neighbourhood=3)

                    visualize_keypoints_3d(im1=batch_data['image'][batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches_norm'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches_l2_norm'),
                                           neighbourhood=3)

                    visualize_keypoints_3d(im1=batch_data['image'][batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches_prob'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches_prob'),
                                           neighbourhood=3)

                    # Save images/KP heatmaps
                    write_image_to_file(image_array=torch.sigmoid(kpts_logits_1[batch_id, ...].squeeze(dim=0)),
                                        affine=batch_data['image_meta_dict']['affine'][batch_id],
                                        metadata_dict=batch_data['image_meta_dict'],
                                        filename=os.path.join(dump_dir, 'kpts_prob.nii.gz'))

                    write_image_to_file(image_array=torch.sigmoid(kpts_logits_2[batch_id, ...].squeeze(dim=0)),
                                        affine=batch_data['image_meta_dict']['affine'][batch_id],
                                        metadata_dict=batch_data['image_meta_dict'],
                                        filename=os.path.join(dump_dir, 'kpts_prob_deformed.nii.gz'))


                    write_image_to_file(image_array=batch_data['image'][batch_id].squeeze(dim=0),
                                        affine=batch_data['image_meta_dict']['affine'][batch_id],
                                        metadata_dict=batch_data['image_meta_dict'],
                                        filename=os.path.join(dump_dir, 'image.nii.gz'))

                    write_image_to_file(image_array=images_hat[batch_id, ...].squeeze(dim=0),
                                        affine=batch_data['image_meta_dict']['affine'][batch_id],
                                        metadata_dict=batch_data['image_meta_dict'],
                                        filename=os.path.join(dump_dir, 'd_image.nii.gz'))




            else: #TODO
                pass



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='saved_outputs')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--kpts_per_batch', type=int, default=512)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--fine_deform', action='store_true')
    parser.add_argument('--window_size', type=int, default=4)

    args = parser.parse_args()

    test(args)
