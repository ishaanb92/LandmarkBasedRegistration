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
    set_determinism(seed=args.seed)

    # Set up data pipeline
    if args.mode == 'val':
        patients = joblib.load('val_patients_{}.pkl'.format(args.dataset))
    elif args.mode == 'train':
        patients = joblib.load('train_patients_{}.pkl'.format(args.dataset))
    elif args.mode == 'all':
        assert(args.dataset == 'dirlab')
        patients = joblib.load('train_patients_{}.pkl'.format(args.dataset))
        patients.extend(joblib.load('val_patients_{}.pkl'.format(args.dataset)))

    elif args.mode == 'test':
        if args.dataset == 'umc':
            patients = joblib.load('test_patients_{}.pkl'.format(args.dataset))
        elif args.dataset == 'dirlab':
            raise ValueError('DIR-Lab does not have a test set')
        elif args.dataset == 'copd':
            patients = [f.path for f in os.scandir(COPD_DIR) if f.is_dir()]

    if args.synthetic is True:
        if args.dataset == 'umc':
            data_dicts = create_data_dicts_lesion_matching(patients)
            data_loader, _ = create_dataloader_lesion_matching(data_dicts=data_dicts,
                                                              train=False,
                                                              batch_size=args.batch_size,
                                                              num_workers=4,
                                                              data_aug=False)
        elif args.dataset == 'dirlab' or args.dataset =='copd':
            data_dicts = create_data_dicts_dir_lab(patients,
                                                   dataset=args.dataset)

            data_loader = create_dataloader_dir_lab(data_dicts=data_dicts,
                                                    batch_size=args.batch_size,
                                                    num_workers=4,
                                                    data_aug=False,
                                                    test=True)


    else: # "Real" data
        if args.dataset == 'umc':
            data_dicts = create_data_dicts_lesion_matching_inference(patients)
            data_loader, _ = create_dataloader_lesion_matching_inference(data_dicts=data_dicts,
                                                                         batch_size=args.batch_size,
                                                                         num_workers=4)
        elif args.dataset == 'dirlab' or args.dataset == 'copd':

            data_dicts = create_data_dicts_dir_lab_paired(patients,
                                                          dataset=args.dataset,
                                                          affine_reg_dir=args.affine_reg_dir)

            data_loader = create_dataloader_dir_lab_paired(data_dicts=data_dicts,
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


    if args.synthetic is True:
        if args.dataset == 'umc':
            coarse_displacements = (4, 8, 8)
            coarse_grid_resolution = (3, 3, 3)
            fine_displacements = (2, 4, 4)
            fine_grid_resolution = (6, 6, 6)
        elif args.dataset == 'dirlab' or args.dataset == 'copd':
            coarse_displacements = (29, 19.84, 9.92)
            fine_displacements = (7.25, 9.92, 9.92)
            coarse_grid_resolution = (2, 2, 2)
            fine_grid_resolution = (3, 3, 3)

    if args.dataset == 'umc':
        roi_size = (128, 128, 64)
        neighbourhood = 3
        pixel_thresh = (2, 4, 4)
    elif args.dataset == 'dirlab' or args.dataset == 'copd':
        roi_size = (128, 128, 96)
        neighbourhood = 10
        pixel_thresh = (1, 2, 2)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if args.synthetic is True:
                # Based on how torch.nn.grid_sample is defined =>
                # Moving image: images
                # Fixed image: images_hat
                if args.dataset == 'umc':
                    images, mask, vessel_mask = (batch_data['image'], batch_data['liver_mask'], batch_data['vessel_mask'])
                elif args.dataset == 'dirlab' or args.dataset == 'copd':
                    images, mask = (batch_data['image'], batch_data['lung_mask'])
                    metadata_list = detensorize_metadata(metadata=batch_data['metadata'],
                                                         batchsz=images.shape[0])

                b, c, i, j, k = images.shape

                # NOTE: This multiplier is used to ensure equivalent deformations for patches and full images.
                # This is done by fixing min and max displacements and ensuring the spacing between control points is the
                # same regardless of the dimensions of the patch. When working with full images, therefore, the control point grid
                # is made "denser" to account for the increase in image size
                deform_grid_multiplier = [floor(i/roi_size[0]+0.5),
                                          floor(j/roi_size[1]+0.5),
                                          floor(k/roi_size[2]+0.5)]

                batch_deformation_grid = \
                    create_batch_deformation_grid(shape=images.shape,
                                                  non_rigid=True,
                                                  coarse=True,
                                                  fine=True,
                                                  coarse_displacements=coarse_displacements,
                                                  fine_displacements=fine_displacements,
                                                  coarse_grid_resolution=(coarse_grid_resolution[2]*deform_grid_multiplier[2],
                                                                          coarse_grid_resolution[1]*deform_grid_multiplier[1],
                                                                          coarse_grid_resolution[0]*deform_grid_multiplier[0]),
                                                 fine_grid_resolution=(fine_grid_resolution[2]*deform_grid_multiplier[2],
                                                                       fine_grid_resolution[1]*deform_grid_multiplier[1],
                                                                       fine_grid_resolution[0]*deform_grid_multiplier[0]))

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

                mask_hat = F.grid_sample(input=mask,
                                         grid=batch_deformation_grid,
                                         align_corners=True,
                                         mode="nearest")

                # Pad images and masks to make dims divisible by 8
                # See: https://docs.monai.io/en/stable/inferers.html#monai.inferers.sliding_window_inference
                if args.dataset == 'umc':
                    excess_pixels_xy = 0
                    excess_pixels_z = 8 - (k%8)
                elif args.dataset == 'dirlab' or args.dataset == 'copd':
                    if i%8 != 0 and j%8 !=0:
                        excess_pixels_xy = 8 - (i%8)
                    else:
                        excess_pixels_xy = 0

                    if k%8 != 0:
                        excess_pixels_z = 8 - (k%8)
                    else:
                        excess_pixels_z = 0

                images = F.pad(images,
                               pad=(0, excess_pixels_z,
                                    0, excess_pixels_xy,
                                    0, excess_pixels_xy),
                               mode='constant')

                images_hat = F.pad(images_hat,
                               pad=(0, excess_pixels_z,
                                    0, excess_pixels_xy,
                                    0, excess_pixels_xy),
                               mode='constant')

                mask = F.pad(mask,
                             pad=(0, excess_pixels_z,
                                  0, excess_pixels_xy,
                                  0, excess_pixels_xy),
                            mode='constant')

                mask_hat = F.pad(mask_hat,
                             pad=(0, excess_pixels_z,
                                  0, excess_pixels_xy,
                                  0, excess_pixels_xy),
                            mode='constant')


                # Concatenate along channel axis so that sliding_window_inference can
                # be used
                assert(images_hat.shape == images.shape)
                images_cat = torch.cat([images, images_hat], dim=1)


                print('Image shape after padding = {}'.format(images.shape))
                print('Mask shape after padding = {}'.format(mask.shape))

                # U-Net outputs via patch-based inference
                try:
                    unet_outputs = sliding_window_inference(inputs=images_cat.to(device),
                                                            roi_size=roi_size,
                                                            sw_device=device,
                                                            device='cpu',
                                                            sw_batch_size=2,
                                                            predictor=model.get_unet_outputs,
                                                            overlap=0.25,
                                                            progress=True)
                except RuntimeError:
                    unet_outputs = sliding_window_inference(inputs=images_cat.to(device),
                                                            roi_size=roi_size,
                                                            sw_device=device,
                                                            device='cpu',
                                                            sw_batch_size=1,
                                                            predictor=model.get_unet_outputs,
                                                            overlap=0.25,
                                                            progress=True)

                kpts_logits_1 = unet_outputs['kpts_logits_1']
                kpts_logits_2 = unet_outputs['kpts_logits_2']
                features_1_low = unet_outputs['features_1_low']
                features_1_high = unet_outputs['features_1_high']
                features_2_low = unet_outputs['features_2_low']
                features_2_high = unet_outputs['features_2_high']



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
                                          mask=mask.to(device),
                                          mask2=mask_hat.to(device),
                                          mode=args.loss_mode)

                # Get ground truth matches based on projecting keypoints using the deformation grid
                gt1, gt2, gt_matches, num_gt_matches, projected_landmarks = \
                                    create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                        kpts2=outputs['kpt_sampling_grid_2'],
                                                                        deformation=batch_deformation_grid,
                                                                        pixel_thresh=pixel_thresh,
                                                                        train=False)

                print('Number of ground truth matches (based on projecting keypoints) = {}'.format(num_gt_matches))
                print('Number of matches based on feature descriptor distance '
                      '& matching probability = {}'.format(torch.nonzero(outputs['matches']).shape[0]))

                # Get rid of image and mask padding
                if args.dataset == 'dirlab':
                    images = images[:, :, :i, :j, :k]
                    images_hat = images_hat[:, :, :i, :j, :k]
                    mask = mask[:, :, :i, :j, :k]
                    mask_hat = mask_hat[:, :, :i, :j, :k]

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
                    if args.dataset == 'umc':
                        i_type = batch_data['scan_id'][batch_id]
                    elif args.dataset == 'dirlab':
                        i_type = batch_data['type'][batch_id]

                    dump_dir = os.path.join(save_dir, patient_id, i_type)

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

                    np.save(file=os.path.join(dump_dir, 'landmarks_projected'),
                            arr=maybe_convert_tensor_to_numpy(projected_landmarks[batch_id, ...]))

                    # Visualize keypoint matches
                    visualize_keypoints_3d(im1=images[batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches'),
                                           neighbourhood=neighbourhood)

                    visualize_keypoints_3d(im1=images[batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches_norm'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches_l2_norm'),
                                           neighbourhood=neighbourhood)

                    visualize_keypoints_3d(im1=images[batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches_prob'][batch_id, ...],
                                           gt_matches=batch_gt_matches,
                                           out_dir=os.path.join(dump_dir, 'matches_prob'),
                                           neighbourhood=neighbourhood)

                    # Save images/KP heatmaps
                    if args.dataset == 'umc':
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
                    elif args.dataset == 'dirlab':
                        save_ras_as_itk(img=images[batch_id, ...],
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'image.mha'))

                        save_ras_as_itk(img=images_hat[batch_id, ...],
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'd_image.mha'))

                        save_ras_as_itk(img=mask[batch_id, ...],
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'mask.mha'))

                        save_ras_as_itk(img=mask_hat[batch_id, ...],
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'd_mask.mha'))

                        save_ras_as_itk(img=torch.sigmoid(kpts_logits_1[batch_id, ...].squeeze(dim=0)),
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'kpts_prob.mha'))

                        save_ras_as_itk(img=torch.sigmoid(kpts_logits_2[batch_id, ...].squeeze(dim=0)),
                                        metadata=metadata_list[batch_id],
                                        fname=os.path.join(dump_dir, 'kpts_prob_deformed.mha'))


            else: # Paired data
                if args.dataset == 'umc':
                    raise NotImplementedError('Paired landmark matching not yet implemented for UMC dataset')

                elif args.dataset == 'dirlab' or args.dataset == 'copd':
                    images, images_hat, mask, mask_hat = (batch_data['moving_image'], batch_data['fixed_image'],\
                                                          batch_data['moving_lung_mask'], batch_data['fixed_lung_mask'])

                    fixed_metadata_list = detensorize_metadata(metadata=batch_data['fixed_metadata'],
                                                               batchsz=images.shape[0])

                    moving_metadata_list = detensorize_metadata(metadata=batch_data['moving_metadata'],
                                                                batchsz=images.shape[0])

                    assert(images.shape == images_hat.shape)

                    b, c, i, j, k = images.shape

                    # Pad images and masks to make dims divisible by 8
                    # See: https://docs.monai.io/en/stable/inferers.html#monai.inferers.sliding_window_inference
                    if i%8 != 0 and j%8 !=0:
                        excess_pixels_xy = 8 - (i%8)
                    else:
                        excess_pixels_xy = 0

                    if k%8 != 0:
                        excess_pixels_z = 8 - (k%8)
                    else:
                        excess_pixels_z = 0

                    images = F.pad(images,
                                   pad=(0, excess_pixels_z,
                                        0, excess_pixels_xy,
                                        0, excess_pixels_xy),
                                   mode='constant')

                    images_hat = F.pad(images_hat,
                                   pad=(0, excess_pixels_z,
                                        0, excess_pixels_xy,
                                        0, excess_pixels_xy),
                                   mode='constant')

                    mask = F.pad(mask,
                                 pad=(0, excess_pixels_z,
                                      0, excess_pixels_xy,
                                      0, excess_pixels_xy),
                                mode='constant')

                    mask_hat = F.pad(mask_hat,
                                 pad=(0, excess_pixels_z,
                                      0, excess_pixels_xy,
                                      0, excess_pixels_xy),
                                mode='constant')


                images_cat = torch.cat([images, images_hat], dim=1)

                b, c, i, j, k = images.shape

                # U-Net outputs via patch-based inference
                unet_outputs = sliding_window_inference(inputs=images_cat.to(device),
                                                        roi_size=roi_size,
                                                        sw_device=device,
                                                        device='cpu',
                                                        sw_batch_size=1,
                                                        predictor=model.get_unet_outputs,
                                                        overlap=0.25,
                                                        progress=True)

                kpts_logits_1 = unet_outputs['kpts_logits_1']
                kpts_logits_2 = unet_outputs['kpts_logits_2']
                features_1_low = unet_outputs['features_1_low']
                features_1_high = unet_outputs['features_1_high']
                features_2_low = unet_outputs['features_2_low']
                features_2_high = unet_outputs['features_2_high']



                features_1 = (features_1_low.to(device), features_1_high.to(device))
                features_2 = (features_2_low.to(device), features_1_high.to(device))

                print(torch.max(torch.max(kpts_logits_1)))
                print(torch.max(torch.max(kpts_logits_2)))

                try:
                    outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                              kpts_2=kpts_logits_2.to(device),
                                              features_1=features_1,
                                              features_2=features_2,
                                              conf_thresh=args.conf_threshold,
                                              num_pts=args.kpts_per_batch,
                                              mask=mask.to(device),
                                              mask2=mask_hat.to(device),
                                              mode=args.loss_mode)
                except RuntimeError:
                    continue

                # How many matches predicted between paired (affinely registered) images?
                for batch_id in range(b):
                    matches = maybe_convert_tensor_to_numpy(outputs['matches'][batch_id, ...])
                    print('Patient {} :: Number of predicted corresponding landmarks = {}'.format(batch_data['patient_id'][batch_id],
                                                                                                  np.nonzero(matches)[0].shape[0]))

                    match_probabilities = maybe_convert_tensor_to_numpy(outputs['match_probabilities'][batch_id, ...])

                    patient_id = batch_data['patient_id'][batch_id]
                    dump_dir = os.path.join(save_dir, patient_id)
                    os.makedirs(dump_dir)

                    np.save(file=os.path.join(dump_dir, 'predicted_matches'),
                            arr=matches)

                    np.save(file=os.path.join(dump_dir, 'match_probabilities'),
                            arr=match_probabilities)

                    # Landmarks are saved in k-j-i order!
                    np.save(file=os.path.join(dump_dir, 'landmarks_fixed'),
                            arr=maybe_convert_tensor_to_numpy(outputs['landmarks_2'][batch_id, ...]))

                    np.save(file=os.path.join(dump_dir, 'landmarks_moving'),
                            arr=maybe_convert_tensor_to_numpy(outputs['landmarks_1'][batch_id, ...]))

                    # Save corr. landmarks as elastix-compatible .txt files (in i-j-k order)
                    save_landmark_predictions_in_elastix_format(landmarks_fixed=outputs['landmarks_2'][batch_id, ...],
                                                                landmarks_moving=outputs['landmarks_1'][batch_id, ...],
                                                                metadata_fixed=fixed_metadata_list[batch_id],
                                                                metadata_moving=moving_metadata_list[batch_id],
                                                                matches=matches,
                                                                save_dir=dump_dir)


                    visualize_keypoints_3d(im1=images[batch_id, ...].squeeze(dim=0),
                                           im2=images_hat[batch_id, ...].squeeze(dim=0),
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches'][batch_id, ...],
                                           gt_matches=None,
                                           out_dir=os.path.join(dump_dir, 'matches'),
                                           neighbourhood=neighbourhood)

                    # Save images
                    save_ras_as_itk(img=images[batch_id, ...].float(),
                                    metadata=moving_metadata_list[batch_id],
                                    fname=os.path.join(dump_dir, 'moving_image.mha'))

                    save_ras_as_itk(img=images_hat[batch_id, ...].float(),
                                    metadata=fixed_metadata_list[batch_id],
                                    fname=os.path.join(dump_dir, 'fixed_image.mha'))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--affine_reg_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='saved_outputs')
    parser.add_argument('--loss_mode', type=str, default='aux')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--kpts_per_batch', type=int, default=512)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--conf_threshold', type=float, default=0.5)

    args = parser.parse_args()

    test(args)
