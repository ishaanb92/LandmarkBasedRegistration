"""
Script to visualize keypoint matches

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import numpy as np
import pandas as pd
import cv2
from lesionmatching.util_scripts.utils import maybe_convert_tensor_to_numpy
import os
import math
import shutil

def save_matches(outputs,batch_idx):
    keypoints_detected = outputs['matches'][0, :, :]
    print(np.nonzero(keypoints_detected))
    indices=np.nonzero(keypoints_detected)

    pd.DataFrame(outputs['landmarks_1'][0, :, :].cpu().detach().numpy()).to_csv("landmarks1.csv", header=None,  index=None)
    pd.DataFrame(outputs['landmarks_2'][0, :, :].cpu().detach().numpy()).to_csv("landmarks2.csv", header=None,  index=None)
    pd.DataFrame(outputs['matches'][0, :, :].cpu().detach().numpy()).to_csv("matches.csv", index=None)

    kpts1=[]
    kpts2=[]

    for j in indices:
        # print(type(outputs['landmarks_1'][0, j[0].item(), :].tolist()))
        kpts1.append(outputs['landmarks_1'][0, j[0].item(), :].tolist())
        kpts2.append(outputs['landmarks_2'][0, j[1].item(), :].tolist())
        # print(outputs['landmarks_1'][0, j[0].item(), :].tolist())
        # print(outputs['landmarks_2'][0, j[1].item(), :].tolist())
    # print(outputs['landmarks_1'][0, :, :])
    #
    with open('img_{}matches1.txt'.format(batch_idx), 'w') as f:
        for line in kpts1:
            f.write(f"{line}\n")

    with open('img_{}matches2.txt'.format(batch_idx), 'w') as f:
        for line in kpts2:
            f.write(f"{line}\n")


def min_max_scaling(image):
    im_max = np.amax(image)
    im_min = np.amin(image)
    s_image = (image-im_min)/(im_max-im_min)
    return s_image

# HACK FUNCTION
def round_float_coords(coord):

    round_coord = math.floor(coord+0.5)
    return int(round_coord)


def visualize_keypoints_2d(im1, im2, output1, output2, pred_mask, gt_mask, out_dir, slice_id, basename="slice"):


    im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RVGB)

    im = np.concatenate([im1, im2],
                        axis=1)

    color = [0, 0, 1]

    for k1, l1 in enumerate(output1):
        kk1, jj1, ii1 = l1
        # FIXME: The kpt indices should not be float!
        kk1 = round_float_coords(kk1)
        jj1 = round_float_coords(jj1)
        ii1 = round_float_coords(ii1)

        cv2.circle(im, (jj1, ii1), 2, color, -1)

        for k2, l2 in enumerate(output2):
            kk2, jj2, ii2 = l2
            kk2 = round_float_coords(kk2)
            jj2 = round_float_coords(jj2)
            ii2 = round_float_coords(ii2)
            cv2.circle(im, (jj2+im1.shape[1], ii2), 2, color, -1)
            if pred_mask[k1, k2] == 1:
                cv2.line(im, (jj1, ii1), (jj2+im1.shape[1], ii2), (0, 0, 1), 1)
            if gt_mask[k1, k2] == 1:
                cv2.line(im, (jj1, ii1), (jj2+im1.shape[1], ii2), (0, 1, 0), 1)


    # Transpose the axes
    im = np.transpose(im, (1, 0, 2))
    cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(basename)),
                (im*255).astype(np.uint8))


def visualize_keypoints_3d(im1, im2, landmarks1, landmarks2, pred_matches, gt_matches, out_dir, neighbourhood=5, verbose=False):
    """
    Function to visualize (predicted and true) landmark correspondences.
    Wraps the 2-D visualization function from https://github.com/monikagrewal/End2EndLandmarks
    such that correspondences found on the same slice can be visualized.

    """
    im1 = maybe_convert_tensor_to_numpy(im1)
    im2 = maybe_convert_tensor_to_numpy(im2)

    # Shape: (K, 3)
    landmarks1 = maybe_convert_tensor_to_numpy(landmarks1)
    landmarks2 = maybe_convert_tensor_to_numpy(landmarks2)

    # Shape: (K, K)
    pred_matches = maybe_convert_tensor_to_numpy(pred_matches)

    # Shape: (K, K)
    if gt_matches is not None:
        gt_matches = maybe_convert_tensor_to_numpy(gt_matches)

    # Rescale images to 0, 1 range for display
#    im1 = min_max_scaling(im1)
#    im2 = min_max_scaling(im2)

    assert(im1.ndim == 3)
    assert(im2.ndim == 3)

    n_slices = im1.shape[-1]

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    for slice_idx in range(n_slices):

        # Avoid the slices at the edges!
        if slice_idx - neighbourhood < 0 or slice_idx + neighbourhood >= n_slices:
            continue

        im1_slice = im1[:, :, slice_idx]
        im2_patch = im2[:, :, slice_idx-neighbourhood:slice_idx+neighbourhood+1]

        # Filter (3-D) keypoints based on the slice "neighbourhood"
        # Get landmarks on the current slice for 'image'
        slice_landmarks1_rows = np.where(landmarks1[:, 0] == slice_idx)[0]

        patch_landmarks2_rows = np.where((landmarks2[:, 0] >= slice_idx - neighbourhood) &
                                         (landmarks2[:, 0] <= slice_idx + neighbourhood))[0]


        n_landmarks1 = slice_landmarks1_rows.shape[0]
        n_landmarks2 = patch_landmarks2_rows.shape[0]

        if n_landmarks1 == 0 or n_landmarks2 == 0:
            continue

        # Get corresponding keypoints and matches for the current slice
        slice_landmarks1 = landmarks1[slice_landmarks1_rows, :]
        patch_landmarks2 = landmarks2[patch_landmarks2_rows, :]


        # Create per-slice match matrices (for pred and GT)
        slice_pred_matches = np.zeros((n_landmarks1, n_landmarks2),
                                       dtype=np.uint8)

        if gt_matches is not None:
            slice_gt_matches = np.zeros((n_landmarks1, n_landmarks2),
                                         dtype=np.uint8)
        else:
            slice_gt_matches = None


        # Create per-slice "match" matrices for prediction and GT
        for i in range(n_landmarks1):
            for j in range(n_landmarks2):
                if pred_matches[slice_landmarks1_rows[i], patch_landmarks2_rows[j]] == 1:
                    slice_pred_matches[i, j] = 1

                if gt_matches is not None:
                    if gt_matches[slice_landmarks1_rows[i], patch_landmarks2_rows[j]] == 1:
                        slice_gt_matches[i, j] = 1

        n_matches = np.where(slice_pred_matches==1)[0].shape[0]

        if gt_matches is not None:
            n_gt_matches = np.where(slice_gt_matches==1)[0].shape[0]
        else:
            n_gt_matches = -1

        if verbose is True:
            print('For slice {}, keypoints found in original = {}, deformed image = {},\
                  pred matches = {} gt matches = {}'.format(slice_idx,
                                                            n_landmarks1,
                                                            n_landmarks2,
                                                            n_matches,
                                                            n_gt_matches))

        visualize_keypoints_2d_neighbourhood(im1=im1_slice,
                                             im2=im2_patch,
                                             output1=slice_landmarks1,
                                             output2=patch_landmarks2,
                                             pred_mask=slice_pred_matches,
                                             gt_mask=slice_gt_matches,
                                             slice_id=slice_idx,
                                             out_dir=out_dir,
                                             basename='slice_{}'.format(slice_idx))



def visualize_keypoints_2d_neighbourhood(im1,
                                         im2,
                                         output1,
                                         output2,
                                         pred_mask,
                                         gt_mask,
                                         out_dir,
                                         slice_id,
                                         basename="slice",
                                         synthetic=True):
    """
    Function to display matches for a given slice in the original image. To show matches that may occur across slices, we define this hybrid function where the we take only one slice from the original image, but fetch a corresponding "neighbourhood" of slices from the deformed image and display the matches

    """


    assert(im1.ndim == 2)
    assert(im2.ndim == 3)

    # k : size of neighbourhood
    i, j, k = im2.shape


    # np.ndarray to "hold" both images (and matches)
    im = np.zeros((i*k, 2*j),
                  dtype=np.float32)

    # Put the original image in the "center" block
    middle_block = (k-1)//2
    im[middle_block*i:(middle_block+1)*i, :j] = im1

    for slice_idx in range(k):
        im2_slice = im2[:, :, slice_idx]
        im[slice_idx*i:(slice_idx+1)*i, j:] = im2_slice

    # Convert the image to OpenCV RGB format
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    color = [0, 0, 1]

    if synthetic is True:
        for k1, l1 in enumerate(output1):
            kk1, jj1, ii1 = l1
            kk1 = round_float_coords(kk1)
            jj1 = round_float_coords(jj1)
            ii1 = round_float_coords(ii1)

            cv2.circle(im, (jj1, middle_block*i + ii1), 2, color, -1)

            for k2, l2 in enumerate(output2):
                kk2, jj2, ii2 = l2
                kk2 = round_float_coords(kk2)
                jj2 = round_float_coords(jj2)
                ii2 = round_float_coords(ii2)

                slice_diff = kk1 - kk2

                cv2.circle(im, (jj2+j, ((middle_block+slice_diff)*i + ii2)), 2, color, -1)

                if pred_mask[k1, k2] == 1:
                    if gt_mask is not None:
                        if gt_mask[k1, k2] == 1: # True positive
                            cv2.line(im, (jj1, middle_block*i + ii1), (jj2+j, (middle_block+slice_diff)*i + ii2), (0, 1, 0), 1) # Green
                        else: # False positive
                            cv2.line(im, (jj1, middle_block*i + ii1), (jj2+j, (middle_block+slice_diff)*i + ii2), (0, 1, 1), 1) # Yellow
                    else: # No GT mask (i.e. paired data, true deformation is unknown)
                        cv2.line(im, (jj1, middle_block*i + ii1), (jj2+j, (middle_block+slice_diff)*i + ii2), (0, 1, 0), 1) # Green

                if gt_mask is not None:
                    if gt_mask[k1, k2] == 1:
                        if pred_mask[k1, k2] == 0: # False negative
                            cv2.line(im, (jj1, middle_block*i + ii1), ((jj2+j, (middle_block+slice_diff)*i + ii2)), (0, 0, 1), 1) # Red
    else:
        raise NotImplementedError

    cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(basename)),
                (im*255).astype(np.uint8))


def overlay_predicted_and_manual_landmarks(fixed_image,
                                           moving_image,
                                           pred_landmarks_fixed,
                                           pred_landmarks_moving,
                                           manual_landmarks_fixed,
                                           manual_landmarks_moving,
                                           smoothed_landmarks_moving=None,
                                           gt_projection_landmarks_moving=None,
                                           out_dir=None,
                                           verbose=False):
    """
    To gain better insight into TRE trends, visualize the overlay of manual and predicted landmarks pairs

    Args:
        fixed_image: (numpy ndarray) RAS axes ordering
        moving_image: (numpy ndarray) RAS axes ordering
        pred_landmarks_fixed: (numpy ndarray) shape : (N, 3) -- in voxels
        pred_landmarks_moving: (numpy ndarray) shape: (N, 3) -- in voxels
        manual_landmarks_fixed: (numpy ndarray) shape: (M, 3) -- in voxels
        manual_landmarks_moving: (numpy ndarray) shape: (M, 3) -- in voxels
        out_dir: (str) Output directory to store all the images
    """


    if os.path.exists(out_dir) is True:
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    fixed_image = maybe_convert_tensor_to_numpy(fixed_image)
    moving_image = maybe_convert_tensor_to_numpy(moving_image)

    # Transpose fixed and moving images to Y-X-Z (non-RAS) orientation
    fixed_image = np.transpose(fixed_image, (1, 0, 2))
    moving_image = np.transpose(moving_image, (1, 0, 2))

    if np.min(fixed_image) < 0 and np.min(moving_image) < 0:
        fixed_image = ((fixed_image - np.min(fixed_image))/(np.max(fixed_image) - np.min(fixed_image)))
        moving_image = ((moving_image - np.min(moving_image))/(np.max(moving_image) - np.min(moving_image)))

    # Predicted landmarks
    pred_landmarks_fixed = maybe_convert_tensor_to_numpy(pred_landmarks_fixed)
    pred_landmarks_moving = maybe_convert_tensor_to_numpy(pred_landmarks_moving)

    # Manual landmarks
    if manual_landmarks_fixed is not None and manual_landmarks_moving is not None:
        manual_landmarks_fixed = maybe_convert_tensor_to_numpy(manual_landmarks_fixed)
        manual_landmarks_moving = maybe_convert_tensor_to_numpy(manual_landmarks_moving)
    else:
        manual_landmarks_fixed = None
        manual_landmarks_moving = None

    # Smoothed predicted moving landmarks
    if smoothed_landmarks_moving is not None:
        smoothed_landmarks_moving = maybe_convert_tensor_to_numpy(smoothed_landmarks_moving)

    # GT projection landmarks moving landmarks
    if gt_projection_landmarks_moving is not None:
        gt_projection_landmarks_moving = maybe_convert_tensor_to_numpy(gt_projection_landmarks_moving)

    n_slices = fixed_image.shape[-1]

    pred_color = [0, 0, 1]
    manual_color = [0, 1, 0]


    for slice_idx in range(n_slices):

        # Predicted landmarks
        pred_landmarks_fixed_rows = np.where(pred_landmarks_fixed[:, 2] == slice_idx)[0]
        n_pred_landmarks = pred_landmarks_fixed_rows.shape[0]

        if n_pred_landmarks != 0:
            # Fixed image landmarks on current slice
            slice_pred_landmarks_fixed = pred_landmarks_fixed[pred_landmarks_fixed_rows, :]

            # Use the same rows because matching landmarks are in corr. rows by definition!!
            # 1. Predicted landmarks in moving image
            patch_pred_landmarks_moving = pred_landmarks_moving[pred_landmarks_fixed_rows, :]
            pred_moving_slices = patch_pred_landmarks_moving[:, 2]
            pred_max_slice = np.amax(pred_moving_slices)
            pred_min_slice = np.amin(pred_moving_slices)

            # 2. Smoothed landmarks in moving image
            if smoothed_landmarks_moving is not None:
                patch_smoothed_landmarks_moving = smoothed_landmarks_moving[pred_landmarks_fixed_rows, :]
                smoothed_moving_slices = patch_smoothed_landmarks_moving[:, 2]
                smoothed_max_slice = np.amax(smoothed_moving_slices)
                smoothed_min_slice = np.amin(smoothed_moving_slices)
            else:
                smoothed_max_slice = -1
                smoothed_min_slice = 10000
                patch_smoothed_landmarks_moving = None
                smoothed_moving_slices = None

            # 3. GT projection in moving image
            if gt_projection_landmarks_moving is not None:
                patch_gt_projection_landmarks_moving = gt_projection_landmarks_moving[pred_landmarks_fixed_rows, :]
                gt_projection_moving_slices = patch_gt_projection_landmarks_moving[:, 2]
                gt_projection_max_slice = np.amax(gt_projection_moving_slices)
                gt_projection_min_slice = np.amin(gt_projection_moving_slices)
                # FIXME: GT projection does not work for copd9 because the TPS transform maps to a -ve slice
                if gt_projection_max_slice > n_slices or gt_projection_min_slice < 0:
                    gt_projection_max_slice = -1
                    gt_projection_min_slice = 1000
                    patch_gt_projection_landmarks_moving = None
                    gt_projection_moving_slices = None
            else:
                gt_projection_max_slice = -1
                gt_projection_min_slice = 1000
                patch_gt_projection_landmarks_moving = None
                gt_projection_moving_slices = None

            pred_max_slice = max(pred_max_slice, smoothed_max_slice, gt_projection_max_slice)
            pred_min_slice = min(pred_min_slice, smoothed_min_slice, gt_projection_min_slice)
        else:
            pred_max_slice = -1
            pred_min_slice = 10000
            slice_pred_landmarks_fixed = None
            patch_pred_landmarks_moving = None
            patch_gt_projection_landmarks_moving = None

        # Manual landmarks
        if manual_landmarks_fixed is not None and manual_landmarks_moving is not None:
            manual_landmarks_fixed_rows = np.where(manual_landmarks_fixed[:, 2] == slice_idx)[0]
            n_manual_landmarks = manual_landmarks_fixed_rows.shape[0]

            if n_manual_landmarks != 0:
                # Fixed image landmarks on current slice
                slice_manual_landmarks_fixed = manual_landmarks_fixed[manual_landmarks_fixed_rows, :]
                # Use the same rows because matching landmarks are in corr. rows by definition!!
                patch_manual_landmarks_moving = manual_landmarks_moving[manual_landmarks_fixed_rows, :]
                manual_moving_slices = patch_manual_landmarks_moving[:, 2]
                manual_max_slice = np.amax(manual_moving_slices)
                manual_min_slice = np.amin(manual_moving_slices)
            else:
                manual_max_slice = 10000
                manual_min_slice = -1
                slice_manual_landmarks_fixed = None
                patch_manual_landmarks_moving = None

            if n_manual_landmarks == 0 and n_pred_landmarks == 0:
                continue
        else:
            manual_max_slice = -1
            manual_min_slice = 10000
            n_manual_landmarks = 0

        if n_manual_landmarks == 0 and n_pred_landmarks == 0:
            continue

        min_slice = round_float_coords(min(pred_min_slice, manual_min_slice))
        max_slice = round_float_coords(max(pred_max_slice, manual_max_slice))

        # Create the neighbourhood
        if slice_idx < min_slice:
            fixed_image_offset = 0
            min_slice = slice_idx
            max_slice = max_slice
        elif slice_idx >= min_slice:
            fixed_image_offset = slice_idx - min_slice
            min_slice = min_slice
            if slice_idx < max_slice:
                max_slice = max_slice
            else:
                max_slice = slice_idx

        if max_slice >= moving_image.shape[-1]:
            print('The affine transformation has transformed this landmark outside image domain. Skip')
            continue

        if min_slice < 0:
            continue

        if verbose is True:
            print('Fixed slice idx: {}, Max slice : {}, Min slice: {},'\
                  ' pred landmarks {}, gt landmarks {}'.format(slice_idx,
                                                             max_slice,
                                                             min_slice,
                                                             n_pred_landmarks,
                                                             n_manual_landmarks))

        moving_image_patch = moving_image[:, :, min_slice:(max_slice+1)]
        i, j, k = moving_image_patch.shape

        # np.ndarray to "hold" both images (and matches)
        im = np.ones((i*2, j*k),
                      dtype=np.float32)


        im[:i, fixed_image_offset*j:(fixed_image_offset+1)*j] = fixed_image[:, :, slice_idx]

        for offset, moving_slice_idx in enumerate(range(min_slice, min(max_slice+1, n_slices))):
            im[i:, offset*j:(offset+1)*j] = moving_image[:, :, moving_slice_idx]

        # Convert the image to OpenCV RGB format
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        # Draw predicted correspondences
        if slice_pred_landmarks_fixed is not None:
            for row_idx in range(slice_pred_landmarks_fixed.shape[0]):
                fx, fy, fz = slice_pred_landmarks_fixed[row_idx]
                mx, my, mz = patch_pred_landmarks_moving[row_idx]

                fx = round_float_coords(fx)
                fy = round_float_coords(fy)
                fz = round_float_coords(fz)

                mx = round_float_coords(mx)
                my = round_float_coords(my)
                mz = round_float_coords(mz)

                cv2.circle(im, (fixed_image_offset*j + fx, fy), 2, pred_color, -1)
                cv2.circle(im, ((mz-min_slice)*j + mx, my+i), 2, pred_color, -1)
                cv2.line(im, (fixed_image_offset*j + fx, fy), ((mz-min_slice)*j + mx, i + my), (0, 0, 1), 1) # Red

        # Draw GT correspondences
        if manual_landmarks_fixed is not None and manual_landmarks_moving is not None:
            if slice_manual_landmarks_fixed is not None:
                for row_idx in range(slice_manual_landmarks_fixed.shape[0]):
                    fx, fy, fz = slice_manual_landmarks_fixed[row_idx]
                    mx, my, mz = patch_manual_landmarks_moving[row_idx]

                    fx = round_float_coords(fx)
                    fy = round_float_coords(fy)
                    fz = round_float_coords(fz)

                    mx = round_float_coords(mx)
                    my = round_float_coords(my)
                    mz = round_float_coords(mz)

                    cv2.circle(im, (fixed_image_offset*j + fx, fy), 2, manual_color, -1)
                    cv2.circle(im, ((mz-min_slice)*j + mx, my+i), 2, manual_color, -1)
                    cv2.line(im, (fixed_image_offset*j + fx, fy), ((mz-min_slice)*j + mx, my+i), (0, 1, 0), 1) # Green

        # Draw predicted correspondences after smoothing
        if slice_pred_landmarks_fixed is not None and smoothed_landmarks_moving is not None:
            for row_idx in range(slice_pred_landmarks_fixed.shape[0]):
                fx, fy, fz = slice_pred_landmarks_fixed[row_idx]
                mx, my, mz = patch_smoothed_landmarks_moving[row_idx]

                fx = round_float_coords(fx)
                fy = round_float_coords(fy)
                fz = round_float_coords(fz)

                mx = round_float_coords(mx)
                my = round_float_coords(my)
                mz = round_float_coords(mz)

                cv2.circle(im, (fixed_image_offset*j + fx, fy), 2, manual_color, -1)
                cv2.circle(im, ((mz-min_slice)*j + mx, my+i), 2, manual_color, -1)
                cv2.line(im, (fixed_image_offset*j + fx, fy), ((mz-min_slice)*j + mx, my+i), (1, 0, 0), 1) # Blue

        # Draw correspondences between predicted landmarks in the fixed image
        # and GT projection
        if slice_pred_landmarks_fixed is not None and gt_projection_landmarks_moving is not None:
                for row_idx in range(slice_pred_landmarks_fixed.shape[0]):
                    fx, fy, fz = slice_pred_landmarks_fixed[row_idx]
                    mx, my, mz = patch_gt_projection_landmarks_moving[row_idx]

                    fx = round_float_coords(fx)
                    fy = round_float_coords(fy)
                    fz = round_float_coords(fz)

                    mx = round_float_coords(mx)
                    my = round_float_coords(my)
                    mz = round_float_coords(mz)

                    cv2.circle(im, (fixed_image_offset*j + fx, fy), 2, manual_color, -1)
                    cv2.circle(im, ((mz-min_slice)*j + mx, my+i), 2, manual_color, -1)
                    cv2.line(im, (fixed_image_offset*j + fx, fy), ((mz-min_slice)*j + mx, my+i), (0, 1, 1), 1) # Yellow

        cv2.imwrite(os.path.join(out_dir, 'slice_{}.jpg'.format(slice_idx)),
                    (im*255).astype(np.uint8))





