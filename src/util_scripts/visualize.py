"""
Script to visualize keypoint matches

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import numpy as np
import pandas as pd
import cv2
from utils.utils import maybe_convert_tensor_to_numpy
import os
import math

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


def visualize_keypoints_3d(im1, im2, landmarks1, landmarks2, pred_matches, gt_matches, out_dir, verbose=False):
    """
    Function to visualize (predicted and true) landmark correspondences.
    Wraps the 2-D visualization function from https://github.com/monikagrewal/End2EndLandmarks
    such that correspondences found on the same slice can be visualized.

    Matches (both predicted and true) found on different slices are ignored (FIXME!)

    """
    im1 = maybe_convert_tensor_to_numpy(im1)
    im2 = maybe_convert_tensor_to_numpy(im2)

    # Shape: (K, 3)
    landmarks1 = maybe_convert_tensor_to_numpy(landmarks1)
    landmarks2 = maybe_convert_tensor_to_numpy(landmarks2)

    # Shape: (K, K)
    pred_matches = maybe_convert_tensor_to_numpy(pred_matches)

    # Shape: (K, K)
    gt_matches = maybe_convert_tensor_to_numpy(gt_matches)

    # Rescale images to 0, 1 range for display
    im1 = min_max_scaling(im1)
    im2 = min_max_scaling(im2)

    assert(im1.ndim == 3)
    assert(im2.ndim == 3)

    n_slices = im1.shape[-1]

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    for slice_idx in range(n_slices):

        # Avoid the slices at the edges!
        if slice_idx - 2 < 0 or slice_idx + 3 > n_slices:
            continue

        im1_slice = im1[:, :, slice_idx]
        im2_patch = im2[:, :, slice_idx-2:slice_idx+3]

        # Filter (3-D) keypoints based on the slice "neighbourhood"
        # Get landmarks on the current slice for 'image'
        slice_landmarks1_rows = np.where(landmarks1[:, 0] == slice_idx)[0]

        # Get all landmarks in the slic "neigbourhood" for the 'image_hat'
        patch_landmarks2_rows = np.where((landmarks2[:, 0] == slice_idx-2) |
                                         (landmarks2[:, 0] == slice_idx-1) |
                                         (landmarks2[:, 0] == slice_idx)   |
                                         (landmarks2[:, 0] == slice_idx+1) |
                                         (landmarks2[:, 0] == slice_idx +2))[0]

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

        slice_gt_matches = np.zeros((n_landmarks1, n_landmarks2),
                                     dtype=np.uint8)


        # Create per-slice "match" matrices for prediction and GT
        for i in range(n_landmarks1):
            for j in range(n_landmarks2):
                if pred_matches[slice_landmarks1_rows[i], patch_landmarks2_rows[j]] == 1:
                    slice_pred_matches[i, j] = 1

                if gt_matches[slice_landmarks1_rows[i], patch_landmarks2_rows[j]] == 1:
                    slice_gt_matches[i, j] = 1

        n_matches = np.where(slice_pred_matches==1)[0].shape[0]
        n_gt_matches = np.where(slice_gt_matches==1)[0].shape[0]

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



def visualize_keypoints_2d_neighbourhood(im1, im2, output1, output2, pred_mask, gt_mask, out_dir, slice_id, basename="slice"):
    """
    Function to display matches for a given slice in the original image. To show matches that may occur across slices, we define this hybrid function where the we take only one slice from the original image, but fetch a corresponding "neighbourhood" of slices from the deformed image and display the matches

    """


    assert(im1.ndim == 2)
    assert(im2.ndim == 3)

    i, j, k = im2.shape


    # np.ndarray to "hold" both images (and matches)
    im = np.zeros((i*k, 2*j),
                  dtype=np.float32)

    # Put the original image in the "center" block
    im[2*i:3*i, :j] = im1

    for slice_idx in range(k):
        im2_slice = im2[:, :, slice_idx]
        im[slice_idx*i:(slice_idx+1)*i, j:] = im2_slice

    # Convert the image to OpenCV RGB format
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    color = [0, 0, 1]

    for k1, l1 in enumerate(output1):
        kk1, jj1, ii1 = l1
        kk1 = round_float_coords(kk1)
        jj1 = round_float_coords(jj1)
        ii1 = round_float_coords(ii1)

        cv2.circle(im, (jj1, 2*i + ii1), 2, color, -1)

        for k2, l2 in enumerate(output2):
            kk2, jj2, ii2 = l2
            kk2 = round_float_coords(kk2)
            jj2 = round_float_coords(jj2)
            ii2 = round_float_coords(ii2)

            slice_diff = kk1 - kk2

            cv2.circle(im, (jj2+j, ((2+slice_diff)*i + ii2)), 2, color, -1)

            if pred_mask[k1, k2] == 1:
                if gt_mask[k1, k2] == 1: # True positive
                    cv2.line(im, (jj1, 2*i + ii1), (jj2+j, (2+slice_diff)*i + ii2), (0, 1, 0), 1) # Green
                else: # False positive
                    cv2.line(im, (jj1, 2*i + ii1), (jj2+j, (2+slice_diff)*i + ii2), (0, 1, 1), 1) # Yellow

            if gt_mask[k1, k2] == 1:
                if pred_mask[k1, k2] == 0: # False negative
                    cv2.line(im, (jj1, 2*i + ii1), ((jj2+j, (2+slice_diff)*i + ii2)), (0, 0, 1), 1) # Red

    cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(basename)),
                (im*255).astype(np.uint8))









