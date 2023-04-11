"""

Based on threshold chosen using the estimated (deformable) displacements, filter landmark pairs with distances larger
than this threshold.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *
from lesionmatching.analysis.metrics import compute_euclidean_distance_between_points
from argparse import ArgumentParser
import shutil
import SimpleITK as sitk
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--softmask', action='store_true')

    args = parser.parse_args()

    if args.softmask is True:
        threshold = 27
    else:
        threshold = 12

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for pdir in pdirs:
        pid = pdir.split(os.sep)[-1]

        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        moving_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                       'moving_image.mha'))

        fixed_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                     'fixed_landmarks_elx.txt'))

        moving_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                      'moving_landmarks_elx.txt'))

        # Convert to voxel idxs
        fixed_image_landmarks_voxels = map_world_coord_to_voxel_index(fixed_image_landmarks,
                                                                      spacing=fixed_image_itk.GetSpacing(),
                                                                      origin=fixed_image_itk.GetOrigin())

        moving_image_landmarks_voxels = map_world_coord_to_voxel_index(moving_image_landmarks,
                                                                      spacing=moving_image_itk.GetSpacing(),
                                                                      origin=moving_image_itk.GetOrigin())

        euc_distances = compute_euclidean_distance_between_points(fixed_image_landmarks_voxels,
                                                                  moving_image_landmarks_voxels)

        fixed_image_landmarks_voxels_filtered = fixed_image_landmarks_voxels[euc_distances<=threshold, :]
        moving_image_landmarks_voxels_filtered = moving_image_landmarks_voxels[euc_distances<=threshold, :]

        print('Patient {} :: Landmarks pairs before filtering = {}, after filtereing = {}'.format(pid,
                                                                                                  fixed_image_landmarks_voxels.shape[0],
                                                                                                  fixed_image_landmarks_voxels_filtered.shape[0]))

        # Convert back to world

        fixed_image_landmarks_filtered = map_voxel_index_to_world_coord(fixed_image_landmarks_voxels_filtered,
                                                                        spacing=fixed_image_itk.GetSpacing(),
                                                                        origin=fixed_image_itk.GetOrigin())

        moving_image_landmarks_filtered = map_voxel_index_to_world_coord(moving_image_landmarks_voxels_filtered,
                                                                         spacing=moving_image_itk.GetSpacing(),
                                                                         origin=moving_image_itk.GetOrigin())

        create_landmarks_file(fixed_image_landmarks_filtered,
                              world=True,
                              fname=os.path.join(pdir, 'fixed_landmarks_elx_threshold.txt'))

        create_landmarks_file(moving_image_landmarks_filtered,
                              world=True,
                              fname=os.path.join(pdir, 'moving_landmarks_elx_threshold.txt'))

