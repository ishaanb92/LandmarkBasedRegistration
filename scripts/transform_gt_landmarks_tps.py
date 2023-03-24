"""

Script to transform fixed GT landmarks from fixed image domain -> moving image domain
as a pre-cursor to quantitatively evaluating registration using the TPS

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
from argparse import ArgumentParser
from lesionmatching.util_scripts.utils import *
from lesionmatching.analysis.metrics import *
from elastix.transformix_interface import *
import joblib
import shutil
import SimpleITK as sitk

TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--affine_reg_dir', type=str, required=True)
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--points_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, help='dirlab or copd')
    parser.add_argument('--smoothing_term', type=float, default=0.1)
    args = parser.parse_args()


    # Patients in the registration directory
    pat_dirs = [f.path for f in os.scandir(args.affine_reg_dir) if f.is_dir()]

    add_library_path(ELASTIX_LIB)

    for affine_reg_dir in pat_dirs:

        # Step 1. Transform anatomical landmarks from fixed image domain
        pid = affine_reg_dir.split(os.sep)[-1]
        pat_point_dir = os.path.join(args.points_dir, pid)


        # To avoid issues that may arise from making images isotropic,
        # we use world coordinates to evaluate registration accuracy
        if args.dataset == 'dirlab':
            fixed_image_landmarks_path = os.path.join(pat_point_dir,
                                                      '{}_4D-75_T00_world_elx.txt'.format(pid))

            moving_image_landmarks_path = os.path.join(pat_point_dir,
                                                       '{}_4D-75_T50_world_elx.txt'.format(pid))
        elif args.dataset == 'copd':
            fixed_image_landmarks_path = os.path.join(pat_point_dir,
                                                      '{}_300_iBH_world_r1_elx.txt'.format(pid))

            moving_image_landmarks_path = os.path.join(pat_point_dir,
                                                       '{}_300_eBH_world_r1_elx.txt'.format(pid))


        shutil.copy(fixed_image_landmarks_path,
                    os.path.join(args.landmarks_dir,
                                 pid,
                                 'fixed_image_landmarks.txt'))

        shutil.copy(moving_image_landmarks_path,
                    os.path.join(args.landmarks_dir,
                                 pid,
                                 'moving_image_landmarks.txt'))

        # Step 2 : Affine transform fixed landmarks
        transform_param_file = os.path.join(affine_reg_dir, 'TransformParameters.0.txt')

        tr_if = TransformixInterface(parameters=transform_param_file,
                                     transformix_path=TRANSFORMIX_BIN)

        affine_transformed_fixed_points_path = tr_if.transform_points(pointsfile_path=fixed_image_landmarks_path,
                                                                      output_dir=affine_reg_dir)

        affine_transformed_fixed_point_arr = parse_transformix_points_output(fpath=affine_transformed_fixed_points_path)

        # Step 3: Load TPS object
        tps_func = joblib.load(os.path.join(args.landmarks_dir,
                                            pid,
                                            'tps_{}.pkl'.format(args.smoothing_term)))

        # Step 4: Transform (affine-transformed) GT fixed landmarks
        # 4-a : Convert world coords to voxel
        fixed_image_itk = sitk.ReadImage(os.path.join(args.landmarks_dir,
                                                      pid,
                                                      'fixed_image.mha'))

        # 4-b : Convert world coordindates to voxel idxs
        affine_transformed_fixed_point_arr_voxel = map_world_coord_to_voxel_index(affine_transformed_fixed_point_arr,
                                                                                  spacing=fixed_image_itk.GetSpacing(),
                                                                                  origin=fixed_image_itk.GetOrigin())

        # 4-c: Scale voxel idxs to [0, 1] range
        affine_transformed_fixed_point_arr_voxel_scaled = np.divide(affine_transformed_fixed_point_arr_voxel,
                                                                    np.expand_dims(np.array(fixed_image_itk.GetSize()),
                                                                                   axis=0))
        # 4-d : Use TPS function to transform points
        transformed_fixed_point_arr_voxel_scaled = tps_func(affine_transformed_fixed_point_arr_voxel_scaled)

        # 4-e : Re-scale to image dimensions
        transformed_fixed_point_arr_voxel = np.multiply(transformed_fixed_point_arr_voxel_scaled,
                                                        np.expand_dims(np.array(fixed_image_itk.GetSize()),
                                                                       axis=0))
        # 4-f : Convert to world coordinates
        transformed_fixed_point_arr = map_voxel_index_to_world_coord(transformed_fixed_point_arr_voxel,
                                                                     spacing=fixed_image_itk.GetSpacing(),
                                                                     origin=fixed_image_itk.GetOrigin())
        # Step 5: Save the landmarks file
        create_landmarks_file(transformed_fixed_point_arr,
                              world=True,
                              fname=os.path.join(args.landmarks_dir,
                                                 pid,
                                                 'transformed_fixed_points_{}.txt'.format(args.smoothing_term)))




