"""

Script to transform fixed GT landmarks from fixed image domain -> moving image domain
as a pre-cursor to registration

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""


import os
import numpy as np
from argparse import ArgumentParser
from elastix.transformix_interface import *
from lesionmatching.util_scripts.utils import *
from lesionmatching.analysis.metrics import *

TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--affine_reg_dir', type=str, required=True)
    parser.add_argument('--bspline_reg_dir', type=str, required=True)
    parser.add_argument('--points_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, help='dirlab or copd')

    args = parser.parse_args()


    # Patients in the registration directory
    pat_dirs = [f.path for f in os.scandir(args.bspline_reg_dir) if f.is_dir()]

    add_library_path(ELASTIX_LIB)


    for bspline_dir in pat_dirs:

        # Step 1. Transform anatomical landmarks from fixed image domain
        pid = bspline_dir.split(os.sep)[-1]
        pat_point_dir = os.path.join(args.points_dir, pid)
        affine_reg_dir = os.path.join(args.affine_reg_dir, pid)

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


        # Step-1-a : Affine transform fixed landmarks
        transform_param_file = os.path.join(affine_reg_dir, 'TransformParameters.0.txt')

        tr_if = TransformixInterface(parameters=transform_param_file,
                                     transformix_path=TRANSFORMIX_BIN)

        affine_transformed_fixed_points_path = tr_if.transform_points(pointsfile_path=fixed_image_landmarks_path,
                                                                      output_dir=affine_reg_dir)


        # Step 2-a : Convert transformix output to array
        affine_transformed_arr = parse_transformix_points_output(fpath=affine_transformed_fixed_points_path)

        # Step 2-b : Convert array to elastix/transformix input format
        create_landmarks_file(landmarks=affine_transformed_arr,
                              world=True,
                              fname=os.path.join(bspline_dir, 'affine_transformed_fixed_points.txt'))

        # Step-3 : Non-rigid transform affine fixed landmarks
        transform_param_file = os.path.join(bspline_dir, 'TransformParameters.1.txt')

        tr_if = TransformixInterface(parameters=transform_param_file,
                                     transformix_path=TRANSFORMIX_BIN)

        transformed_fixed_points_path = tr_if.transform_points(pointsfile_path=os.path.join(bspline_dir, 'affine_transformed_fixed_points.txt'),
                                                               output_dir=bspline_dir)

        # Step 4-a : Convert transformix output to array
        transformed_fixed_point_arr = parse_transformix_points_output(fpath=transformed_fixed_points_path)

        # Step 4-b : Convert array to elastix/transformix input format
        create_landmarks_file(landmarks=transformed_fixed_point_arr,
                              world=True,
                              fname=os.path.join(bspline_dir, 'transformed_fixed_points.txt'))

        # Copy the fixed, moving, transformed fixed landmarks to the bspline registration directory
        shutil.copy(fixed_image_landmarks_path,
                    os.path.join(bspline_dir, 'fixed_image_landmarks.txt'))

        shutil.copy(moving_image_landmarks_path,
                    os.path.join(bspline_dir, 'moving_image_landmarks.txt'))
