""""
Script to evaluate "goodness" of registration quantitatively using the TRE metric.
This metric is computed using (manually annotated) anatomical landmarks and the
transformation estimated by the registration algorithm

Steps:
    1. Using the transformation estimated by the registration algorithm, transform landmarks from the fixed image domain
       to the moving image domain

    2. Parse the different points files to extract voxel/world coordinates

    3. Compute landmark error (pre- and post-registration)

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
    parser.add_argument('--reg_dir', type=str, required=True)
    parser.add_argument('--points_dir', type=str, required=True)


    args = parser.parse_args()


    # Patients in the registration directory
    pat_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    add_library_path(ELASTIX_LIB)


    for pdir in pat_dirs:

        # Step 1. Transform anatomical landmarks from fixed image domain
        pid = pdir.split(os.sep)[-1]
        pat_point_dir = os.path.join(args.points_dir, pid)

        # To avoid issues that may arise from making images isotropic,
        # we use world coordinates to evaluate registration accuracy
        fixed_image_landmarks_path = os.path.join(pat_point_dir,
                                                  '{}_4D-75_T00_world_elx.txt'.format(pid))

        moving_image_landmarks_path = os.path.join(pat_point_dir,
                                                   '{}_4D-75_T50_world_elx.txt'.format(pid))

        transform_param_file = os.path.join(pdir, 'TransformParameters.2.txt')

        tr_if = TransformixInterface(parameters=transform_param_file,
                                     transformix_path=TRANSFORMIX_BIN)

        transformed_fixed_points_path = tr_if.transform_points(pointsfile_path=fixed_image_landmarks_path,
                                                               output_dir=pat_point_dir)

        # Copy the fixed, moving, transformed fixed landmarks to the registration directory
        shutil.copy(fixed_image_landmarks_path,
                    os.path.join(pdir, 'fixed_image_landmarks.txt'))

        shutil.copy(moving_image_landmarks_path,
                    os.path.join(pdir, 'moving_image_landmarks.txt'))

        shutil.copy(transformed_fixed_points_path,
                    os.path.join(pdir, 'transformed_fixed_image_landmarks.txt'))


        # Step 2. Parse the points output
        # Step 2-a Parse the transformix points output
        # Shape : (N, 3)
        transformed_fixed_points_arr = parse_transformix_points_output(fpath=transformed_fixed_points_path)

        # Step 2-b Parse the moving landmarks file
        # Shape: (N, 3)
        moving_points_arr = parse_points_file(fpath=moving_image_landmarks_path)
        fixed_points_arr = parse_points_file(fpath=fixed_image_landmarks_path)

        if moving_points_arr is None or fixed_points_arr is None:
            continue

        # Step 3 Compute TRE
        # Step 3-a Compute TRE before registration
        spatial_errors_pre_reg = compute_euclidean_distance_between_points(x1=fixed_points_arr,
                                                                           x2=moving_points_arr)
        # Step 3-b Compute TRE after registration
        spatial_errors_post_reg = compute_euclidean_distance_between_points(x1=transformed_fixed_points_arr,
                                                                            x2=moving_points_arr)

        print('Patient {} :: TRE :: Pre-reg : {} +/- {}  Post-reg :: {} +/- {}'.format(pid,
                                                                                       np.mean(spatial_errors_pre_reg),
                                                                                       np.std(spatial_errors_pre_reg),
                                                                                       np.mean(spatial_errors_post_reg),
                                                                                       np.std(spatial_errors_post_reg)))


        # Save the results
        np.save(file=os.path.join(pdir, 'pre_reg_error.npy'),
                arr=spatial_errors_pre_reg)
        np.save(file=os.path.join(pdir, 'post_reg_error.npy'),
                arr=spatial_errors_post_reg)


