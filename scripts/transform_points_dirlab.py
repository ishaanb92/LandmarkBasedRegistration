"""

Script that uses the estimated deformation (by the registration) to map
landmarks from the fixed image to the moving image domain to evaluate the registration

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
from elastix.transformix_interface import *
from lesionmatching.util_scripts.utils import add_library_path

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
        pid = pdir.split(os.sep)[-1]
        pat_point_dir = os.path.join(args.points_dir, pid)

        # To avoid issues that may arise from making images isotropic,
        # we use world coordinates to evaluate registration accuracy
        fixed_image_landmarks_path = os.path.join(pat_point_dir,
                                                  '{}_4D-75_T00_world_elx.txt'.format(pid))

        moving_image_landmarks_path = os.path.join(pat_point_dir,
                                                   '{}_4D-75_T00_world_elx.txt'.format(pid))

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

