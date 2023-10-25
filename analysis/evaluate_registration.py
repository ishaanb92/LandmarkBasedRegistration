""""
Script to evaluate "goodness" of registration quantitatively using the TRE metric.
This metric is computed using (manually annotated) anatomical landmarks and the
transformation estimated by the registration algorithm

Steps:
    1. Parse the (transformed) fixed/moving points files to extract corresponding world coordinates

    2. Compute landmark error (pre- and post-registration)

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
    parser.add_argument('--smoothing_term', type=float, default=0.0)

    args = parser.parse_args()


    # Patients in the registration directory
    pat_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    add_library_path(ELASTIX_LIB)

    median_reg_error = np.zeros((len(pat_dirs)),
                                dtype=np.float32)

    for idx, pdir in enumerate(pat_dirs):

        pid = pdir.split(os.sep)[-1]

        fixed_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                'fixed_image_landmarks.txt'))

        moving_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                 'moving_image_landmarks.txt'))


        try:
            affine_tr_fixed_points = parse_points_file(fpath=os.path.join(pdir,
                                                                          'affine_transformed_fixed_points.txt'))

            if args.smoothing_term == 0:
                transformed_fixed_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                                    'transformed_fixed_points.txt'))
            else:
                transformed_fixed_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                                    'transformed_fixed_points_{}.txt'.format(args.smoothing_term)))

            spatial_errors_pre_reg = compute_euclidean_distance_between_points(x1=fixed_points_arr,
                                                                               x2=moving_points_arr)

            spatial_errors_post_affine = compute_euclidean_distance_between_points(x1=affine_tr_fixed_points,
                                                                                   x2=moving_points_arr)

            spatial_errors_post_reg = compute_euclidean_distance_between_points(x1=transformed_fixed_points_arr,
                                                                                x2=moving_points_arr)

            print('Patient {} :: TRE :: Pre-reg : {:.3f} +/- {:.3f}  Affine: {:.3f} +/- {:.3f} Post-reg :: {:.3f} +/- {:.3f}'.format(pid,
                                                                                                             np.mean(spatial_errors_pre_reg),
                                                                                                             np.std(spatial_errors_pre_reg),
                                                                                                             np.mean(spatial_errors_post_affine),
                                                                                                             np.std(spatial_errors_post_affine),
                                                                                                             np.mean(spatial_errors_post_reg),
                                                                                                             np.std(spatial_errors_post_reg)))

            # Save the results
            np.save(file=os.path.join(pdir, 'pre_reg_error.npy'),
                    arr=spatial_errors_pre_reg)

            np.save(file=os.path.join(pdir, 'post_affine_error.npy'),
                    arr=spatial_errors_post_affine)

            np.save(file=os.path.join(pdir, 'post_reg_error.npy'),
                    arr=spatial_errors_post_reg)

            median_reg_error[idx] = np.median(spatial_errors_post_reg)
        except FileNotFoundError: # Only affine registration

            if args.smoothing_term == 0:
                transformed_fixed_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                                    'transformed_fixed_points.txt'))
            else:
                transformed_fixed_points_arr = parse_points_file(fpath=os.path.join(pdir,
                                                                                    'transformed_fixed_points_{}.txt'.format(args.smoothing_term)))

            spatial_errors_pre_reg = compute_euclidean_distance_between_points(x1=fixed_points_arr,
                                                                               x2=moving_points_arr)


            spatial_errors_post_reg = compute_euclidean_distance_between_points(x1=transformed_fixed_points_arr,
                                                                                x2=moving_points_arr)

            print('Patient {} :: TRE :: Pre-reg : {:.3f} +/- {:.3f}  Post-reg :: {:.3f} +/- {:.3f}'.format(pid,
                                                                                                           np.mean(spatial_errors_pre_reg),
                                                                                                           np.std(spatial_errors_pre_reg),
                                                                                                           np.mean(spatial_errors_post_reg),
                                                                                                           np.std(spatial_errors_post_reg)))


            np.save(file=os.path.join(pdir, 'pre_reg_error.npy'),
                    arr=spatial_errors_pre_reg)

            np.save(file=os.path.join(pdir, 'post_reg_error.npy'),
                    arr=spatial_errors_post_reg)

            median_reg_error[idx] = np.median(spatial_errors_post_reg)


    print('Median registration error : {}'.format(np.median(median_reg_error)))

