"""

To check the "goodness" of our point correspondences, we fit a thin-plate spline (TPS) through all the point correspondences s.t. the TPS defines the transformation T : F->M. This transfomation can be then be analyzed to reason about performance or even filter a subset of point correspondences

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import SimpleITK as sitk
import numpy as np
from argparse import ArgumentParser
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *
from lesionmatching.data.deformations import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--mode', type=str, default='nn', help='nn or gt')

    args = parser.parse_args()

    if args.dataset == 'copd':
        points_dir = '/home/ishaan/COPDGene/points'
    elif args.dataset == 'dirlab':
        points_dir = '/home/ishaan/DIR-Lab/points'

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for pdir in pdirs:

        pid = pdir.split(os.sep)[-1]

        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        moving_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'moving_image.mha'))

        moving_image_np = convert_itk_to_ras_numpy(moving_image_itk)

        fixed_image_shape = fixed_image_itk.GetSize()
        moving_image_shape = moving_image_itk.GetSize()

        if args.mode == 'nn':
            fixed_points_arr = parse_points_file(os.path.join(pdir,
                                                              'fixed_landmarks_elx.txt'))

            moving_points_arr = parse_points_file(os.path.join(pdir,
                                                               'moving_landmarks_elx.txt'))
        elif args.mode == 'gt':
            fixed_points_arr = parse_points_file(os.path.join(points_dir,
                                                              pid,
                                                              '{}_300_iBH_world_r1_elx.txt'.format(pid)))

            moving_points_arr = parse_points_file(os.path.join(points_dir,
                                                               pid,
                                                               '{}_300_eBH_world_r1_elx.txt'.format(pid)))
        else:
            raise RuntimeError('{} is not a valid option for mode'.format(args.mode))

        # Scale the coordinates between [0, 1]

        fixed_points_scaled = np.divide(fixed_points_arr,
                                        np.expand_dims(np.array(fixed_image_shape),
                                                       axis=0))

        moving_points_scaled = np.divide(moving_points_arr,
                                         np.expand_dims(np.array(moving_image_shape),
                                                        axis=0))

        print('Fitting TPS for landmark correspondences for Patient {}'.format(pid))

        # 1. Fit thin-plate spline to define DVF based on point correspondences
        # Smoothing term is set to 0 for now, we want exact interpolation to study the
        # properties of the deformations defined by the (predicted/GT) landmark correspondences
        T = construct_tps_defromation(p1=fixed_points_scaled,
                                      p2=moving_points_scaled,
                                      shape=np.array(fixed_image_shape))

        # Save the transformed grid (to avoid recomputation)
        if args.mode == 'nn':
            np.save(file=os.path.join(pdir, 'tps_transformed_grid.npy'),
                    arr=T)
        elif args.mode == 'gt':
            np.save(file=os.path.join(pdir, 'tps_transformed_grid_gt.npy'),
                    arr=T)
        else:
            raise RuntimeError('{} is not a valid option for mode'.format(args.mode))

        # 2. Compute determinant of Jacobian
        jac_det = calculate_jacobian_determinant(deformed_grid=T)

        print('Jacobian determinant :: Min = {}, Max = {}'.format(np.amin(jac_det),
                                                                  np.amax(jac_det)))
        # 3. Save the jac_det as an ITK image
        if args.mode == 'nn':
            save_ras_as_itk(img=jac_det,
                            metadata={'spacing':fixed_image_itk.GetSpacing(),
                                      'origin':fixed_image_itk.GetOrigin(),
                                      'direction':fixed_image_itk.GetDirection()},
                            fname=os.path.join(pdir, 'jac_det.mha'))
        elif args.mode == 'gt':
            save_ras_as_itk(img=jac_det,
                            metadata={'spacing':fixed_image_itk.GetSpacing(),
                                      'origin':fixed_image_itk.GetOrigin(),
                                      'direction':fixed_image_itk.GetDirection()},
                            fname=os.path.join(pdir, 'jac_det_gt.mha'))
        else:
            raise RuntimeError('{} is not a valid option for mode'.format(args.mode))

        # 4. Resample moving image based on TPS deformation estimated from point pairs
        moving_image_resampled_np = resample_image(image=moving_image_np,
                                                   transformed_coordinates=T)

        if args.mode == 'nn':
            save_ras_as_itk(img=moving_image_resampled_np,
                            metadata={'spacing':moving_image_itk.GetSpacing(),
                                      'origin':moving_image_itk.GetOrigin(),
                                      'direction':moving_image_itk.GetDirection()},
                            fname=os.path.join(pdir, 'tps_resampled_moving_image.mha'))
        elif args.mode == 'gt':
            save_ras_as_itk(img=moving_image_resampled_np,
                            metadata={'spacing':moving_image_itk.GetSpacing(),
                                      'origin':moving_image_itk.GetOrigin(),
                                      'direction':moving_image_itk.GetDirection()},
                            fname=os.path.join(pdir, 'tps_resampled_moving_image_gt.mha'))
        else:
            raise RuntimeError('{} is not a valid option for mode'.format(args.mode))







