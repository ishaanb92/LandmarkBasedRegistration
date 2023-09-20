"""

To check the "goodness" of our point correspondences, we fit a thin-plate spline (TPS) through all the point correspondences s.t. the TPS defines the transformation T : F->M.
This transfomation can be then be used to warp the moving image to check whether the DL-based landmark correspondences produce a plausible deformation.
Additionally using a smoothing term > 0, the correspondences can be be edited

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
    parser.add_argument('--landmarks_dir', type=str, default=None)
    parser.add_argument('--affine_reg_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--mode', type=str, default='nn', help='nn or gt')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--transform_grid', action='store_true')

    args = parser.parse_args()

    if args.dataset == 'copd':
        points_dir = '/home/ishaan/COPDGene/points'
        data_dir = '/home/ishaan/COPDGene/mha'
    elif args.dataset == 'dirlab':
        points_dir = '/home/ishaan/DIR-Lab/points'
        data_dir = '/home/ishaan/DIR-Lab/mha'


    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    if args.mode == 'nn':
        LAMBDAS = [0.0, 0.01, 0.05]
        assert(args.landmarks_dir is not None)
    elif args.mode == 'gt':
        LAMBDAS = [0]


    for pdir in pdirs:

        pid = pdir.split(os.sep)[-1]

        patient_data_dir = os.path.join(data_dir,
                                        pid)

        if args.dataset == 'dirlab':
            fixed_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_T00_iso.mha'.format(pid)))

            moving_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_T50_iso.mha'.format(pid)))
        elif args.dataset == 'copd':
            fixed_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_iBHCT_iso.mha'.format(pid)))

            moving_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_eBHCT_iso.mha'.format(pid)))

        moving_image_np = convert_itk_to_ras_numpy(moving_image_itk)

        fixed_image_shape = fixed_image_itk.GetSize()
        moving_image_shape = moving_image_itk.GetSize()

        # Landmarks in fixed and moving images are specified in world coordinates
        if args.mode == 'nn':
            fixed_points_arr = parse_points_file(os.path.join(pdir,
                                                              'fixed_landmarks_elx.txt'))

            moving_points_arr = parse_points_file(os.path.join(pdir,
                                                               'moving_landmarks_elx.txt'))
        elif args.mode == 'gt':

            #assert(args.affine_reg_dir is not None)

            if args.dataset == 'copd':
                fixed_points_arr = parse_points_file(os.path.join(points_dir,
                                                                  pid,
                                                                  '{}_300_iBH_world_r1_elx.txt'.format(pid)))
            elif args.dataset == 'dirlab':
                fixed_points_arr = parse_points_file(os.path.join(points_dir,
                                                                  pid,
                                                                  '{}_4D-75_T00_world_elx.txt'.format(pid)))


            # Since the moving image is already affine-registered, use the inverse affine transformed moving landmarks
            if args.affine_reg_dir is not None:
                moving_points_arr = parse_points_file(os.path.join(args.affine_reg_dir,
                                                                   pid,
                                                                   'transformed_moving_landmarks_elx.txt'))
            else:
                if args.dataset == 'copd':
                    moving_points_arr = parse_points_file(os.path.join(points_dir,
                                                                      pid,
                                                                      '{}_300_eBH_world_r1_elx.txt'.format(pid)))
                elif args.dataset == 'dirlab':
                    moving_points_arr = parse_points_file(os.path.join(points_dir,
                                                                       pid,
                                                                      '{}_4D-75_T50_world_elx.txt'.format(pid)))

        else:
            raise RuntimeError('{} is not a valid option for mode'.format(args.mode))


        # Convert world coordinates to voxel coordinates
        fixed_points_arr = map_world_coord_to_voxel_index(world_coords=fixed_points_arr,
                                                          spacing=fixed_image_itk.GetSpacing(),
                                                          origin=fixed_image_itk.GetOrigin())

        moving_points_arr = map_world_coord_to_voxel_index(world_coords=moving_points_arr,
                                                           spacing=moving_image_itk.GetSpacing(),
                                                           origin=moving_image_itk.GetOrigin())

        # Scale the coordinates between [0, 1]
        fixed_points_scaled = np.divide(fixed_points_arr,
                                        np.expand_dims(np.array(fixed_image_shape),
                                                       axis=0))

        moving_points_scaled = np.divide(moving_points_arr,
                                         np.expand_dims(np.array(moving_image_shape),
                                                        axis=0))

        for lmbda in LAMBDAS:
            print('Fitting TPS for landmark correspondences for Patient {}, lambda = {}'.format(pid, lmbda))

            # 1. Fit thin-plate spline to define DVF based on point correspondences
            # Smoothing term is set to 0 for now, we want exact interpolation to study the
            # properties of the deformations defined by the (predicted/GT) landmark correspondences
            tps_interpolator = RBFInterpolator(y=fixed_points_scaled,
                                               d=moving_points_scaled,
                                               smoothing=lmbda,
                                               kernel='thin_plate_spline',
                                               degree=1)


            if args.transform_grid:
                transformed_grid = transform_grid(transform=tps_interpolator,
                                                  shape=np.array(fixed_image_itk.GetSize()))

                # Save the transformed grid (to avoid recomputation)
                if args.mode == 'nn':
                    np.save(file=os.path.join(pdir, 'tps_transformed_grid_{}.npy'.format(lmbda)),
                            arr=transformed_grid)
                elif args.mode == 'gt':
                    if args.affine_reg_dir is None:
                        np.save(file=os.path.join(points_dir, pid, 'tps_transformed_grid_gt_all.npy'),
                                arr=transformed_grid)
                    else:
                        np.save(file=os.path.join(points_dir, pid, 'tps_transformed_grid_gt_only_bspline.npy'),
                                arr=transformed_grid)
                else:
                    raise RuntimeError('{} is not a valid option for mode'.format(args.mode))
                # Resample moving image based on TPS deformation estimated from point pairs
                moving_image_resampled_np = resample_image(image=moving_image_np,
                                                           transformed_coordinates=transformed_grid)

                if args.mode == 'nn':
                    save_ras_as_itk(img=moving_image_resampled_np,
                                    metadata={'spacing':moving_image_itk.GetSpacing(),
                                              'origin':moving_image_itk.GetOrigin(),
                                              'direction':moving_image_itk.GetDirection()},
                                    fname=os.path.join(pdir, 'tps_resampled_moving_image_{}.mha'.format(lmbda)))
                elif args.mode == 'gt':
                    save_ras_as_itk(img=moving_image_resampled_np,
                                    metadata={'spacing':moving_image_itk.GetSpacing(),
                                              'origin':moving_image_itk.GetOrigin(),
                                              'direction':moving_image_itk.GetDirection()},
                                    fname=os.path.join(pdir, 'tps_resampled_moving_image_gt.mha'))
                else:
                    raise RuntimeError('{} is not a valid option for mode'.format(args.mode))

                # 2. Compute determinant of Jacobian
                jac_det = calculate_jacobian_determinant(deformed_grid=transformed_grid)

                print('Jacobian determinant :: Min = {}, Max = {}'.format(np.amin(jac_det),
                                                                          np.amax(jac_det)))
                # 3. Save the jac_det as an ITK image
                if args.mode == 'nn':
                    save_ras_as_itk(img=jac_det,
                                    metadata={'spacing':fixed_image_itk.GetSpacing(),
                                              'origin':fixed_image_itk.GetOrigin(),
                                              'direction':fixed_image_itk.GetDirection()},
                                    fname=os.path.join(pdir, 'jac_det_{}.mha'.format(lmbda)))
                elif args.mode == 'gt':
                    save_ras_as_itk(img=jac_det,
                                    metadata={'spacing':fixed_image_itk.GetSpacing(),
                                              'origin':fixed_image_itk.GetOrigin(),
                                              'direction':fixed_image_itk.GetDirection()},
                                    fname=os.path.join(pdir, 'jac_det_gt.mha'))
                else:
                    raise RuntimeError('{} is not a valid option for mode'.format(args.mode))

            # 4. Save the TPS deformation as a .pkl file
            if args.mode == 'nn':
                joblib.dump(tps_interpolator,
                            os.path.join(pdir, 'tps_{}.pkl'.format(lmbda)))
            elif args.mode == 'gt':
                joblib.dump(tps_interpolator,
                            os.path.join(points_dir, pid, 'tps_gt.pkl'))

