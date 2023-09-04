"""

Script to visualize GT landmarks pairs and (DL-)predicted landmark pairs on the same image.

Additionally, this script also computes the landmark localization error:
    For a corresponding landmark pair X_f, X_m, the localization error is defined as:
        LocError = d(X_m, T(X_f)), where T is the GT deformation estimated from manual landmarks (by fitting a thin-plate spline)
        and d is the Euclidean distance

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from argparse import ArgumentParser
import SimpleITK as sitk
from lesionmatching.analysis.visualize import *
from lesionmatching.analysis.metrics import *
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str)
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--points_dir', type=str, default='/home/ishaan/COPDGene/points')
    parser.add_argument('--affine_reg_dir', type=str, help='Affine registration directory contains GT moving image landmarks')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--use_threshold', action='store_true')
    parser.add_argument('--show_gt_matches', action='store_true')
    parser.add_argument('--show_gt_projection', action='store_true')
    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]

        print('Processing patient {}'.format(pid))
        if args.affine_reg_dir is not None:
            affine_pdir = os.path.join(args.affine_reg_dir, pid)
        else:
            affine_pdir = None

        points_pdir = os.path.join(args.points_dir, pid)

        # 1. Read fixed and (affine-transformed) moving images

        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        # Convert ITK image to a RAS ordered numpy ndarray
        fixed_image_np = np.transpose(sitk.GetArrayFromImage(fixed_image_itk),
                                      (2, 1, 0))

        moving_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                       'moving_image.mha'))

        # Convert ITK image to a RAS ordered numpy ndarray
        moving_image_np = np.transpose(sitk.GetArrayFromImage(moving_image_itk),
                                      (2, 1, 0))

        # 2-a. Read fixed and moving predicted landmarks .txt files

        if args.use_threshold is False:
            fixed_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                            'fixed_landmarks_elx.txt'))

            moving_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                             'moving_landmarks_elx.txt'))
        else:
            fixed_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                            'fixed_landmarks_elx_threshold.txt'))

            moving_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                             'moving_landmarks_elx_threshold.txt'))

        # Moving landmarks after TPS-based smoothing
        if args.smoothing > 0:
            if args.use_threshold is False:
                moving_image_landmarks_smoothed_world = parse_points_file(os.path.join(pdir,
                                                                                       'moving_landmarks_elx_{}.txt'.format(args.smoothing)))
            else:
                moving_image_landmarks_smoothed_world = parse_points_file(os.path.join(pdir,
                                                                                       'moving_landmarks_elx_threshold_{}.txt'.format(args.smoothing)))
        else:
            moving_image_landmarks_smoothed_world = None
            moving_image_landmarks_smoothed_voxels = None


        # 2-b. Convert world coordinates to voxels

        fixed_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=fixed_image_landmarks_world,
                                                                      spacing=fixed_image_itk.GetSpacing(),
                                                                      origin=fixed_image_itk.GetOrigin())

        moving_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=moving_image_landmarks_world,
                                                                       spacing=moving_image_itk.GetSpacing(),
                                                                       origin=moving_image_itk.GetOrigin())

        if moving_image_landmarks_smoothed_world is not None:
            moving_image_landmarks_smoothed_voxels = map_world_coord_to_voxel_index(world_coords=moving_image_landmarks_smoothed_world,
                                                                                    spacing=moving_image_itk.GetSpacing(),
                                                                                    origin=moving_image_itk.GetOrigin())

        # Map fixed image landmarks into moving image using TPS defined by GT corr.
        fixed_image_landmarks_voxels_scaled = np.divide(fixed_image_landmarks_voxels,
                                                        np.expand_dims(np.array(fixed_image_itk.GetSize()),
                                                                       axis=0))

        tps_func = joblib.load(os.path.join(points_pdir,
                                            'tps_gt.pkl'))

        gt_projection_landmarks_scaled = tps_func(fixed_image_landmarks_voxels_scaled)

        gt_projection_landmarks = np.multiply(gt_projection_landmarks_scaled,
                                              np.expand_dims(np.array(moving_image_itk.GetSize()),
                                                             axis=0))

        gt_projection_landmarks_world = map_voxel_index_to_world_coord(gt_projection_landmarks,
                                                                       spacing=moving_image_itk.GetSpacing(),
                                                                       origin=moving_image_itk.GetOrigin())


        # 3-a. Read and fixed and (affine-transformed) moving GT landmarks .txt files
        if args.dataset == 'copd':
            gt_fixed_image_landmarks_world = parse_points_file(os.path.join(points_pdir,
                                                                            '{}_300_iBH_world_r1_elx.txt'.format(pid)))
        elif args.dataset == 'dirlab':
            raise NotImplementedError

        if affine_pdir is not None:
            gt_moving_image_landmarks_world = parse_points_file(os.path.join(affine_pdir,
                                                                             'transformed_moving_landmarks_elx.txt'))
        else:
            gt_moving_image_landmarks_world = parse_points_file(os.path.join(points_pdir,
                                                                             '{}_300_eBH_world_r1_elx.txt'.format(pid)))


        # 3-b. Convert world coordinates to voxels
        gt_fixed_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=gt_fixed_image_landmarks_world,
                                                                         spacing=fixed_image_itk.GetSpacing(),
                                                                         origin=fixed_image_itk.GetOrigin())

        gt_moving_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=gt_moving_image_landmarks_world,
                                                                          spacing=moving_image_itk.GetSpacing(),
                                                                          origin=moving_image_itk.GetOrigin())

        if args.out_dir is not None:
            out_dir = os.path.join(args.out_dir, pid)
        else:
            out_dir = os.path.join(pdir, 'overlay')

        if os.path.exists(out_dir) is True:
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        # 4. Overlay GT and predicted landmarks correspondences
        overlay_predicted_and_manual_landmarks(fixed_image=fixed_image_np,
                                               moving_image=moving_image_np,
                                               pred_landmarks_fixed=fixed_image_landmarks_voxels,
                                               pred_landmarks_moving=moving_image_landmarks_voxels,
                                               manual_landmarks_fixed=gt_fixed_image_landmarks_voxels,
                                               manual_landmarks_moving=gt_moving_image_landmarks_voxels,
                                               smoothed_landmarks_moving=moving_image_landmarks_smoothed_voxels,
                                               gt_projection_landmarks_moving=gt_projection_landmarks,
                                               out_dir=out_dir,
                                               verbose=False,
                                               show_gt_matches=args.show_gt_matches,
                                               show_gt_projection=args.show_gt_projection)
        # 5. Save errors
        # 5-1 Localization Error: d(X_m, T(X_f))
        euclidean_error_pred_gt = compute_euclidean_distance_between_points(moving_image_landmarks_world,
                                                                            gt_projection_landmarks_world)
        np.save(file=os.path.join(pdir,
                                  'euclidean_error_pred_gt_proj.npy'),
                arr=euclidean_error_pred_gt)

        dim_wise_erros_pred_gt = np.abs(np.subtract(moving_image_landmarks_world,
                                                    gt_projection_landmarks_world))

        np.save(file=os.path.join(pdir,
                                  'dimwise_error_pred_gt_proj.npy'),
                arr=dim_wise_erros_pred_gt)

        # 5-2 Error(smoothed, GT projection)
        if moving_image_landmarks_smoothed_world is not None:
            euclidean_error_smoothed_gt = compute_euclidean_distance_between_points(moving_image_landmarks_smoothed_world,
                                                                                    gt_projection_landmarks_world)
            np.save(file=os.path.join(pdir,
                                      'euclidean_error_smoothed_gt_proj_{}.npy'.format(args.smoothing)),
                    arr=euclidean_error_smoothed_gt)

            dim_wise_erros_smoothed_gt = np.abs(np.subtract(moving_image_landmarks_smoothed_world,
                                                            gt_projection_landmarks_world))

            np.save(file=os.path.join(pdir,
                                      'dimwise_error_smoothed_gt_proj_{}.npy'.format(args.smoothing)),
                    arr=dim_wise_erros_smoothed_gt)



