"""

Script to visualize GT landmarks pairs and (DL-)predicted landmark pairs on the same image.


@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from argparse import ArgumentParser
import SimpleITK as sitk
from lesionmatching.analysis.visualize import *
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str)
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--points_dir', type=str, default='/home/ishaan/COPDGene/points')
    parser.add_argument('--affine_reg_dir', type=str, help='Affine registration directory contains GT moving image landmarks')

    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]

        print('Processing patient {}'.format(pid))

        affine_pdir = os.path.join(args.affine_reg_dir, pid)
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

        fixed_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                        'fixed_landmarks_elx.txt'))

        moving_image_landmarks_world = parse_points_file(os.path.join(pdir,
                                                         'moving_landmarks_elx.txt'))

        # 2-b. Convert world coordinates to voxels

        fixed_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=fixed_image_landmarks_world,
                                                                      spacing=fixed_image_itk.GetSpacing(),
                                                                      origin=fixed_image_itk.GetOrigin())

        moving_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=moving_image_landmarks_world,
                                                                       spacing=moving_image_itk.GetSpacing(),
                                                                       origin=moving_image_itk.GetOrigin())

        # 3-a. Read and fixed and (affine-transformed) moving GT landmarks .txt files
        if args.dataset == 'copd':
            gt_fixed_image_landmarks_world = parse_points_file(os.path.join(points_pdir,
                                                                            '{}_300_iBH_world_r1_elx.txt'.format(pid)))
        elif args.dataset == 'dirlab':
            raise NotImplementedError

        gt_moving_image_landmarks_world = parse_points_file(os.path.join(affine_pdir,
                                                                         'transformed_moving_landmarks_elx.txt'))

        # 3-b. Convert world coordinates to voxels
        gt_fixed_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=gt_fixed_image_landmarks_world,
                                                                         spacing=fixed_image_itk.GetSpacing(),
                                                                         origin=fixed_image_itk.GetOrigin())

        gt_moving_image_landmarks_voxels = map_world_coord_to_voxel_index(world_coords=gt_moving_image_landmarks_world,
                                                                          spacing=moving_image_itk.GetSpacing(),
                                                                          origin=moving_image_itk.GetOrigin())

        # 4. Overlay GT and predicted landmarks correspondences
        overlay_predicted_and_manual_landmarks(fixed_image=fixed_image_np,
                                               moving_image=moving_image_np,
                                               pred_landmarks_fixed=fixed_image_landmarks_voxels,
                                               pred_landmarks_moving=moving_image_landmarks_voxels,
                                               manual_landmarks_fixed=gt_fixed_image_landmarks_voxels,
                                               manual_landmarks_moving=gt_moving_image_landmarks_voxels,
                                               out_dir=os.path.join(pdir, 'overlay'))

