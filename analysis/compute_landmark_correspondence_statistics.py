"""

Script to compute landmark correspondence statistics

1. For each patient, computed the number of predicted landmark correspondences
2. Report how many of these correspondences lie inside the mask (A correspondence is inside the mask if BOTH landmarks are inside the mask)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
from argparse import ArgumentParser
import SimpleITK as sitk
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='umc')
    parser.add_argument('--vessel', action='store_true')
    parser.add_argument('--sift', action='store_true')

    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    n_landmarks = np.zeros(shape=(len(pdirs,)),
                           dtype=np.int32)

    n_landmarks_inside = np.zeros(shape=(len(pdirs,)),
                                  dtype=np.int32)

    for idx, pdir in enumerate(pdirs):

        error_found = False

        pid = pdir.split(os.sep)[-1]

        pdata_dir = os.path.join(args.data_dir, pid)

        # 1. Read the mask images
        if args.dataset == 'copd':
            if args.sift is False:
                fixed_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                             'lung_mask_iBHCT_dl_iso.mha'))

                moving_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                              'lung_mask_eBHCT_dl_iso.mha'))
            else:
                fixed_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                             'lung_mask_iBHCT_dl.mha'))

                moving_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                              'lung_mask_eBHCT_dl.mha'))
        elif args.dataset == 'umc':
            # Find the fixed and moving directories
            scan_dirs  = [f.path for f in os.scandir(pdata_dir) if f.is_dir()]

            # Figure out which idx is baseline and which is follow-up
            s_id_0 = scan_dirs[0].split(os.sep)[-1]
            s_id_1 = scan_dirs[1].split(os.sep)[-1]
            timestamp_0 = create_datetime_object_from_str(s_id_0)
            timestamp_1 = create_datetime_object_from_str(s_id_1)
            if timestamp_0 > timestamp_1: # Scan0 occurs after Scan1
                baseline_idx = 1
                followup_idx = 0
            else:
                baseline_idx=0
                followup_idx=1

            if args.vessel is False:
                fixed_mask_itk = sitk.ReadImage(os.path.join(scan_dirs[baseline_idx],
                                                             'LiverMask_dilated.nii'))

                moving_mask_itk = sitk.ReadImage(os.path.join(scan_dirs[followup_idx],
                                                              'LiverMask_dilated.nii'))
            else:
                fixed_mask_itk = sitk.ReadImage(os.path.join(scan_dirs[baseline_idx],
                                                             'vessel_mask_dilated.nii'))

                moving_mask_itk = sitk.ReadImage(os.path.join(scan_dirs[followup_idx],
                                                              'vessel_mask_dilated.nii'))


        fixed_mask_np = convert_itk_to_ras_numpy(fixed_mask_itk)
        moving_mask_np = convert_itk_to_ras_numpy(moving_mask_itk)


        # 2. Parse the .txt files into numpy arrays
        if args.dataset == 'copd':
            fixed_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                         'fixed_landmarks_elx.txt'))
            moving_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                          'moving_landmarks_elx.txt'))

            if args.sift is False:
                # 3. Convert physical coordinates into voxel indices
                fixed_image_landmarks_voxel = map_world_coord_to_voxel_index(world_coords=fixed_image_landmarks,
                                                                             spacing=fixed_mask_itk.GetSpacing(),
                                                                             origin=fixed_mask_itk.GetOrigin()).astype(np.int32)

                moving_image_landmarks_voxel = map_world_coord_to_voxel_index(world_coords=moving_image_landmarks,
                                                                              spacing=moving_mask_itk.GetSpacing(),
                                                                              origin=moving_mask_itk.GetOrigin()).astype(np.int32)
            else:
                fixed_image_landmarks_voxel = fixed_image_landmarks.astype(np.int32)
                moving_image_landmarks_voxel = moving_image_landmarks.astype(np.int32)

        elif args.dataset == 'umc':
            fixed_image_landmarks_voxel = parse_points_file(fpath=os.path.join(pdir,
                                                                               'fixed_landmarks_voxels.txt')).astype(np.int32)

            moving_image_landmarks_voxel = parse_points_file(fpath=os.path.join(pdir,
                                                                                'moving_landmarks_voxels.txt')).astype(np.int32)

        assert(fixed_image_landmarks_voxel.shape[0] == moving_image_landmarks_voxel.shape[0])
        n_landmarks[idx] = fixed_image_landmarks_voxel.shape[0]
        max_landmark_coordinates_fixed = np.max(fixed_image_landmarks_voxel, axis=0)
        max_landmark_coordinates_moving = np.max(moving_image_landmarks_voxel, axis=0)

        if max_landmark_coordinates_fixed[0] >= fixed_mask_np.shape[0] \
                or max_landmark_coordinates_fixed[1] >= fixed_mask_np.shape[1] \
                or max_landmark_coordinates_fixed[2] >= fixed_mask_np.shape[2]:
            print('Landmarks (fixed) predicted outside image domain for patient {}'.format(pid))
            error_found = True

        if max_landmark_coordinates_moving[0] >= moving_mask_np.shape[0] \
                or max_landmark_coordinates_moving[1] >= moving_mask_np.shape[1] \
                or max_landmark_coordinates_moving[2] >= moving_mask_np.shape[2]:
            print('Landmarks (moving) predicted outside image domain for patient {}'.format(pid))
            error_found = True

        if error_found is True:
            rows_to_delete = []
            # Filter the landmark correspondences
            fixed_image_landmarks_voxel_x = fixed_image_landmarks_voxel[:, 0]
            fixed_image_landmarks_voxel_y = fixed_image_landmarks_voxel[:, 1]
            fixed_image_landmarks_voxel_z = fixed_image_landmarks_voxel[:, 2]

            moving_image_landmarks_voxel_x = moving_image_landmarks_voxel[:, 0]
            moving_image_landmarks_voxel_y = moving_image_landmarks_voxel[:, 1]
            moving_image_landmarks_voxel_z = moving_image_landmarks_voxel[:, 2]

            fixed_rows_to_delete_x = np.where(fixed_image_landmarks_voxel_x >= fixed_mask_np.shape[0])
            fixed_rows_to_delete_y = np.where(fixed_image_landmarks_voxel_y >= fixed_mask_np.shape[1])
            fixed_rows_to_delete_z = np.where(fixed_image_landmarks_voxel_z >= fixed_mask_np.shape[2])

            moving_rows_to_delete_x = np.where(moving_image_landmarks_voxel_x >= moving_mask_np.shape[0])
            moving_rows_to_delete_y = np.where(moving_image_landmarks_voxel_y >= moving_mask_np.shape[1])
            moving_rows_to_delete_z = np.where(moving_image_landmarks_voxel_z >= moving_mask_np.shape[2])

            if len(fixed_rows_to_delete_x) != 0:
                rows_to_delete.extend(list(fixed_rows_to_delete_x[0]))

            if len(fixed_rows_to_delete_y) != 0:
                rows_to_delete.extend(list(fixed_rows_to_delete_y[0]))

            if len(fixed_rows_to_delete_z) != 0:
                rows_to_delete.extend(list(fixed_rows_to_delete_z[0]))

            if len(moving_rows_to_delete_x) != 0:
                rows_to_delete.extend(list(moving_rows_to_delete_x[0]))

            if len(moving_rows_to_delete_y) != 0:
                rows_to_delete.extend(list(moving_rows_to_delete_y[0]))

            if len(moving_rows_to_delete_z) != 0:
                rows_to_delete.extend(list(moving_rows_to_delete_z[0]))

            rows_to_delete = list(set(rows_to_delete)) # Get rid of duplicate entries
            rows_to_delete = np.array(rows_to_delete)

            fixed_image_landmarks_voxel = np.delete(fixed_image_landmarks_voxel,
                                                    rows_to_delete,
                                                    axis=0)

            moving_image_landmarks_voxel = np.delete(moving_image_landmarks_voxel,
                                                     rows_to_delete,
                                                     axis=0)

        # 4. Count number of landmarks inside the respective lung masks
        fixed_mask_idxs = fixed_mask_np[fixed_image_landmarks_voxel[:, 0],
                                        fixed_image_landmarks_voxel[:, 1],
                                        fixed_image_landmarks_voxel[:, 2]]

        moving_mask_idxs = moving_mask_np[moving_image_landmarks_voxel[:, 0],
                                          moving_image_landmarks_voxel[:, 1],
                                          moving_image_landmarks_voxel[:, 2]]

        correspondence_status = np.multiply(fixed_mask_idxs, moving_mask_idxs)
        landmarks_inside = np.nonzero(correspondence_status)[0].shape[0]
        frac_inside = landmarks_inside/fixed_image_landmarks_voxel.shape[0]

        n_landmarks_inside[idx] = landmarks_inside

        print('Correspondences inside soft-mask region for patient {} = {}/{} ({} %)'.format(pid,
                                                                                             landmarks_inside,
                                                                                             fixed_image_landmarks_voxel.shape[0],
                                                                                             frac_inside*100))

    n_fraction_inside = np.divide(n_landmarks_inside,
                                  n_landmarks)

    print('Overall statistics')

    print('Number of landmarks (Mean +/- std-dev) :: {} +/- {}'.format(np.mean(n_landmarks),
                                                                       np.std(n_landmarks)))

    print('Fraction of landmarks inside (Mean +/- std-dev) :: {} +/- {}'.format(np.mean(n_fraction_inside),
                                                                                np.std(n_fraction_inside)))

    print('Number of landmarks (Median, q25, q75) :: {} {}-{}'.format(np.median(n_landmarks),
                                                                      np.percentile(n_landmarks, 25),
                                                                      np.percentile(n_landmarks, 75)))

    print('Fraction of landmarks inside (Median, q25, q75) :: {} {}-{}'.format(np.median(n_fraction_inside),
                                                                               np.percentile(n_fraction_inside, 25),
                                                                               np.percentile(n_fraction_inside, 75)))
