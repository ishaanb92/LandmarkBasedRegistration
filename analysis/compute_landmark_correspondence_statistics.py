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

DATA_DIR = '/home/ishaan/COPDGene/mha'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)

    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    n_landmarks = np.zeros(shape=(len(pdirs,)),
                           dtype=np.int32)

    n_landmarks_inside = np.zeros(shape=(len(pdirs,)),
                                  dtype=np.int32)

    for idx, pdir in enumerate(pdirs):

        pid = pdir.split(os.sep)[-1]

        pdata_dir = os.path.join(DATA_DIR, pid)

        # 1. Read the mask images
        fixed_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                     'lung_mask_iBHCT_dl_iso.mha'))

        moving_mask_itk = sitk.ReadImage(os.path.join(pdata_dir,
                                                      'lung_mask_eBHCT_dl_iso.mha'))

        fixed_mask_np = convert_itk_to_ras_numpy(fixed_mask_itk)
        moving_mask_np = convert_itk_to_ras_numpy(moving_mask_itk)


        # 2. Parse the .txt files into numpy arrays
        fixed_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                     'fixed_landmarks_elx.txt'))
        moving_image_landmarks = parse_points_file(fpath=os.path.join(pdir,
                                                                      'moving_landmarks_elx.txt'))

        assert(fixed_image_landmarks.shape[0] == moving_image_landmarks.shape[0])
        n_landmarks[idx] = fixed_image_landmarks.shape[0]

        # 3. Convert physical coordinates into voxel indices
        fixed_image_landmarks_voxel = map_world_coord_to_voxel_index(world_coords=fixed_image_landmarks,
                                                                     spacing=fixed_mask_itk.GetSpacing(),
                                                                     origin=fixed_mask_itk.GetOrigin()).astype(np.int32)

        moving_image_landmarks_voxel = map_world_coord_to_voxel_index(world_coords=moving_image_landmarks,
                                                                      spacing=moving_mask_itk.GetSpacing(),
                                                                      origin=moving_mask_itk.GetOrigin()).astype(np.int32)

        # 4. Count number of landmarks inside the respective lung masks
        fixed_mask_idxs = fixed_mask_np[fixed_image_landmarks_voxel[:, 0],
                                        fixed_image_landmarks_voxel[:, 1],
                                        fixed_image_landmarks_voxel[:, 2]]

        moving_mask_idxs = moving_mask_np[moving_image_landmarks_voxel[:, 0],
                                          moving_image_landmarks_voxel[:, 1],
                                          moving_image_landmarks_voxel[:, 2]]

        correspondence_status = np.multiply(fixed_mask_idxs, moving_mask_idxs)
        landmarks_inside = np.nonzero(correspondence_status)[0].shape[0]
        frac_inside = landmarks_inside/fixed_image_landmarks.shape[0]

        n_landmarks_inside[idx] = landmarks_inside

        print('Correspondences inside lung for patient {} = {}/{} ({} %)'.format(pid,
                                                                               landmarks_inside,
                                                                               fixed_image_landmarks.shape[0],
                                                                               frac_inside*100))

    n_fraction_inside = np.divide(n_landmarks_inside,
                                  n_landmarks)

    print('Overall statistics')

    print('Number of landmarks :: {} +/- {}'.format(np.mean(n_landmarks),
                                                    np.std(n_landmarks)))
    print('Fraction of landmarks inside :: {} +/- {}'.format(np.mean(n_fraction_inside),
                                                             np.std(n_fraction_inside)))

