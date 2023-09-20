"""

Script to compute rescaling statistics for COPD in the absence of lung masks during inference.

We use median of the maximum and minimum intensities (inside the lung) for the DIR-Lab dataset

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import SimpleITK as sitk
import numpy as np
from lesionmatching.util_scripts.image_utils import get_min_max_from_image
import joblib

DIRLAB_DIR = '/home/ishaan/DIR-Lab/mha'
COPD_DIR = '/home/ishaan/COPDGene/mha'
DATASET = 'COPD'

if __name__ == '__main__':


    if DATASET == 'DIRLAB':
        pat_dirs = [f.path for f in os.scandir(DIRLAB_DIR) if f.is_dir()]
    elif DATASET == 'COPD':
        pat_dirs = [f.path for f in os.scandir(COPD_DIR) if f.is_dir()]

    fixed_image_max = []
    moving_image_max = []

    fixed_image_min = []
    moving_image_min = []


    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]
        if DATASET == 'DIRLAB':
            fixed_image_fname = '{}_T00_iso.mha'.format(pid)
            moving_image_fname = '{}_T50_iso.mha'.format(pid)
            fixed_mask_fname = 'lung_mask_T00_dl_iso.mha'
            moving_mask_fname = 'lung_mask_T50_dl_iso.mha'
        elif DATASET == 'COPD':
            fixed_image_fname = '{}_iBHCT_iso.mha'.format(pid)
            moving_image_fname = '{}_eBHCT_iso.mha'.format(pid)
            fixed_mask_fname = 'lung_mask_iBHCT_dl_iso.mha'
            moving_mask_fname = 'lung_mask_eBHCT_dl_iso.mha'

        fixed_image = sitk.ReadImage(os.path.join(pdir,
                                                  fixed_image_fname))

        moving_image = sitk.ReadImage(os.path.join(pdir,
                                                   moving_image_fname))

        fixed_mask = sitk.ReadImage(os.path.join(pdir,
                                                 fixed_mask_fname))

        moving_mask = sitk.ReadImage(os.path.join(pdir,
                                                  moving_mask_fname))

        fixed_max, fixed_min = get_min_max_from_image(image=fixed_image,
                                                      mask=fixed_mask)

        moving_max, moving_min = get_min_max_from_image(image=moving_image,
                                                        mask=moving_mask)

        fixed_image_max.append(fixed_max)
        fixed_image_min.append(fixed_min)
        moving_image_max.append(moving_max)
        moving_image_min.append(moving_min)


    image_stats = {}

    image_stats['fixed_image_max'] = np.median(np.array(fixed_image_max))
    image_stats['moving_image_max'] = np.median(np.array(moving_image_max))
    image_stats['fixed_image_min'] = np.median(np.array(fixed_image_min))
    image_stats['moving_image_min'] = np.median(np.array(moving_image_min))

    if DATASET == 'DIRLAB':
        joblib.dump(image_stats,
                    os.path.join(COPD_DIR, 'rescaling_stats.pkl'))
    elif DATASET == 'COPD':
        joblib.dump(image_stats,
                    os.path.join(COPD_DIR, 'rescaling_stats_copd.pkl'))



