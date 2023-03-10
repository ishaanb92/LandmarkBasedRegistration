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


if __name__ == '__main__':


    pat_dirs = [f.path for f in os.scandir(DIRLAB_DIR) if f.is_dir()]

    fixed_image_max = []
    moving_image_max = []

    fixed_image_min = []
    moving_image_min = []


    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]
        fixed_image = sitk.ReadImage(os.path.join(pdir,
                                                  '{}_T00_iso.mha'.format(pid)))

        moving_image = sitk.ReadImage(os.path.join(pdir,
                                                   '{}_T50_iso_affine.mha'.format(pid)))

        fixed_mask = sitk.ReadImage(os.path.join(pdir,
                                                 'lung_mask_T00_dl_iso.mha'))

        moving_mask = sitk.ReadImage(os.path.join(pdir,
                                                  'lung_mask_T50_dl_iso_affine.mha'))

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

    joblib.dump(image_stats,
                os.path.join(COPD_DIR, 'rescaling_stats.pkl'))


