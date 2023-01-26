""""

Script to resample DIR-Lab images to isotropic spacing (1.0mm, 1.0mm, 1.0mm)
This resamples the images to the spacing used to train the landmark NN
These resampled image files will be used to evaluate elastix-based registraion

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from argparse import ArgumentParser
from lesionmatching.util_scripts.image_utils import resample_itk_image_to_new_spacing
import SimpleITK as sitk
from elastix.elastix_interface import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'dirlab':
        image_types = ['T00', 'T50']
    elif args.dataset == 'copd':
        image_types = ['iBHCT', 'eBHCT']

    # Collect all the patient directories
    pat_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    for pdir in pat_dirs:
        image_prefix = pdir.split(os.sep)[-1]
        # Resample images and masks
        for imtype in image_types:
            img_itk = sitk.ReadImage(os.path.join(pdir, '{}_{}.mha'.format(image_prefix,
                                                                           imtype)))

            resampled_img_itk = resample_itk_image_to_new_spacing(image=img_itk,
                                                                  new_spacing=(1.0, 1.0, 1.0),
                                                                  interp_order=3)

            sitk.WriteImage(resampled_img_itk,
                            os.path.join(pdir, '{}_{}_iso.mha'.format(image_prefix,
                                                                      imtype)))
