""""

Script to compute the mean DWI MR image for a patient

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""


DATA_DIR = '/home/ishaan/UMC_Data/follow_up_scans'
IMG_TYPE = 'DCE'

import os
import numpy as np
import SimpleITK as sitk

def compute_mean_image(img, label):

    assert(isinstance(img, sitk.Image))
    assert(isinstance(label, sitk.Image))

    img_np = sitk.GetArrayFromImage(img)

    # Compute mean over b-values
    img_np_mean = np.mean(img_np, axis=0)

    img_itk_mean = sitk.GetImageFromArray(img_np_mean)

    # Populate metadata fields
    img_itk_mean.SetOrigin(label.GetOrigin())
    img_itk_mean.SetDirection(label.GetDirection())
    img_itk_mean.SetSpacing(label.GetSpacing())

    return img_itk_mean


if __name__ == '__main__':

    pat_dirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

    for p_dir in pat_dirs:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            print('Processing: {}'.format(s_dir))
            try:
                if IMG_TYPE == 'DCE':
                    img = sitk.ReadImage(os.path.join(s_dir, 'e-THRIVE_reg.nii'))
                elif IMG_TYPE == 'DWI':
                    img = sitk.ReadImage(os.path.join(s_dir, 'DWI_reg.nii'))
            except RuntimeError:
                print('Image not found in {}'.format(s_dir))
                continue

            label = sitk.ReadImage(os.path.join(s_dir, 'LiverMask.nii'))
            mean_img = compute_mean_image(img=img,
                                          label=label)

            if IMG_TYPE == 'DCE':
                sitk.WriteImage(mean_img,
                                os.path.join(s_dir, 'DCE_mean.nii'))
            elif IMG_TYPE == 'DWI':
                sitk.WriteImage(mean_img,
                                os.path.join(s_dir, 'DWI_reg_mean.nii'))



