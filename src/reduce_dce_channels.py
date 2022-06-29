"""

Script to reduce DCE channels from 16 to 6

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl


"""

import numpy as np
import SimpleITK as sitk
import os


DATA_DIR = '/home/ishaan/UMC_Data/follow_up_scans'

def reduce_image_channels(dce_img, label, save_dir=None):

    assert(isinstance(dce_img, sitk.Image))

    origin = label.GetOrigin()
    spacing = label.GetSpacing()
    direction = label.GetDirection()

    dce_np = sitk.GetArrayFromImage(dce_img)

    channels, d, h, w = dce_np.shape

    dce_reduced_np = np.zeros((6, d, h, w), dtype=np.float32)

    dce_reduced_np[0, ...] = dce_np[0, ...]
    dce_reduced_np[1, ...] = np.mean(dce_np[1:6, ...], axis=0).astype(np.float32)
    dce_reduced_np[2, ...] = np.mean(dce_np[6:10, ...], axis=0).astype(np.float32)
    dce_reduced_np[3, ...] = np.mean(dce_np[10:12, ...], axis=0).astype(np.float32)
    dce_reduced_np[4, ...] = np.mean(dce_np[12:15, ...], axis=0).astype(np.float32)
    dce_reduced_np[5, ...] = dce_np[15, ...].astype(np.float32)

    for chidx in range(6):
        temp_itk = sitk.GetImageFromArray(dce_reduced_np[chidx, ...])
        temp_itk.SetOrigin(origin)
        temp_itk.SetSpacing(spacing)
        temp_itk.SetDirection(direction)
        sitk.WriteImage(temp_itk, os.path.join(save_dir, 'DCE_channel_{}.nii'.format(chidx)))

if __name__ == '__main__':

    pat_dirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]


    for p_dir in pat_dirs:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            print('Processing: {}'.format(s_dir))
            dce_img = sitk.ReadImage(os.path.join(s_dir, 'e-THRIVE_reg.nii'))
            label = sitk.ReadImage(os.path.join(s_dir, 'LiverMask.nii'))
            reduce_image_channels(dce_img, label, save_dir=s_dir)
