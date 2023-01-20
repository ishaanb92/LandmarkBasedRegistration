"""

Script to use trained DL model to create good lung masks!
Repository link: https://github.com/JoHof/lungmask

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import SimpleITK as sitk
from lungmask import mask
import os

DATA_DIR = '/home/ishaan/DIR-Lab/mha'

if __name__ == '__main__':

    case_dirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

    im_types = ['T00', 'T50']
    for cdir in case_dirs:
        cid = cdir.split(os.sep)[-1]
        for itype in im_types:
            img_file = os.path.join(cdir, '{}_{}_iso.mha'.format(cid, itype))
            img = sitk.ReadImage(img_file)
            spacing = img.GetSpacing()
            direction = img.GetDirection()
            origin = img.GetOrigin()
            # Obtain lung mask prediction
            lung_mask_np = mask.apply(img)
            # Convert lung mask tensor to ITK and append image metadata
            lung_mask_itk = sitk.GetImageFromArray(lung_mask_np)
            lung_mask_itk.SetSpacing(spacing)
            lung_mask_itk.SetDirection(direction)
            lung_mask_itk.SetOrigin(origin)
            sitk.WriteImage(lung_mask_itk,
                            os.path.join(cdir, 'lung_mask_{}_dl_iso.mha'.format(itype)))

