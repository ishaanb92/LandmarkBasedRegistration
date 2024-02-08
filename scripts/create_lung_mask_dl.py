"""

Script to use trained DL model to create good lung masks!
Repository link: https://github.com/JoHof/lungmask

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import SimpleITK as sitk
from lungmask import mask
import os
from argparse import ArgumentParser
import numpy as np

FORMAT = 'mha'

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    case_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    if args.dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']
    elif args.dataset == 'dirlab':
        im_types = ['T00', 'T50']

    for cdir in case_dirs:
        cid = cdir.split(os.sep)[-1]
        for itype in im_types:
            img_file = os.path.join(cdir, '{}_{}.{}'.format(cid, itype, FORMAT))
            img = sitk.ReadImage(img_file)
            spacing = img.GetSpacing()
            direction = img.GetDirection()
            origin = img.GetOrigin()
            # Obtain lung mask prediction
            lung_mask_np = mask.apply(img)
            lung_mask_np = np.where(lung_mask_np > 1, 1, lung_mask_np)
            # Convert lung mask tensor to ITK and append image metadata
            lung_mask_itk = sitk.GetImageFromArray(lung_mask_np)
            lung_mask_itk.SetSpacing(spacing)
            lung_mask_itk.SetDirection(direction)
            lung_mask_itk.SetOrigin(origin)
            sitk.WriteImage(lung_mask_itk,
                            os.path.join(cdir, 'lung_mask_{}_dl.mha'.format(itype)))

