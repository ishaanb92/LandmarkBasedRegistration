"""

Script to create heatmap from diff images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np

PAT = 'copd10'
slice_idx = 212
DATA_DIR = '/home/ishaan/COPDGene/mha'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dirs', type=str, nargs='+')
    args = parser.parse_args()

    # Get fixed mask
    fixed_mask_itk = sitk.ReadImage(os.path.join(DATA_DIR,
                                                 PAT,
                                                 'lung_mask_iBHCT_dl_iso.mha'))

    for reg_dir in args.reg_dirs:
        diff_image_itk = sitk.ReadImage(os.path.join(reg_dir,
                                                     PAT,
                                                     'diff_image.mha'))

        diff_image_np = sitk.GetArrayFromImage(diff_image_itk)


        fixed_mask_np = sitk.GetArrayFromImage(fixed_mask_itk)

        diff_slice = diff_image_np[slice_idx, ...]

        diff_slice_masked = diff_slice*fixed_mask_np[slice_idx, ...].astype(np.float32)

        fig, ax = plt.subplots()

        pcm = ax.pcolormesh(diff_slice_masked, cmap='hot')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        pcm.set_clim(0, 1000)
        fig.colorbar(pcm, orientation='vertical')

        fig.savefig(os.path.join(reg_dir,
                                 'heatmap.png'))
