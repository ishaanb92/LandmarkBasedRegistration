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

PAT = 'copd1'
slice_idx = 150
DATA_DIR = '/home/ishaan/COPDGene/mha'
MAX_VAL = 1024

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, required=True)
    args = parser.parse_args()


    diff_image_itk = sitk.ReadImage(os.path.join(args.reg_dir,
                                                 PAT,
                                                 'diff_image.mha'))

    diff_image_np = sitk.GetArrayFromImage(diff_image_itk)

    # Get fixed mask
    fixed_mask_itk = sitk.ReadImage(os.path.join(DATA_DIR,
                                                 PAT,
                                                 'lung_mask_iBHCT_dl_iso.mha'))

    fixed_mask_np = sitk.GetArrayFromImage(fixed_mask_itk)

    diff_slice = diff_image_np[slice_idx, ...]

    diff_slice_masked = diff_slice*fixed_mask_np[slice_idx, ...]

    print(np.amax(diff_slice_masked))

    diff_slice_masked_rescaled = (diff_slice_masked*255.0)/MAX_VAL

    diff_slice_masked_rescaled = diff_slice_masked_rescaled.astype(np.uint8)

    heatmap = cv2.applyColorMap(diff_slice_masked_rescaled, cv2.COLORMAP_HOT)

    cv2.imwrite(os.path.join(args.reg_dir, 'heatmap.jpg'),
                heatmap)


