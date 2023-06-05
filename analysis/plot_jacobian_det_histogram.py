"""

Script to plot Jacobian histogram distribution to gauge "realness" of estimated transformations

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lesionmatching.util_scripts.image_utils import convert_itk_to_ras_numpy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, required=True)
    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    jac_dict = {}
    jac_dict['Jacobian det'] = []
    jac_dict['Patient ID'] = []

    for pdir in pat_dirs:

        pid = pdir.split(os.sep)[-1]

        # Read the Jacobian determinant ITK image
        jac_det = sitk.ReadImage(os.path.join(pdir,
                                              'spatialJacobian.mha'))

        # The jacobian det. is computed on the fixed image domain
        fixed_image_mask = sitk.ReadImage(os.path.join(pdir,
                                                       'fixed_mask.mha'))

        jac_det_np = convert_itk_to_ras_numpy(jac_det)
        fixed_image_mask_np = convert_itk_to_ras_numpy(fixed_image_mask)

        n_lung_voxels = np.nonzero(fixed_image_mask_np)[0].shape[0]

        jac_dict['Jacobian det'].extend(list(jac_det_np[fixed_image_mask_np==1].flatten()))
        jac_dict['Patient ID'].extend([pid for i in range(n_lung_voxels)])

    # Convert dict to DF
    jac_df = pd.DataFrame.from_dict(jac_dict)

    # Plot the histogram
    fig, ax = plt.subplots()

    sns.histplot(data=jac_df,
                 x='Jacobian det',
                 hue='Patient ID',
                 ax=ax)

    ax.set_xlim((-3, 3))

    fig.savefig(os.path.join(args.reg_dir,
                             'jac_det_hist.png'))

    fig.savefig(os.path.join(args.reg_dir,
                             'jac_det_hist.pdf'))








