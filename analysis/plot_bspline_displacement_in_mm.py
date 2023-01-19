"""

This script is used to convert the displacement used in Eppenhof and Pluim (2019) from voxel to physical distance (in mm)
to ensure equivalence. Instead of resizing images to the same size, we wish to resample them to the same spacing so a fixed
patch size corresponds to the "same" physical space

We want the following B-spline transform parameters in mm:
    1. Control grid spacing => This decides the grid dimensions i.e. number of control points for the B-spline transform
    2. Displacement range(s) in mm

In his paper, Koen uses the following B-spline parameters for 128x128x128 images:
    Control grid size:
        * Augmentations: 2 x 2 x 2
        * Coarse: 4 x 4 x 4
        * Fine: 8 x 8 x 8

    Maximum control grid displacements:
        * Augmentations : (3.2, 6.4, 12.8)
        * Coarse : (3.2, 6.4, 12.8)
        * Fine : (3.2, 3.2, 3.2)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import SimpleITK as sitk
import numpy as np
import os
from math import floor
from lesionmatching.util_scripts.utils import *
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = '/home/ishaan/DIR-Lab/mha'

if __name__ == '__main__':


    # Control-grid spacings
    aug_cg_spacing_in_voxels = (128, 128, 128)
    coarse_cg_spacing_in_voxels = (128/(4-1), 128/(4-1), 128/(4-1))
    fine_cg_spacing_in_voxels = (128/(8-1), 128/(8-1), 128/(8-1))

    # Control-grid displacement ranges
    # The number n along an axis indicates a displacement range of [-n, n] (in voxels)
    max_aug_cg_disp_in_voxels = (3.2, 6.4, 12.8)
    max_coarse_cg_disp_in_voxels = max_aug_cg_disp_in_voxels
    max_fine_cg_disp_in_voxels = (3.2, 3.2, 3.2)

    pat_dirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

    # Maximum control grid displacement in mm
    max_aug_cg_disp_in_mm = []
    max_coarse_cg_disp_in_mm = []
    max_fine_cg_disp_in_mm = []

    # Control grid spacing in mm
    aug_cg_spacing_in_mm = []
    coarse_cg_spacing_in_mm = []
    fine_cg_spacing_in_mm = []

    img_sizes = []
    for pdir in pat_dirs:
        image_prefix = pdir.split(os.sep)[-1]
        image_fname = os.path.join(pdir, '{}_T00_smaller.mha'.format(image_prefix))
        img_itk = sitk.ReadImage(image_fname)
        spacing = list(img_itk.GetSpacing())

        # Save the size of the resampled image
        iso_img_itk = sitk.ReadImage(os.path.join(pdir,
                                                  '{}_T00_iso.mha'.format(image_prefix)))
        size = list(iso_img_itk.GetSize())

        img_sizes.append(size)

        # Convert all distances from voxel to mm
        max_aug_cg_disp_in_mm.append(convert_voxel_to_mm(voxel_units=max_aug_cg_disp_in_voxels,
                                                         spacing=spacing))

        max_coarse_cg_disp_in_mm.append(convert_voxel_to_mm(voxel_units=max_coarse_cg_disp_in_voxels,
                                                            spacing=spacing))

        max_fine_cg_disp_in_mm.append(convert_voxel_to_mm(voxel_units=max_fine_cg_disp_in_voxels,
                                                          spacing=spacing))

        aug_cg_spacing_in_mm.append(convert_voxel_to_mm(voxel_units=aug_cg_spacing_in_voxels,
                                                        spacing=spacing))

        coarse_cg_spacing_in_mm.append(convert_voxel_to_mm(voxel_units=coarse_cg_spacing_in_voxels,
                                                           spacing=spacing))

        fine_cg_spacing_in_mm.append(convert_voxel_to_mm(voxel_units=fine_cg_spacing_in_voxels,
                                                         spacing=spacing))


    img_sizes = np.array(img_sizes)

    print('Median image size after resampling: {}'.format(np.median(img_sizes, axis=0)))

    # Create plot capturing all the trends
    fig, axs = plt.subplots(2, 3,
                            figsize=(12, 8))

    # Convert all the (nested) lists into pandas DFs for easy plotting
    cols = ['X-axis', 'Y-axis', 'Z-axis']


    # Plot control grid displacements (in mm) in top row
    # 1. Control grid displacements for augmentations
    max_aug_cg_disp_df = convert_2d_datastructure_to_pandas(ds=max_aug_cg_disp_in_mm,
                                                       columns=cols)


    print('Maximum aug. cg displacement median = {}'.format(max_aug_cg_disp_df.median(axis=0)))

    sns.boxplot(data=max_aug_cg_disp_df,
                ax=axs[0, 0])

    axs[0, 0].set_title('Aug. CG displacements in mm')

    # 2. Control grid displacements for coarse deformations
    max_coarse_cg_disp_df = convert_2d_datastructure_to_pandas(ds=max_coarse_cg_disp_in_mm,
                                                                columns=cols)

    print('Maximum coarse cg displacement median = {}'.format(max_coarse_cg_disp_df.median(axis=0)))

    sns.boxplot(data=max_coarse_cg_disp_df,
                ax=axs[0, 1])

    axs[0, 1].set_title('Coarse CG displacements in mm')

    # 3. Control grid displacements for fine deformations
    max_fine_cg_disp_df = convert_2d_datastructure_to_pandas(ds=max_fine_cg_disp_in_mm,
                                                              columns=cols)

    print('Maximum fine cg displacement median = {}'.format(max_fine_cg_disp_df.median(axis=0)))
    sns.boxplot(data=max_fine_cg_disp_df,
                ax=axs[0, 2])

    axs[0, 2].set_title('Fine CG displacements in mm')

    # Plot control grid spacings (in mm) in bottom row
    # 1. Control grid spacings for augmentations
    aug_cg_spacing_df = convert_2d_datastructure_to_pandas(ds=aug_cg_spacing_in_mm,
                                                           columns=cols)

    print('Aug. cg spacing median = {}'.format(aug_cg_spacing_df.median(axis=0)))
    sns.boxplot(data=aug_cg_spacing_df,
                ax=axs[1, 0])

    axs[1, 0].set_title('Aug. CG spacings in mm')

    # 2. Control grid spacings for coarse deformations
    coarse_cg_spacing_df = convert_2d_datastructure_to_pandas(ds=coarse_cg_spacing_in_mm,
                                                              columns=cols)

    print('Coarse cg spacing median = {}'.format(coarse_cg_spacing_df.median(axis=0)))
    sns.boxplot(data=coarse_cg_spacing_df,
                ax=axs[1, 1])

    axs[1, 1].set_title('Coarse CG spacings in mm')

    # 3. Control grid spacings for fine deformations
    fine_cg_spacing_df = convert_2d_datastructure_to_pandas(ds=fine_cg_spacing_in_mm,
                                                              columns=cols)

    print('Fine cg spacing median = {}'.format(fine_cg_spacing_df.median(axis=0)))
    sns.boxplot(data=fine_cg_spacing_df,
                ax=axs[1, 2])

    axs[1, 2].set_title('Fine CG spacings in mm')

    fig.savefig(os.path.join(DATA_DIR, 'bspline_params_in_mm.png'),
                bbox_inches='tight')




