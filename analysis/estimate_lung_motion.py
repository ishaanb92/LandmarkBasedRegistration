"""

Estimate (deformable) lung motion by fitting a thin-plate spline between the manual landmarks
This estimation depends on the inital affine alignment

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
    parser.add_argument('--deformation_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--affine_reg_dir', type=str, required=True)
    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.deformation_dir) if f.is_dir()]

    disp_dict = {}
    disp_dict['X-disp'] = []
    disp_dict['Y-disp'] = []
    disp_dict['Z-disp'] = []
    disp_dict['Euclidean'] = []
    disp_dict['Patient ID'] = []

    for pdir in pdirs:
        pid = pdir.split(os.sep)[-1]

        patient_data_dir = os.path.join(args.data_dir, pid)
        patient_affine_reg_dir = os.path.join(args.affine_reg_dir, pid)

        print('Estimating displacements for patient {}'.format(pid))

        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        # Fixed image mask
        fixed_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                                'lung_mask_T00_dl_iso.mha'))

        fixed_image_lung_mask_np = convert_itk_to_ras_numpy(fixed_image_lung_mask_itk)

        # Moving image mask
        moving_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_affine_reg_dir,
                                                                 'moving_lung_mask_affine',
                                                                 'result.mha'))

        moving_image_lung_mask_np = convert_itk_to_ras_numpy(moving_image_lung_mask_itk)

        # FIXME: This is ugly and hacky. Saving the padded mask after test.py completes is a
        # cleaner fix.
        # Pad masks to match saved image dims (padded for compliance with MONAI inferers)
        i, j, k = fixed_image_lung_mask_np.shape

        if i%8 != 0 and j%8 !=0:
            excess_pixels_xy = 8 - (i%8)
        else:
            excess_pixels_xy = 0

        if k%8 != 0:
            excess_pixels_z = 8 - (k%8)
        else:
            excess_pixels_z = 0

        fixed_image_lung_mask_np = np.pad(fixed_image_lung_mask_np,
                                          ((0, excess_pixels_xy),
                                           (0, excess_pixels_xy),
                                           (0, excess_pixels_z)))

        moving_image_lung_mask_np = np.pad(moving_image_lung_mask_np,
                                           ((0, excess_pixels_xy),
                                           (0, excess_pixels_xy),
                                           (0, excess_pixels_z)))

        # Construct super-mask using a logical OR between fixed and moving masks
        # because we are only interested in displacements of voxels inside the lung
        relevant_displacement_mask = np.where(fixed_image_lung_mask_np+moving_image_lung_mask_np>=1,
                                              1,
                                              0).astype(np.float32)

        n_lung_voxels = np.nonzero(relevant_displacement_mask)[0].shape[0]

        # 1. Read the TPS transformed grid
        tps_grid = np.load(os.path.join(pdir,
                                        'tps_transformed_grid_gt.npy'))
        ndim, X, Y, Z = tps_grid.shape

        grid = np.array(np.meshgrid(np.linspace(0, 1, X),
                                    np.linspace(0, 1, Y),
                                    np.linspace(0, 1, Z),
                                    indexing="ij"),
                        dtype=np.float32)

        # 2. Calculate displacement grid
        disp_grid = np.subtract(tps_grid,
                                grid)

        # 3. Scale the displacement grid from [0, 1] to image dims
        fixed_image_dims = np.array(fixed_image_itk.GetSize())[:, None, None, None]

        disp_grid_scaled = np.multiply(disp_grid,
                                       fixed_image_dims)

        # FIXME: Skip voxel units -> mm conversion because of 1mm isotropic spacing
        X_disp = disp_grid_scaled[0, ...]
        Y_disp = disp_grid_scaled[1, ...]
        Z_disp = disp_grid_scaled[2, ...]


        euc_disp = np.sqrt(np.power(X_disp, 2) + np.power(Y_disp, 2) + np.power(Z_disp, 2))

        disp_dict['X-disp'].extend(list(X_disp[relevant_displacement_mask==1].flatten()))
        disp_dict['Y-disp'].extend(list(Y_disp[relevant_displacement_mask==1].flatten()))
        disp_dict['Z-disp'].extend(list(Z_disp[relevant_displacement_mask==1].flatten()))

        disp_dict['Euclidean'].extend(list(euc_disp[relevant_displacement_mask==1].flatten()))
        disp_dict['Patient ID'].extend([pid for i in range(n_lung_voxels)])

    # Construct pandas DF for easy plotting
    disp_df = pd.DataFrame.from_dict(disp_dict)

    # Plot histograms
    fig, axs = plt.subplots(2, 2,
                            figsize=(20, 20))

    sns.histplot(data=disp_df,
                 x='X-disp',
                 hue='Patient ID',
                 ax=axs[0, 0])

    sns.histplot(data=disp_df,
                 x='Y-disp',
                 hue='Patient ID',
                 ax=axs[0, 1])

    sns.histplot(data=disp_df,
                 x='Z-disp',
                 hue='Patient ID',
                 ax=axs[1, 0])

    sns.histplot(data=disp_df,
                 x='Euclidean',
                 hue='Patient ID',
                 ax=axs[1, 1])

    fig.savefig(os.path.join(args.deformation_dir,
                             'estimated_displacements.png'),
                bbox_inches='tight')


