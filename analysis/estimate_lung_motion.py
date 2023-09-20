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
    parser.add_argument('--deformation_dir', type=str, required=True, help='Directory with manual landmarks')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--affine_reg_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='dirlab')
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

        if args.affine_reg_dir is not None:
            patient_affine_reg_dir = os.path.join(args.affine_reg_dir, pid)
        else:
            patient_affine_reg_dir = None

        print('Estimating displacements for patient {}'.format(pid))

        if args.dataset == 'dirlab':
            fixed_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_T00_iso.mha'.format(pid)))
        elif args.dataset == 'copd':
            fixed_image_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                          '{}_iBHCT_iso.mha'.format(pid)))

        # Fixed image mask
        if args.dataset == 'dirlab':
            fixed_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                                    'lung_mask_T00_dl_iso.mha'))
        elif args.dataset == 'copd':
            fixed_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                                    'lung_mask_iBHCT_dl_iso.mha'))

        fixed_image_lung_mask_np = convert_itk_to_ras_numpy(fixed_image_lung_mask_itk)

        # Moving image mask
        if patient_affine_reg_dir is not None:
            moving_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_affine_reg_dir,
                                                                     'moving_lung_mask_affine',
                                                                     'result.mha'))
        else:
            if args.dataset == 'dirlab':
                moving_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                                        'lung_mask_T50_dl_iso.mha'))
            else:
                moving_image_lung_mask_itk = sitk.ReadImage(os.path.join(patient_data_dir,
                                                                        'lung_mask_eBHCT_dl_iso.mha'))

        moving_image_lung_mask_np = convert_itk_to_ras_numpy(moving_image_lung_mask_itk)

        relevant_displacement_mask = fixed_image_lung_mask_np
        n_lung_voxels = np.nonzero(relevant_displacement_mask)[0].shape[0]

        # 1. Read the TPS transformed grid (NOTE: The TPS captures only deformable changes)
        if args.dataset == 'dirlab':
            tps_grid = np.load(os.path.join(pdir,
                                            'tps_transformed_grid_gt_only_bspline.npy'))
        elif args.dataset == 'copd':
            #NOTE: Trying to model only the deformable motion using a thin-plate splines results in transformation with extremely large Jacobians.
            # Therefore, we model a thin-plate spline using the manual landmarks in the original fixed and moving domains => This captures both affine and deformable motion
            tps_grid = np.load(os.path.join(pdir,
                                            'tps_transformed_grid_gt_all.npy'))
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

#        disp_grid_scaled = np.multiply(disp_grid,
#                                       fixed_image_dims)

        # NOTE: Disable scaling of displacement, will be easier to model equivalent deformations in patches
        disp_grid_scaled = disp_grid
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

    x_disp_99p = np.percentile(disp_df['X-disp'].to_numpy(),
                               q=99)
    y_disp_99p = np.percentile(disp_df['Y-disp'].to_numpy(),
                               q=99)
    z_disp_99p = np.percentile(disp_df['Z-disp'].to_numpy(),
                               q=99)
    euc_disp_99p = np.percentile(disp_df['Euclidean'].to_numpy(),
                                 q=99)

    x_disp_1p = np.percentile(disp_df['X-disp'].to_numpy(),
                               q=1)
    y_disp_1p = np.percentile(disp_df['Y-disp'].to_numpy(),
                               q=1)
    z_disp_1p = np.percentile(disp_df['Z-disp'].to_numpy(),
                               q=1)
    euc_disp_1p = np.percentile(disp_df['Euclidean'].to_numpy(),
                                 q=1)

    # Plot histograms
    fig, axs = plt.subplots(2, 2,
                            figsize=(20, 20))

    sns.histplot(data=disp_df,
                 x='X-disp',
                 hue='Patient ID',
                 ax=axs[0, 0])

    axs[0, 0].axvline(x=x_disp_1p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')

    axs[0, 0].axvline(x=x_disp_99p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')


    sns.histplot(data=disp_df,
                 x='Y-disp',
                 hue='Patient ID',
                 ax=axs[0, 1])

    axs[0, 1].axvline(x=y_disp_1p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')

    axs[0, 1].axvline(x=y_disp_99p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')


    sns.histplot(data=disp_df,
                 x='Z-disp',
                 hue='Patient ID',
                 ax=axs[1, 0])

    axs[1, 0].axvline(x=z_disp_1p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')

    axs[1, 0].axvline(x=z_disp_99p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')


    sns.histplot(data=disp_df,
                 x='Euclidean',
                 hue='Patient ID',
                 ax=axs[1, 1])

    axs[1, 1].axvline(x=euc_disp_1p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')

    axs[1, 1].axvline(x=euc_disp_99p,
                      ymin=0,
                      ymax=1,
                      c='r',
                      linestyle='--')


    fig.savefig(os.path.join(args.deformation_dir,
                             'estimated_displacements.png'),
                bbox_inches='tight')

    disp_df.to_pickle(os.path.join(args.deformation_dir,
                                   'displacement_df.pkl'))




    print('99 percentile :: X-disp = {} Y-disp = {} Z-disp = {} Euclidean = {}'.format(x_disp_99p,
                                                                                       y_disp_99p,
                                                                                       z_disp_99p,
                                                                                       euc_disp_99p))

    print('1 percentile :: X-disp = {} Y-disp = {} Z-disp = {} Euclidean = {}'.format(x_disp_1p,
                                                                                      y_disp_1p,
                                                                                      z_disp_1p,
                                                                                      euc_disp_1p))


