"""

Script to plot landmark correspondence errors.

To compute landmark correspondence, we do not know the "true" deformation betwee the fixed and moving images. However, this can be estimated by fitting a thin-plate spline between the manual landmarks. We use this spline to create "ground truth" projections of predicted landmarks in the fixed image. The distance between this projection and the corresponding (predicted/smoothed) landmark in the moving image is used to calculate the error.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--smoothing_terms', type=float, nargs='+')


    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    error_dict = {}
    error_dict['Patient ID'] = []
    error_dict['Error (mm)'] = []
    error_dict['X-error (mm)'] = []
    error_dict['Y-error (mm)'] = []
    error_dict['Z-error (mm)'] = []
    error_dict['Landmark type'] = []

    for pdir in pdirs:
        pid = pdir.split(os.sep)[-1]
        # Error: Pred <-> GT projection
        euc_pred_error = np.load(os.path.join(pdir,
                                              'euclidean_error_pred_gt_proj.npy'))

        dim_wise_pred_error = np.load(os.path.join(pdir,
                                                   'dimwise_error_pred_gt_proj.npy'))

        error_dict['Patient ID'].extend([pid for i in range(euc_pred_error.shape[0])])
        error_dict['Error (mm)'].extend(list(euc_pred_error))
        error_dict['X-error (mm)'].extend(list(dim_wise_pred_error[:, 0]))
        error_dict['Y-error (mm)'].extend(list(dim_wise_pred_error[:, 1]))
        error_dict['Z-error (mm)'].extend(list(dim_wise_pred_error[:, 2]))
        error_dict['Landmark type'].extend(['DL-based' for i in range(euc_pred_error.shape[0])])

        for sterm in args.smoothing_terms:
            euc_pred_error = np.load(os.path.join(pdir,
                                                  'euclidean_error_smoothed_gt_proj_{}.npy'.format(sterm)))

            dim_wise_pred_error = np.load(os.path.join(pdir,
                                                       'dimwise_error_smoothed_gt_proj_{}.npy'.format(sterm)))

            error_dict['Patient ID'].extend([pid for i in range(euc_pred_error.shape[0])])
            error_dict['Error (mm)'].extend(list(euc_pred_error))
            error_dict['X-error (mm)'].extend(list(dim_wise_pred_error[:, 0]))
            error_dict['Y-error (mm)'].extend(list(dim_wise_pred_error[:, 1]))
            error_dict['Z-error (mm)'].extend(list(dim_wise_pred_error[:, 2]))
            error_dict['Landmark type'].extend(['Smoothing = {}'.format(sterm) for i in range(euc_pred_error.shape[0])])

    # Create pandas DF
    # Construct pandas DF
    error_df = pd.DataFrame.from_dict(error_dict)

    # Error
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(data=error_df,
                x='Patient ID',
                y='Error (mm)',
                hue='Landmark type',
                ax=ax)

    fig.savefig(os.path.join(args.landmarks_dir, 'euc_errors.png'),
                bbox_inches='tight')

    # X-Error
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(data=error_df,
                x='Patient ID',
                y='X-error (mm)',
                hue='Landmark type',
                ax=ax)

    fig.savefig(os.path.join(args.landmarks_dir, 'x_errors.png'),
                bbox_inches='tight')

    # Y-Error
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(data=error_df,
                x='Patient ID',
                y='Y-error (mm)',
                hue='Landmark type',
                ax=ax)

    fig.savefig(os.path.join(args.landmarks_dir, 'y_errors.png'),
                bbox_inches='tight')

    # Z-Error
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(data=error_df,
                x='Patient ID',
                y='Z-error (mm)',
                hue='Landmark type',
                ax=ax)

    fig.savefig(os.path.join(args.landmarks_dir, 'z_errors.png'),
                bbox_inches='tight')
