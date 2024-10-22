"""

Script to plot TRE trends for different registration configurations for comparison

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from lesionmatching.util_scripts.utils import *
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, help='Directory where all results are stored')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--folders', type=str, help='Registration configurations we want to compare' ,
                    nargs='+')
    parser.add_argument('--legends', type=str, help='Legends for registration comparisons', nargs='+')
    parser.add_argument('--output_file', type=str, help='Name of output file', default='comparison.png')
    parser.add_argument('--title', type=str, default=None, nargs='+')
    parser.add_argument('--plot_affine', action='store_true')
    parser.add_argument('--plot_pre_reg', action='store_true')
    args = parser.parse_args()

    assert(isinstance(args.folders, list))
    assert(isinstance(args.legends, list))

    if args.save_dir is None:
        save_dir = args.result_dir
    else:
        save_dir = args.save_dir
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

    # Extract patient IDs from baseline directory
    if args.dataset == 'copd':
        pids = ['copd1', 'copd2', 'copd3', 'copd4', 'copd5', 'copd6', 'copd7', 'copd8', 'copd9', 'copd10']
    else:
        raise NotImplementedError('Directory names for dataset {} not known'.format(args.dataset))

    tre_dict = {}
    tre_dict['Patient ID'] = []
    tre_dict['TRE (mm)'] = []
    tre_dict['Registration type'] = []

    affine_done = False
    # Loop over patients
    for pid in pids:
        # Loop over registration configurations
        for idx, rdir in enumerate(args.folders):
            pdir = os.path.join(args.result_dir, rdir, pid)
            if args.plot_pre_reg is True:
                if idx == 0:
                    pre_reg_tre = np.load(os.path.join(pdir, 'pre_reg_error.npy'))
                    tre_dict['Patient ID'].extend([pid for i in range(pre_reg_tre.shape[0])])
                    tre_dict['Registration type'].extend(['No registration'for i in range(pre_reg_tre.shape[0])])
                    tre_dict['TRE (mm)'].extend(list(pre_reg_tre))

            if args.plot_affine is True:
                if os.path.exists(os.path.join(pdir, 'post_affine_error.npy')) and affine_done is False:
                    affine_reg_tre = np.load(os.path.join(pdir, 'post_affine_error.npy'))
                    tre_dict['Patient ID'].extend([pid for i in range(post_reg_tre.shape[0])])
                    tre_dict['Registration type'].extend(['Elastix-Affine' for i in range(post_reg_tre.shape[0])])
                    tre_dict['TRE (mm)'].extend(list(affine_reg_tre))
                    affine_done = True

            post_reg_tre = np.load(os.path.join(pdir, 'post_reg_error.npy'))
            tre_dict['Patient ID'].extend([pid for i in range(post_reg_tre.shape[0])])
            tre_dict['Registration type'].extend([args.legends[idx] for i in range(post_reg_tre.shape[0])])
            tre_dict['TRE (mm)'].extend(list(post_reg_tre))

        affine_done = False


    # Construct pandas DF
    tre_df = pd.DataFrame.from_dict(tre_dict)

    # Use the DF to construct box-plot
    fig, ax = plt.subplots()


    sns.boxplot(data=tre_df,
                x='Patient ID',
                y='TRE (mm)',
                hue='Registration type',
                fliersize=0.1,
                ax=ax)

    if args.title is not None:
        ax.set_title(' '.join(args.title))

    ax.set_ylim((0, 50))

#    ax.tick_params(axis='both', labelsize=15)
#    ax.yaxis.label.set_size(15)

    # Adjust legend position
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              fancybox=True,
              shadow=True,
              ncol=4)

    fig.savefig(os.path.join(save_dir, args.output_file),
                bbox_inches='tight')



