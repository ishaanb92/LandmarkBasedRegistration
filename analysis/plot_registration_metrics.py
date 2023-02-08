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
    parser.add_argument('--result_dirs', type=str, help='Registration configurations we want to compare' ,
                    nargs='+')
    parser.add_argument('--legends', type=str, help='Legends for registration comparisons', nargs='+')
    parser.add_argument('--output_file', type=str, help='Name of output file', default='/home/ishaan/paper4/elastix_results/comparison.png')

    args = parser.parse_args()

    assert(isinstance(args.result_dirs, list))
    assert(isinstance(args.legends, list))

    baseline_dir = args.result_dirs[0]

    # Extract patient IDs from baseline directory
    pids = [f.path.split(os.sep)[-1] for f in os.scandir(baseline_dir) if f.is_dir()]

    tre_dict = {}
    tre_dict['Patient ID'] = []
    tre_dict['TRE (mm)'] = []
    tre_dict['Registration type'] = []

    # Loop over patients
    for pid in pids:
        # Pre-registration TRE
        pre_reg_tre = np.load(os.path.join(baseline_dir, pid, 'pre_reg_error.npy'))
        tre_dict['Patient ID'].extend([pid for i in range(pre_reg_tre.shape[0])])
        tre_dict['Registration type'].extend(['Pre-registration' for i in range(pre_reg_tre.shape[0])])
        tre_dict['TRE (mm)'].extend(list(pre_reg_tre))
        # Loop over registration configurations
        for idx, rdir in enumerate(args.result_dirs):
            pdir = os.path.join(rdir, pid)
            post_reg_tre = np.load(os.path.join(rdir, pid, 'post_reg_error.npy'))
            tre_dict['Patient ID'].extend([pid for i in range(pre_reg_tre.shape[0])])
            tre_dict['Registration type'].extend([args.legends[idx] for i in range(post_reg_tre.shape[0])])
            tre_dict['TRE (mm)'].extend(list(post_reg_tre))


    # Construct pandas DF
    tre_df = pd.DataFrame.from_dict(tre_dict)

    # Use the DF to construct box-plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(data=tre_df,
                x='Patient ID',
                y='TRE (mm)',
                hue='Registration type',
                ax=ax)

    fig.savefig(args.output_file,
                bbox_inches='tight')



