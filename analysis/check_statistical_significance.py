"""

Check statistical significance of difference in TRE for a pair of registration methods

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
from argparse import ArgumentParser
from scipy.stats import wilcoxon
import pandas as pd
from tabulate import tabulate

ALPHA = 0.05

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--result_dirs', type=str, help='Registration configurations we want to compare' ,
                    nargs='+')

    parser.add_argument('--legends', type=str, help='Legends' ,
                    nargs='+')

    args = parser.parse_args()


    assert(len(args.result_dirs) == 2)

    pids = [f.path.split(os.sep)[-1] for f in os.scandir(args.result_dirs[0]) if f.is_dir()]


    significance_dict = {}
    significance_dict['Patient ID'] = []
    significance_dict['p-value'] = []
    significance_dict['Significant'] = []
    significance_dict['{} median TRE'.format(args.legends[0])] = []
    significance_dict['{} median TRE'.format(args.legends[1])] = []
    significance_dict['Who is better?'] = []

    for pid in pids:

        significance_dict['Patient ID'].append(pid)

        tre_1 = np.load(os.path.join(args.result_dirs[0], pid, 'post_reg_error.npy'))
        tre_2 = np.load(os.path.join(args.result_dirs[1], pid, 'post_reg_error.npy'))

        # Compute test statistic and p-value
        _, p_value = wilcoxon(tre_1, tre_2)
        significance_dict['p-value'].append(p_value)

        if p_value < ALPHA:
            significance_dict['Significant'].append('Yes')
            if np.median(tre_1) < np.median(tre_2):
                significance_dict['Who is better?'].append(args.legends[0])
            else:
                significance_dict['Who is better?'].append(args.legends[1])
        else:
            significance_dict['Significant'].append('No')
            significance_dict['Who is better?'].append('N/A')

        significance_dict['{} median TRE'.format(args.legends[0])].append(np.median(tre_1))
        significance_dict['{} median TRE'.format(args.legends[1])].append(np.median(tre_2))


    sig_df = pd.DataFrame.from_dict(significance_dict)
    print(tabulate(sig_df, headers='keys', tablefmt='psql'))

