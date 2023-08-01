"""

Script that performs statistical tests over multiple configurations in a two-step way.

Step 1: Perform the Kruskal-Wallis test to see if we can reject the null hypothesis (H0: The population mean of all groups are equal)
Step 2: If p(KH) < 0.05 => this null hypothesis can be rejected. To check for pairwise differences, we perform Dunn's test with a Bonferroni correction to account for the multiple groups

See: https://stats.stackexchange.com/questions/25815/post-hoc-tests-after-kruskal-wallis-dunns-test-or-bonferroni-corrected-mann-wh

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""


import os
import numpy as np
from argparse import ArgumentParser
from scipy.stats import kruskal
import pandas as pd
from tabulate import tabulate
from scikit_posthocs import posthoc_dunn

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--result_dirs', type=str, help='Registration configurations we want to compare' ,
                    nargs='+', required=True)

    parser.add_argument('--legends', type=str, help='Legends' ,
                    nargs='+', required=True)

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    pids = [f.path.split(os.sep)[-1] for f in os.scandir(args.result_dirs[0]) if f.is_dir()]


    significance_dict = {}
    significance_dict['Patient ID'] = []
    significance_dict['Pairwise differences found'] = []

    for pid in pids: # Loop over patients
        samples = []
        significance_dict['Patient ID'].append(pid)

        for rdir in args.result_dirs:
            samples.append(np.load(os.path.join(rdir,
                                                pid,
                                                'post_reg_error.npy')))

        # Perform Kruskal-Wallis test to test for differences between groups
        _, p = kruskal(*samples)

        if p < 0.05: # Can reject KH H0
            # Perform Dunn's test
            result_df = posthoc_dunn(a=samples,
                                     p_adjust='bonferroni')

            result_np = result_df.to_numpy()

            # Get the lower triangular matrix
            result_tril = np.tril(result_np)
            result_tril = np.where(result_tril==0, 1.0, result_tril)

            # Find the s.s. different pairs
            idx_tuple = np.nonzero(result_tril < 0.05)

            n_pairs = idx_tuple[0].shape[0]

            diff_pairs = []
            for pair_id in range(n_pairs):
                diff_pairs.append((idx_tuple[0][pair_id], idx_tuple[1][pair_id]))

            diff_str = ""
            for dpair in diff_pairs:
                diff_str += "({}, {}), ".format(args.legends[dpair[0]], args.legends[dpair[1]])

            significance_dict['Pairwise differences found'].append(diff_str)

        else:
            significance_dict['Pairwise differences found'].append('No')

    sig_df = pd.DataFrame.from_dict(significance_dict)
    print(tabulate(sig_df, headers='keys', tablefmt='psql'))




