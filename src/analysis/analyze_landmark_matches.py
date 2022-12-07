"""

Script to plot metrics related to landmark correspondences
in image pairs related via a synthetic (non-rigid) deformation

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
import numpy as np
from metrics import *
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_correspondence_metric_dict(result_dir=None,
                                   mode='both'):

    # Compute total true positives, false positives, and false negatives
    pat_dirs  = [f.path for f in os.scandir(args.result_dir) if f.is_dir()]

    metric_dict = {}
    metric_dict['Patient'] = []
    metric_dict['True Positives'] = []
    metric_dict['False Positives'] = []
    metric_dict['False Negatives'] = []

    index = []

    for p_dir in pat_dirs:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            gt_matches = np.load(os.path.join(s_dir, 'gt_matches.npy'))
            if mode == 'both':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches.npy'))
            elif mode == 'norm':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_norm.npy'))
            elif mode == 'prob':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_pred.npy'))

            per_image_metric_dict = get_match_statistics(gt=gt_matches,
                                                         pred=pred_matches)

            for metric in per_image_metric_dict.keys():
                metric_dict[metric].append(per_image_metric_dict[metric])

            metric_dict['Patient'].append('{}-{}'.format(s_dir.split(os.sep)[-1],
                                                         s_dir.split(os.sep)[-2]))

    return metric_dict


def plot_bar_graph(metric_dict, fname):

    assert(isinstance(metric_dict, dict))

    metric_df = pd.DataFrame.from_dict(metric_dict)

    metric_df_reshape = pd.melt(metric_df,
                                id_vars='Patient',
                                var_name='Detection',
                                value_name='Count')



    fig, ax = plt.subplots()

    sns.barplot(data=metric_df_reshape,
                x='Patient',
                y='Count',
                hue='Detection',
                ax=ax)

    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim((0, 1024))

    fig.savefig(fname+'.png',
                bbox_inches='tight')
    fig.savefig(fname+'.pdf',
                bbox_inches='tight')

    plt.close()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)

    args = parser.parse_args()

    metric_dict = get_correspondence_metric_dict(result_dir=args.result_dir,
                                                 mode='both')

    plot_bar_graph(metric_dict,
                   fname=os.path.join(args.result_dir, 'detection_stats_both'))

    metric_dict = get_correspondence_metric_dict(result_dir=args.result_dir,
                                                 mode='norm')

    plot_bar_graph(metric_dict,
                   fname=os.path.join(args.result_dir, 'detection_stats_norm'))




