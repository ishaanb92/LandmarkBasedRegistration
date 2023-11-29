"""
Script to plot lesion correspondence metrics to compare different registration/landmarks configurations

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--legends', type=str, nargs='+', default=[])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--title', type=str, default=None, nargs='+')
    parser.add_argument('--output_file', type=str, default=None)


    args = parser.parse_args()

    assert(len(args.result_dirs) == len(args.legends))

    plot_dict = {}
    plot_dict['Configuration'] = []
    plot_dict['Metric'] = []
    plot_dict['Metric Type'] = []

    for idx, rdir in enumerate(args.result_dirs):
        legend = args.legends[idx]
        # Read the pd data frame with statistics
        metric_df = pd.read_pickle(os.path.join(rdir,
                                                'matching_metrics.pkl'))

        # Compute sensitivity, specificity
        true_positives = metric_df['Correct Matches'].sum()
        false_positives = metric_df['Incorrect Matches'].sum()
        false_negatives = metric_df['Missed Matches'].sum()
        true_negatives = metric_df['True Negatives'].sum()

        sensitivity = true_positives/(true_positives + false_negatives)
        specificity = true_negatives/(true_negatives + false_positives)
        accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)

        # Create a row for accuracy
        plot_dict['Configuration'].append(legend)
        plot_dict['Metric'].append(accuracy)
        plot_dict['Metric Type'].append('Accuracy')

        # Create a row for sensitivity
        plot_dict['Configuration'].append(legend)
        plot_dict['Metric'].append(sensitivity)
        plot_dict['Metric Type'].append('Sensitivity')

        # Create a row for specificity
        plot_dict['Configuration'].append(legend)
        plot_dict['Metric'].append(specificity)
        plot_dict['Metric Type'].append('Specificity')


    plot_df = pd.DataFrame.from_dict(plot_dict)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    print('Size in inches = {}'.format(fig.get_size_inches()))

    sns.barplot(data=plot_df,
                x='Configuration',
                y='Metric',
                hue='Metric Type',
                ax=ax)

    if args.title is not None:
        ax.set_title(' '.join(args.title))

    ax.set_ylim((0, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='lower right')
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)


    fig.savefig(os.path.join(args.save_dir,
                             '{}.pdf'.format(args.output_file)),
                bbox_inches='tight')

    fig.savefig(os.path.join(args.save_dir,
                             '{}.jpg'.format(args.output_file)),
                bbox_inches='tight')

