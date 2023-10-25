"""

Script to plot localization error for landmark models

"""

import os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, default=None)
    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for idx, pdir in enumerate(pdirs):

        if idx == 0:
            loc_errors = np.load(os.path.join(pdir,
                                              'euclidean_error_pred_gt_proj.npy'))
        else:
            loc_errors = np.concatenate([loc_errors,
                                        np.load(os.path.join(pdir,
                                                             'euclidean_error_pred_gt_proj.npy'))],
                                        axis=0)


    median = np.median(loc_errors)
    q25 = np.percentile(loc_errors, 25)
    q75 = np.percentile(loc_errors, 75)
    iqr = q75 - q25
    print('Localization error (in mm) = {} +/- {}'.format(np.mean(loc_errors),
                                                          np.std(loc_errors)))

    print('Localization error (in mm) :: Median = {}, q25 = {}, q75 = {} IQR = {}'.format(median,
                                                                                          q25,
                                                                                          q75,
                                                                                          iqr))

    # Plot histogram of localization errors
    fig, ax = plt.subplots()

    sns.histplot(data=loc_errors,
                 ax=ax,
                 stat='count')

    max_error = np.amax(loc_errors)

    ax.set_xlabel('Localization Error (mm)')
    ax.set_ylabel('Count')
    ax.vlines(x=[np.percentile(loc_errors, 25),
                np.percentile(loc_errors, 50),
                np.percentile(loc_errors, 75)],
              ymin=0,
              ymax=1,
             colors='r',
             linestyles='dashed',
             transform=ax.get_xaxis_transform()
             )

    max_error_95p = np.percentile(loc_errors, 95)

    ax.set_xlim((0, 100))

    fig.savefig(os.path.join(args.landmarks_dir,
                             'localization_error_histplot.pdf'),
                bbox_inches='tight')
