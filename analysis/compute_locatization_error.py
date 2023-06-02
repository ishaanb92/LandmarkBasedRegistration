"""

Script to plot localization error for landmark models

"""

import os
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, default=None)
    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    # TODO: Plot localization errors for multiple configs together
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
    iqr = np.percentile(loc_errors, 75) - np.percentile(loc_errors, 25)
    print('Localization error (in mm) = {} +/- {}'.format(np.mean(loc_errors),
                                                          np.std(loc_errors)))

    print('Localization error (in mm) :: Median = {}, IQR = {}'.format(median,
                                                                       iqr))
