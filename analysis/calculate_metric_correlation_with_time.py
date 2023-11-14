"""

Script to compute correlation of metrics (sensitivity/specificity) with time between scans

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
import joblib
from argparse import ArgumentParser
from lesionmatching.util_scripts.utils import create_datetime_object_from_str
import pandas as pd
from scipy.stats import spearmanr

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--legends', type=str, nargs='+', default=[])

    args = parser.parse_args()

    pat_data_dirs = joblib.load('test_patients_umc.pkl')

    n_days_arr = np.zeros((len(pat_data_dirs),),
                          dtype=np.float32)

    assert(len(args.result_dirs) == len(args.legends))

    sensitivity_dict = {}
    specificity_dict = {}

    # Read metric DF from file and store in a list
    metric_df_list = []
    for rdir in args.result_dirs:
        # Get metrics DF
        metrics_df = pd.read_pickle(os.path.join(rdir,
                                                 'matching_metrics.pkl'))
        metric_df_list.append(metrics_df)

    for legend in args.legends:
        sensitivity_dict[legend] = []
        specificity_dict[legend] = []

    for idx, pddir in enumerate(pat_data_dirs):

        pid = pddir.split(os.sep)[-1] # Get patient ID

        scan_dirs  = [f.path for f in os.scandir(pddir) if f.is_dir()]

        s_id_0 = scan_dirs[0].split(os.sep)[-1]
        s_id_1 = scan_dirs[1].split(os.sep)[-1]
        timestamp_0 = create_datetime_object_from_str(s_id_0)
        timestamp_1 = create_datetime_object_from_str(s_id_1)
        if timestamp_0 > timestamp_1: # Scan0 occurs after Scan1
            n_days = (timestamp_0 - timestamp_1).days
        elif timestamp_0 < timestamp_1:
            n_days = (timestamp_1 - timestamp_0).days
        else:
            n_days = 0

        n_days_arr[idx] = n_days

        # Loop over results for this patient
        for ridx, rdir in enumerate(args.result_dirs):
            metric_df = metric_df_list[ridx]
            pat_row = metric_df.loc[metric_df['Patient ID'] == pid]


            tp = pat_row['Correct Matches'].values[0]
            fp = pat_row['Incorrect Matches'].values[0]
            fn = pat_row['Missed Matches'].values[0]
            tn = pat_row['True Negatives'].values[0]

            if tp + fn != 0:
                sensitivity = tp/(tp + fn)
            else:
                sensitivity = 1.0
            if tn + fp != 0:
                specificity = tn/(tn + fp)
            else:
                specificity = 1.0

            sensitivity_dict[args.legends[ridx]].append(sensitivity)
            specificity_dict[args.legends[ridx]].append(specificity)


    # Report correlations
    for legend in args.legends:
        sens_res = spearmanr(np.array(sensitivity_dict[legend]),
                             n_days_arr)

        spec_res = spearmanr(np.array(specificity_dict[legend]),
                             n_days_arr)
        print('{} :: Sensitivity : Coeff {} p-value {}'.format(legend,
                                                               sens_res.correlation,
                                                               sens_res.pvalue))

        print('{} :: Sensitivity : Coeff {} p-value {}'.format(legend,
                                                               spec_res.correlation,
                                                               spec_res.pvalue))





