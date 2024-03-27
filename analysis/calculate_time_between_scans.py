"""

Script to calculate time between scans

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
import joblib
from argparse import ArgumentParser
from lesionmatching.util_scripts.utils import create_datetime_object_from_str
import pandas as pd

if __name__ == '__main__':

    pat_data_dirs = joblib.load('test_patients_umc.pkl')
    pat_data_dirs.extend(joblib.load('train_patients_umc.pkl'))
    pat_data_dirs.extend(joblib.load('val_patients_umc.pkl'))

    n_days_arr = np.zeros((len(pat_data_dirs),),
                          dtype=np.float32)

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

    print(np.median(n_days_arr))
    print(np.percentile(n_days_arr, q=25))
    print(np.percentile(n_days_arr, q=75))

