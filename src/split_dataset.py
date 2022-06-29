"""

Script to split dataset into train, validation, and test based on random seed(s)
Output is a dictionary containing directory paths for each of the 3 splits

@author:Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import joblib
import random
from argparse import ArgumentParser
from math import floor

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_val_split_seed', type=int, default=1234)
    parser.add_argument('--test_split_seed', type=int, default=5678)

    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    n_patients = len(pat_dirs)
    n_test = floor(0.2*n_patients + 0.5)

    n_train_val = n_patients-n_test
    n_val = 4
    n_train = n_train_val-n_val

    random.Random(args.test_split_seed).shuffle(pat_dirs)

    test_patients = pat_dirs[:n_test]
    train_val_patients = pat_dirs[n_test:]

    # Another shuffle in case we want to do multiple trainings keeping the
    # same test set but different train-val splits (eg: ensembles)
    random.Random(args.train_val_split_seed).shuffle(train_val_patients)
    train_patients = train_val_patients[:n_train]
    val_patients = train_val_patients[n_train:]

    joblib.dump(train_patients, 'train_patients.pkl')
    joblib.dump(val_patients, 'val_patients.pkl')
    joblib.dump(test_patients, 'test_patients.pkl')


