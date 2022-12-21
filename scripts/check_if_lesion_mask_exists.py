""""

Script to see if a lesion mask exists for a patient/scan

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from argparse import ArgumentParser
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--test', action='store_true', help='Use flag so that patients in the test-set are registered')

    args = parser.parse_args()

    if args.test is False:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients.pkl')))
    else:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'test_patients.pkl'))

    missing_lesion_masks = []

    for pat_dir in pat_dirs:
        p_id = pat_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(pat_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            if os.path.exists(os.path.join(s_dir, '3DLesionAnnotations.nii')) is False:
                missing_lesion_masks.append(p_id)

    # Remove duplicates (if lesion masks for scans are missing)
    missing_lesion_masks = list(set(missing_lesion_masks))

    print('Lesion masks for {} patients absent'.format(len(missing_lesion_masks)))

    if args.test is True:
        joblib.dump(value=missing_lesion_masks,
                    filename='missing_lesions_testset.pkl')
    else:
        joblib.dump(value=missing_lesion_masks,
                    filename='missing_lesions_devset.pkl')

