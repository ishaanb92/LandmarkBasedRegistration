"""

Script to check for the existence of lesion masks + GT lesion correspondences for the UMC dataset

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
import copy
import random
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)

    args = parser.parse_args()

    random.seed(1234)

    # Get a list of patient IDs
    pat_ids = [f.name for f in os.scandir(args.data_dir) if f.is_dir()]

    missing_lesion_mask_pats = []
    no_corr_pats = []
    test_patients = []

    for pid in pat_ids:
        # Intialize flags
        has_lesion_mask = False
        has_lesion_corr = False

        # Check for lesion masks
        lesion_masks_found = 0
        pat_dir = os.path.join(args.data_dir, pid)
        scan_dirs = [f.path for f in os.scandir(pat_dir) if f.is_dir()]
        for sdir in scan_dirs:
            if os.path.exists(os.path.join(sdir, '3DLesionAnnotations.nii')) is True:
                lesion_masks_found += 1

        if lesion_masks_found == 2:
            has_lesion_mask = True

        # Check for lesion correspondences
        pat_gt_dir = os.path.join(args.gt_dir, pid)
        if os.path.exists(os.path.join(pat_gt_dir, 'lesion_links.json')) is True:
            has_lesion_corr = True

        # Put this patient in the right "slot"
        if has_lesion_mask is False: # Can only be used to train DL-model
            missing_lesion_mask_pats.append(os.path.join(args.data_dir,
                                                         pid))
        else: # Lesion mask found
            if has_lesion_corr is False: # No correspondences (this list is reducible)
                no_corr_pats.append(os.path.join(args.data_dir,
                                                 pid))
            else: # Has both lesion masks and correspondences
                test_patients.append(os.path.join(args.data_dir,
                                                  pid))

    print('Patients with missing lesion masks: {}'.format(len(missing_lesion_mask_pats)))
    print('Patients with missing correspondences: {}'.format(len(no_corr_pats)))
    print('Patients with lesion masks and correspondences: {}'.format(len(test_patients)))

    # Train + Val : Patients with missing lesion masks can be used to train the DL-model
    train_and_val_patients = missing_lesion_mask_pats
    train_and_val_patients.extend(no_corr_pats)

    random.shuffle(train_and_val_patients)
    train_patients = train_and_val_patients[:25]
    val_patients = train_and_val_patients[25:]

    joblib.dump(train_patients,
                'train_patients_umc.pkl')
    joblib.dump(val_patients,
                'val_patients_umc.pkl')
    joblib.dump(test_patients,
                'test_patients_umc.pkl')
