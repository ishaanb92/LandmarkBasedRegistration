"""

Script to compute lesion correspondence metrics

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from lesionmatching.analysis.lesion_correspondence import *
import numpy as np
import SimpleITK as sitk
from argparse import ArgumentParser
import glob
import shutil
import joblib
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import json

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, help='Directory containing registration results')
    parser.add_argument('--gt_dir', type=str, help='Path to file containing ground truth matches')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()


    pat_dirs  = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    print('Number of patients = {}'.format(len(pat_dirs)))

    review_patients = joblib.load(os.path.join(args.reg_dir, 'patients_to_review.pkl'))
    print('{} patients need to be reviewed'.format(len(review_patients)))

    failed_registrations = joblib.load(os.path.join(args.reg_dir, 'failed_registrations.pkl'))
    print('Registration failed for {} patients :: {}'.format(len(failed_registrations), failed_registrations))

    missing_lesion_masks = joblib.load(os.path.join(args.reg_dir, 'missing_lesion_masks.pkl'))
    print('Patients with (at least one) lesion mask(s) missing = {}'.format(len(missing_lesion_masks)))


    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_matches = 0
    true_negatives = 0
    unmatched_lesions_fixed = 0
    unmatched_lesions_moving = 0

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split(os.sep)[-1]

        if pat_id in missing_lesion_masks:
            continue

        # Load graph
        try:
            dgraph = joblib.load(os.path.join(pat_dir, 'corr_graph.pkl'))
        except:
            continue

        # Check if the constructed graph is bipartite!
        assert(bipartite.is_bipartite(dgraph))

        predicted_lesion_matches = construct_pairs_from_graph(dgraph,
                                                              min_overlap=0.0)

        gt_lesion_corr = os.path.join(args.gt_dir, pat_id, 'lesion_links.json')

        if os.path.exists(gt_lesion_corr) is False:
            print('GT correspondences absent for Patient {}'.format(pat_id))
            continue

        with open(gt_lesion_corr) as f:
            gt_dict = json.load(f)

        true_lesion_matches = construct_pairs_from_gt(gt_dict)

        count_dict = compute_detection_metrics(predicted_lesion_matches=predicted_lesion_matches,
                                               true_lesion_matches=true_lesion_matches)

        unmatched_lesions_moving += find_unmatched_lesions_in_moving_images(dgraph,
                                                                            min_overlap=0.0)

        if args.verbose is True:
            print('Patient {} :: GT matches = {} Unmatched lesion (GT) = {} True positive = {} False positives = {} '\
                   'False negatives = {}  True negatives = {}'.format(pat_id,
                                                                     count_dict['TM'],
                                                                     count_dict['UM'],
                                                                     count_dict['TP'],
                                                                     count_dict['FP'],
                                                                     count_dict['FN'],
                                                                     count_dict['TN']))
        true_matches += count_dict['TM']
        unmatched_lesions_fixed += count_dict['UM']
        true_positives += count_dict['TP']
        false_positives += count_dict['FP']
        false_negatives += count_dict['FN']
        true_negatives += count_dict['TN']

    print('True matches found via visual inspection = {}'.format(true_matches))
    print('Unmatched lesions in the fixed lesion mask = {}'.format(unmatched_lesions_fixed))
    print('Unmatched lesions in the moving lesion mask = {}'.format(unmatched_lesions_moving))
    print('True positive matches = {}'.format(true_positives))
    print('False matches predicted = {}'.format(false_positives))
    print('Matches missed = {}'.format(false_negatives))

    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)
    print('Sensitivity = {}, Specificity = {}'.format(sensitivity,
                                                      specificity))
