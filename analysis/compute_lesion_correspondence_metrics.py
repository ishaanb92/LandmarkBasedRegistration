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
from networkx.exception import NetworkXPointlessConcept
import pandas as pd
import json
from tabulate import tabulate

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, help='Directory containing registration results')
    parser.add_argument('--gt_dir', type=str, help='Path to file containing ground truth matches')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()


    pat_dirs  = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    print('Number of patients = {}'.format(len(pat_dirs)))

    review_patients = joblib.load(os.path.join(args.reg_dir, 'patients_to_review.pkl'))
    print('{} patients need to be reviewed :: {}'.format(len(review_patients), review_patients))

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

    metric_dict = {}
    metric_dict['Patient ID'] = []
    metric_dict['Correct Matches'] = []
    metric_dict['Incorrect Matches'] = []
    metric_dict['Missed Matches'] = []
    metric_dict['True Negatives'] = []
    metric_dict['LN_Pred'] = []
    metric_dict['LN_GT'] = []
    metric_dict['LD_Pred'] = []
    metric_dict['LD_GT'] = []

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split(os.sep)[-1]

        print('Computing metrics for Patient {}'.format(pat_id))

        if pat_id in missing_lesion_masks:
            continue

        metric_dict['Patient ID'].append(pat_id)

        # Load graph
        try:
            dgraph = joblib.load(os.path.join(pat_dir, 'corr_graph.pkl'))
        except:
            continue

        # Splitting the bipartite graph creates a problem. The position of the
        # lesion in the list is no longer the lesion idx
        try:
            moving_lesion_nodes, fixed_lesion_nodes = bipartite.sets(dgraph)
        except NetworkXPointlessConcept:
            print('No measurable lesions for patient {}'.format(pat_id))
            continue

        n_fixed_lesions = len(fixed_lesion_nodes)
        n_moving_lesions = len(moving_lesion_nodes)

        # Create an ordered list where position in the list corr. to the lesion ID
        fixed_lesion_nodes_ordered = [None]*n_fixed_lesions
        moving_lesion_nodes_ordered = [None]*n_moving_lesions

        for lesion_idx in range(n_fixed_lesions):
            for list_idx, f_lesion_node in enumerate(list(fixed_lesion_nodes)):
                idx = f_lesion_node.get_idx()
                if idx == lesion_idx:
                    fixed_lesion_nodes_ordered[lesion_idx] = f_lesion_node
                    break

        for lesion_idx in range(n_moving_lesions):
            for list_idx, m_lesion_node in enumerate(list(moving_lesion_nodes)):
                idx = m_lesion_node.get_idx()
                if idx == lesion_idx:
                    moving_lesion_nodes_ordered[lesion_idx] = m_lesion_node
                    break

        # Check if the constructed graph is bipartite!
        assert(bipartite.is_bipartite(dgraph))

        pred_dict = construct_dict_from_graph(dgraph,
                                              min_diameter=0)

        gt_lesion_corr = os.path.join(args.gt_dir, pat_id, 'lesion_links.json')

        if os.path.exists(gt_lesion_corr) is False:
            print('GT correspondences absent for Patient {}'.format(pat_id))
            continue

        with open(gt_lesion_corr) as f:
            gt_dict = json.load(f)

        gt_dict = preprocess_gt_dict(gt_dict)

        if args.verbose is True:
            print('Patient {} pred dict : {}'.format(pat_id, pred_dict))
            print('Patient {} GT dict: {}'.format(pat_id, gt_dict))


        count_dict = compute_detection_metrics(pred_dict=pred_dict,
                                               gt_dict=gt_dict)

        metric_dict['Correct Matches'].append(count_dict['TP'])
        metric_dict['Incorrect Matches'].append(count_dict['FP'])
        metric_dict['Missed Matches'].append(count_dict['FN'])
        metric_dict['True Negatives'].append(count_dict['TN'])
        metric_dict['LN_Pred'].append(count_dict['LN_Pred'])
        metric_dict['LN_GT'].append(count_dict['LN_GT'])
        metric_dict['LD_Pred'].append(count_dict['LD_Pred'])
        metric_dict['LD_GT'].append(count_dict['LD_GT'])


    # Create pandas DF
    metric_df = pd.DataFrame.from_dict(metric_dict)
    print(tabulate(metric_df,
                   headers='keys',
                   tablefmt='psql'))

    metric_df.to_pickle(os.path.join(args.reg_dir,
                                     'matching_metrics.pkl'))

    true_positives = metric_df['Correct Matches'].sum()
    false_positives = metric_df['Incorrect Matches'].sum()
    false_negatives = metric_df['Missed Matches'].sum()
    true_negatives = metric_df['True Negatives'].sum()
    disappeared_lesions_gt = metric_df['LD_GT'].sum()
    disappeared_lesions_pred = metric_df['LD_Pred'].sum()
    new_lesions_gt = metric_df['LN_GT'].sum()
    new_lesions_pred = metric_df['LN_Pred'].sum()

    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)
    accuracy = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
    new_lesions_found = new_lesions_pred/new_lesions_gt
    disappeared_lesions_found = disappeared_lesions_pred/disappeared_lesions_gt
    print('Accuracy = {} Sensitivity = {}, Specificity = {}'.format(accuracy,
                                                                    sensitivity,
                                                                    specificity))

    print('% new lesions found = {} % disappeared lesions found = {}'.format(new_lesions_found,
                                                                             disappeared_lesions_found))

