"""

Script to compute lesion correspondence metrics

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from segmentation_metrics.lesion_correspondence import *
import numpy as np
import SimpleITK as sitk
from argparse import ArgumentParser
import glob
import shutil
import joblib
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd

def construct_pairs_from_graph(dgraph, min_overlap=0.5):
    """
    Function to construct a list of matched lesions
    Example ouput : [(fixed_lesion_id_0, moving_lesion_id_0), ..., (fixed_lesion_id_n, moving_lesion_id_n)]

    """
    moving_lesion_nodes, fixed_lesion_nodes = bipartite.sets(dgraph)

    assert('moving' in list(moving_lesion_nodes)[0].get_name().lower())
    assert('fixed' in list(fixed_lesion_nodes)[0].get_name().lower())

    predicted_lesion_matches = []
    for f_node in fixed_lesion_nodes:
        for m_node in moving_lesion_nodes:
            edge_weight = dgraph[f_node][m_node]['weight']
            if edge_weight > min_overlap:
                predicted_lesion_matches.append((f_node.get_idx(),
                                                 m_node.get_idx()))

    return predicted_lesion_matches


def find_unmatched_lesions_in_moving_images(dgraph, min_overlap=0.5):

    unmatched_lesions = 0

    moving_lesion_nodes, fixed_lesion_nodes = bipartite.sets(dgraph)

    assert('moving' in list(moving_lesion_nodes)[0].get_name().lower())
    assert('fixed' in list(fixed_lesion_nodes)[0].get_name().lower())

    for m_node in moving_lesion_nodes:
        match_counter = 0
        for f_node in fixed_lesion_nodes:
            edge_weight = dgraph[m_node][f_node]['weight']
            if edge_weight > min_overlap:
                match_counter += 1
        # Check match counter, if it hasn't incremented => No match found!
        if match_counter == 0:
            unmatched_lesions += 1


    return unmatched_lesions

def construct_pairs_from_gt(pat_df):
    """
    Construct pair of tuples from the dataframe.
    The output is a list of tuples of the form:  [(fixed_lesion_id_0, moving_lesion_id_0), ..., (fixed_lesion_id_n, moving_lesion_id_n)]

    """
    gt_lesion_matches = []
    for baseline_id, follow_id in zip(list(pat_df['Baseline'].to_numpy()), list(pat_df['Follow-up'].to_numpy())):
        gt_lesion_matches.append((baseline_id, follow_id))

    return gt_lesion_matches


def compute_detection_metrics(predicted_lesion_matches, true_lesion_matches):

    # Compute from ground truth
    true_matches = 0
    unmatched_lesions = 0


    # Compare prediciton to ground truth
    true_positive_matches = 0
    false_positive_matches = 0
    false_negatives = 0
    true_negatives = 0


    # How many matches were found via visual inspection?
    for true_pair in true_lesion_matches:
        if true_pair[1] != 'None':
            true_matches += 1
        else:
            unmatched_lesions += 1


    # How many matches were found via image registration + resampling?
    for pred_pair in predicted_lesion_matches:
        if pred_pair in true_lesion_matches:
            true_positive_matches += 1
        else:
            false_positive_matches += 1

    # How many matches were missed after registration + resampling?
    for true_pair in true_lesion_matches:
        if true_pair in predicted_lesion_matches:
            continue
        else:
            if true_pair[1] != 'None':
                false_negatives += 1

    # How many lesions (in the fixed image) that did not have a match via visual inspection
    # also do not have a non-zero edge weight in the corr. graph?
    unmatched_fixed_lesions = [p[0] for p in true_lesion_matches if p[1] == 'None']
    fixed_lesions_in_pred = [p[0] for p in predicted_lesion_matches]

    for u_lesion in unmatched_fixed_lesions:
        if u_lesion in fixed_lesions_in_pred:
            continue # Already accounted by false positives!
        else:
            true_negatives += 1


    count_dict = {}
    count_dict['UM'] = unmatched_lesions
    count_dict['TM'] = true_matches
    count_dict['TP'] = true_positive_matches
    count_dict['FP'] = false_positive_matches
    count_dict['FN'] = false_negatives
    count_dict['TN'] = true_negatives
    return count_dict


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Directory containing registration results')
    parser.add_argument('--gt', type=str, help='Path to file containing ground truth matches')


    args = parser.parse_args()


    pat_dirs  = [f.path for f in os.scandir(args.out_dir) if f.is_dir()]

    print('Number of patients = {}'.format(len(pat_dirs)))

    review_patients = joblib.load(os.path.join(args.out_dir, 'patients_to_review.pkl'))
    print('{} patients need to be reviewed'.format(len(review_patients)))

    failed_registrations = joblib.load(os.path.join(args.out_dir, 'failed_registrations.pkl'))
    print('Registration failed for {} patients'.format(len(failed_registrations)))

    missing_lesion_masks = joblib.load(os.path.join(args.out_dir, 'missing_lesion_masks.pkl'))
    print('Patients with (at least one) lesion mask(s) missing = {}'.format(len(missing_lesion_masks)))

    gt_matches_df = pd.read_excel(args.gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_matches = 0
    true_negatives = 0
    unmatched_lesions_fixed = 0
    unmatched_lesions_moving = 0

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split(os.sep)[-1]

        if pat_id in review_patients or pat_id in failed_registrations or pat_id in missing_lesion_masks:
            continue

        # Load graph
        dgraph = joblib.load(os.path.join(pat_dir, 'corr_graph.pkl'))

        # Check if the constructed graph is bipartite!
        assert(bipartite.is_bipartite(dgraph))

        predicted_lesion_matches = construct_pairs_from_graph(dgraph,
                                                              min_overlap=0.0)

        # Compare predicted matches to GT
        pat_df = gt_matches_df[gt_matches_df['Patient ID'] == int(pat_id)]

        if pat_df.empty is True:
            print('Patient {} not found in the ground truth records'.format(pat_id))
            continue

        true_lesion_matches = construct_pairs_from_gt(pat_df)

        count_dict = compute_detection_metrics(predicted_lesion_matches=predicted_lesion_matches,
                                               true_lesion_matches=true_lesion_matches)

        unmatched_lesions_moving += find_unmatched_lesions_in_moving_images(dgraph,
                                                                            min_overlap=0.0)

        print('Patient {} :: True matches = {} Unmatched lesions = {} True positive = {} False positives = {} \
               False negatives = {}  True negatives = {}'.format(pat_id,
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
