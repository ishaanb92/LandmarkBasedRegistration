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

def construct_pairs_from_gt(pat_df):

    gt_lesion_matches = []
    for baseline_id, follow_id in zip(list(pat_df['Baseline'].to_numpy()), list(pat_df['Follow-up'].to_numpy())):
        gt_lesion_matches.append((baseline_id, follow_id))

    return gt_lesion_matches

def compute_detection_metrics(predicted_lesion_matches, true_lesion_matches):


    true_matches = 0
    false_positive_matches = 0
    false_negatives = 0

    for pred_pair in predicted_lesion_matches:
        if pred_pair in true_lesion_matches:
            true_matches += 1
        else:
            false_positive_matches += 1

    for true_pair in true_lesion_matches:
        if true_pair in predicted_lesion_matches:
            continue
        else:
            if true_pair[1] != 'None': # If the 2nd ID in the tuple is None => No match found via manual inspection!
                false_negatives += 1

    count_dict = {}
    count_dict['TP'] = true_matches
    count_dict['FP'] = false_positive_matches
    count_dict['FN'] = false_negatives
    return count_dict


# TODO: Store unmatched lesions since these are important diagnostically as well! E.g. : Lesions that went away or new ones


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

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split(os.sep)[-1]

        if pat_id in review_patients or pat_id in failed_registrations or pat_id in missing_lesion_masks:
            continue

        # Load graph
        dgraph = joblib.load(os.path.join(pat_dir, 'corr_graph.pkl'))

        # Check if the constructed graph is bipartite!
        assert(bipartite.is_bipartite(dgraph))

        predicted_lesion_matches = construct_pairs_from_graph(dgraph)

        # Compare predicted matches to GT
        pat_df = gt_matches_df[gt_matches_df['Patient ID'] == int(pat_id)]

        if pat_df.empty is True:
            print('Patient {} not found in the ground truth records'.format(pat_id))
            continue

        true_lesion_matches = construct_pairs_from_gt(pat_df)
        count_dict = compute_detection_metrics(predicted_lesion_matches=predicted_lesion_matches,
                                               true_lesion_matches=true_lesion_matches)

        print('Patient {} :: True predicted matches = {}, False predicted matches = {}, Missed matches = {}'.format(pat_id,
                                                                                                                    count_dict['TP'],
                                                                                                                    count_dict['FP'],
                                                                                                                    count_dict['FN']))
        true_positives += count_dict['TP']
        false_positives += count_dict['FP']
        false_negatives += count_dict['FN']


    print('True matches found = {}'.format(true_positives))
    print('False matches predicted = {}'.format(false_positives))
    print('Matches missed = {}'.format(false_negatives))


