"""

Script to plot metrics related to landmark correspondences
in image pairs related via a synthetic (non-rigid) deformation

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
import numpy as np
from metrics import *
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_correspondence_metric_dict(result_dir=None,
                                   mode='both'):

    # Compute total true positives, false positives, and false negatives
    pat_dirs  = [f.path for f in os.scandir(args.result_dir) if f.is_dir()]

    metric_dict = {}
    metric_dict['Patient'] = []
    metric_dict['True Positives'] = []
    metric_dict['False Positives'] = []
    metric_dict['False Negatives'] = []

    for p_dir in pat_dirs:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            gt_matches = np.load(os.path.join(s_dir, 'gt_matches.npy'))
            if mode == 'both':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches.npy'))
            elif mode == 'norm':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_norm.npy'))
            elif mode == 'prob':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_pred.npy'))

            per_image_metric_dict = get_match_statistics(gt=gt_matches,
                                                         pred=pred_matches)

            for metric in per_image_metric_dict.keys():
                metric_dict[metric].append(per_image_metric_dict[metric])

            metric_dict['Patient'].append('{}-{}'.format(s_dir.split(os.sep)[-1],
                                                         s_dir.split(os.sep)[-2]))

    return metric_dict


def plot_bar_graph(metric_dict, fname):

    assert(isinstance(metric_dict, dict))

    metric_df = pd.DataFrame.from_dict(metric_dict)

    metric_df_reshape = pd.melt(metric_df,
                                id_vars='Patient',
                                var_name='Detection',
                                value_name='Count')



    fig, ax = plt.subplots()

    sns.barplot(data=metric_df_reshape,
                x='Patient',
                y='Count',
                hue='Detection',
                ax=ax)

    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim((0, 1024))

    fig.savefig(fname+'.png',
                bbox_inches='tight')
    fig.savefig(fname+'.pdf',
                bbox_inches='tight')

    plt.close()



def match_landmarks(landmarks,
                    projected_landmarks,
                    matches):

    indices_orginal, indices_projected = np.nonzero(matches)

    # Shape: [K', 3] (K' <= K)
    matched_landmarks_original = landmarks[indices_orginal, :]
    matched_landmarks_projected = projected_landmarks[indices_projected, :]

    landmark_pairs = np.concatenate([np.expand_dims(matched_landmarks_original, axis=0),
                                     np.expand_dims(matched_landmarks_projected, axis=0)],
                                    axis=0)

    return landmark_pairs


def compute_spatial_error(landmark_pairs,
                          error_type='euclidean',
                          voxel_spacing=(1.543, 1.543, 1.543)):

    # Shape: [K', 3] (K' <= K)
    landmarks_original = landmark_pairs[0, ...]
    landmarks_projected = landmark_pairs[1, ...]

    # See broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    voxel_spacing = np.expand_dims(np.array(voxel_spacing),
                                   axis=0)

    if error_type == 'euclidean':
        spatial_error = np.sqrt(np.sum(np.power(np.subtract(landmarks_original,
                                                            landmarks_projected)*voxel_spacing, 2),
                                       axis=1))
    else:
        if error_type == 'x':
            dim = 2
        elif error_type == 'y':
            dim = 1
        elif error_type == 'z':
            dim = 0
        else:
            raise RuntimeError('Unknown spatial error type {}'.format(error_type))

        spatial_error = np.abs(np.subtract(landmarks_projected[:, dim],
                                           landmarks_original[:, dim])*voxel_spacing[:, dim])

    return spatial_error



def create_spatial_error_dict(result_dir=None,
                              mode='both',
                              voxel_spacing=(1.543, 1.543, 1.543)):

    # Compute total true positives, false positives, and false negatives
    pat_dirs  = [f.path for f in os.scandir(args.result_dir) if f.is_dir()]

    tp_spatial_errors = {}
    tp_spatial_errors['Euclidean'] = []
    tp_spatial_errors['X-error'] = []
    tp_spatial_errors['Y-error'] = []
    tp_spatial_errors['Z-error'] = []

    fp_spatial_errors = {}
    fp_spatial_errors['Euclidean'] = []
    fp_spatial_errors['X-error'] = []
    fp_spatial_errors['Y-error'] = []
    fp_spatial_errors['Z-error'] = []

    for p_dir in pat_dirs:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            # Get landmarks, shape : [K, 3]
            landmarks = np.load(os.path.join(s_dir, 'landmarks_original.npy'))
            projected_landmarks = np.load(os.path.join(s_dir, 'landmarks_projected.npy'))

            # Get GT matches, shape: [K, K]
            gt_matches = np.load(os.path.join(s_dir, 'gt_matches.npy'))

            # Get pred matches
            if mode == 'both':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches.npy'))
            elif mode == 'norm':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_norm.npy'))
            elif mode == 'prob':
                pred_matches = np.load(os.path.join(s_dir, 'pred_matches_prob.npy'))
            else:
                raise RuntimeError('Invalid mode {}'.format(mode))

            # Element-wise multiplication to get TP, FP match matrices
            tp_matches = gt_matches*pred_matches
            fp_matches = (1-gt_matches)*pred_matches

            # Based on matches and landmark, get TP and FP landmark pairs
            tp_landmark_pairs = match_landmarks(landmarks=landmarks,
                                                projected_landmarks=projected_landmarks,
                                                matches=tp_matches)

            fp_landmark_pairs = match_landmarks(landmarks=landmarks,
                                                projected_landmarks=projected_landmarks,
                                                matches=fp_matches)

            # True positive spatial matching errors
            tp_spatial_error_euc = compute_spatial_error(landmark_pairs=tp_landmark_pairs,
                                                         error_type='euclidean',
                                                         voxel_spacing=voxel_spacing)

            tp_spatial_error_x = compute_spatial_error(landmark_pairs=tp_landmark_pairs,
                                                       error_type='x',
                                                       voxel_spacing=voxel_spacing)

            tp_spatial_error_y = compute_spatial_error(landmark_pairs=tp_landmark_pairs,
                                                       error_type='y',
                                                       voxel_spacing=voxel_spacing)

            tp_spatial_error_z = compute_spatial_error(landmark_pairs=tp_landmark_pairs,
                                                       error_type='z',
                                                       voxel_spacing=voxel_spacing)

            tp_spatial_errors['Euclidean'].extend(list(tp_spatial_error_euc))
            tp_spatial_errors['X-error'].extend(list(tp_spatial_error_x))
            tp_spatial_errors['Y-error'].extend(list(tp_spatial_error_y))
            tp_spatial_errors['Z-error'].extend(list(tp_spatial_error_z))

            # False positive spatial matching errors
            fp_spatial_error_euc = compute_spatial_error(landmark_pairs=fp_landmark_pairs,
                                                         error_type='euclidean',
                                                         voxel_spacing=voxel_spacing)

            fp_spatial_error_x = compute_spatial_error(landmark_pairs=fp_landmark_pairs,
                                                       error_type='x',
                                                       voxel_spacing=voxel_spacing)

            fp_spatial_error_y = compute_spatial_error(landmark_pairs=fp_landmark_pairs,
                                                       error_type='y',
                                                       voxel_spacing=voxel_spacing)

            fp_spatial_error_z = compute_spatial_error(landmark_pairs=fp_landmark_pairs,
                                                       error_type='z',
                                                       voxel_spacing=voxel_spacing)

            fp_spatial_errors['Euclidean'].extend(list(fp_spatial_error_euc))
            fp_spatial_errors['X-error'].extend(list(fp_spatial_error_x))
            fp_spatial_errors['Y-error'].extend(list(fp_spatial_error_y))
            fp_spatial_errors['Z-error'].extend(list(fp_spatial_error_z))

    return tp_spatial_errors, fp_spatial_errors


def create_spatial_error_boxplot(tp_spatial_errors,
                                 fp_spatial_errors,
                                 fname=None):

    assert(isinstance(tp_spatial_errors, dict))
    assert(isinstance(fp_spatial_errors, dict))

    tp_df = pd.DataFrame.from_dict(tp_spatial_errors)
    fp_df = pd.DataFrame.from_dict(fp_spatial_errors)

    max_error = np.amax(fp_df.to_numpy(),
                        axis=None)

    fig, ax = plt.subplots(nrows=1,
                           ncols=2,
                           figsize=(10, 5))

    sns.boxplot(data=tp_df,
                ax=ax[0])

    sns.boxplot(data=fp_df,
                ax=ax[1])

    ax[0].set_title('True positive spatial errors')
    ax[1].set_title('False positive spatial errors')

    ax[0].set_ylabel('Error (mm)')
    ax[1].set_ylabel('Error (mm)')

    ax[0].tick_params(axis="x", rotation=45)
    ax[1].tick_params(axis="x", rotation=45)

    ax[0].set_ylim((0, max_error))
    ax[1].set_ylim((0, max_error))


    fig.savefig(fname+'.png',
                bbox_inches='tight')

    fig.savefig(fname+'.pdf',
                bbox_inches='tight')



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)

    args = parser.parse_args()

    # Get TP, FP, and FN counts
    metric_dict = get_correspondence_metric_dict(result_dir=args.result_dir,
                                                 mode='both')

    plot_bar_graph(metric_dict,
                   fname=os.path.join(args.result_dir, 'detection_stats_both'))

    metric_dict = get_correspondence_metric_dict(result_dir=args.result_dir,
                                                 mode='norm')

    plot_bar_graph(metric_dict,
                   fname=os.path.join(args.result_dir, 'detection_stats_norm'))

    # Compute spatial matching errors for true and false positives
    tp_spatial_errors, fp_spatial_errors = create_spatial_error_dict(result_dir=args.result_dir,
                                                                     mode='both',
                                                                     voxel_spacing=(1.543, 1.543, 1.543))

    create_spatial_error_boxplot(tp_spatial_errors=tp_spatial_errors,
                                 fp_spatial_errors=fp_spatial_errors,
                                 fname=os.path.join(args.result_dir, 'spatial_errors'))


    # Compute spatial matching errors for true and false positives
    tp_spatial_errors, fp_spatial_errors = create_spatial_error_dict(result_dir=args.result_dir,
                                                                     mode='norm',
                                                                     voxel_spacing=(1.543, 1.543, 1.543))

    create_spatial_error_boxplot(tp_spatial_errors=tp_spatial_errors,
                                 fp_spatial_errors=fp_spatial_errors,
                                 fname=os.path.join(args.result_dir, 'spatial_errors_norm'))
