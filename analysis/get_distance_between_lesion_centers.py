"""

Quick and dirty script to get distance between lesion centers (after resampling the moving lesion mask)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
import joblib
from argparse import ArgumentParser
import json
from lesionmatching.analysis.lesion_correspondence import *
import pandas as pd

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)

    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    tp_distances = []
    fp_distances = []
    fn_distances = []
    pids = []
    lesion_ids = []
    fp_pids = []
    fp_lesion_ids = []

    for pat_dir in pat_dirs:

        pid = pat_dir.split(os.sep)[-1]
        # Load graph
        try:
            dgraph = joblib.load(os.path.join(pat_dir, 'corr_graph.pkl'))
        except:
            raise FileNotFoundError

        # Load GT matches
        with open(os.path.join(args.gt_dir,
                               pid,
                               'lesion_links.json')) as f:
            gt_matches = json.load(f)

        gt_dict = preprocess_gt_dict(gt_matches)

        gt_forward_dict = gt_dict['forward_dict']

        # Loop over edges and report distanes
        for u, v in dgraph.edges():
            # Check each edge just once!
            if 'fixed' in u.get_name().lower():
                fixed_lesion_id = u.get_idx()
                moving_lesion_id = v.get_idx()
                fixed_lesion_name = u.get_name()
                moving_lesion_name = v.get_name()

                prediction = dgraph.get_edge_data(u, v)['weight']
                lesion_center_distance = dgraph.get_edge_data(u, v)['distance']

                if prediction == 1:
                    # Check GT
                    if moving_lesion_name in gt_forward_dict[fixed_lesion_name]: # True positive
                        tp_distances.append(lesion_center_distance)
                    else:
                        fp_distances.append(lesion_center_distance)
                        fp_pids.append(pid)
                        fp_lesion_ids.append((fixed_lesion_id, moving_lesion_id))
                else: # Match not predicted
                    if len(gt_forward_dict[fixed_lesion_name]) > 0:
                        if moving_lesion_name in gt_forward_dict[fixed_lesion_name]: # False negative
                            fn_distances.append(lesion_center_distance)
                            # Maintain record of FNs
                            pids.append(pid)
                            lesion_ids.append((fixed_lesion_id, moving_lesion_id))


    tp_distances = np.array(tp_distances).astype(np.float32)
    fp_distances = np.array(fp_distances).astype(np.float32)
    fn_distances = np.array(fn_distances).astype(np.float32)

    # Plot histogram
    fig, ax = plt.subplots()
    bins = np.linspace(0, 50, 100)
    ax.hist(tp_distances, bins=bins, label='True Positives', alpha=0.8, edgecolor='black')
    ax.hist(fp_distances, bins=bins, label='False Positives', alpha=0.8, edgecolor='black')
    ax.hist(fn_distances, bins=bins, label='False Negatives', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Distance between lesion centers (mm)')
    ax.set_ylabel('Count')
    ax.vlines(x=[10],
              ymin=0,
              ymax=1,
              colors='r',
              linestyles='dashed',
              transform=ax.get_xaxis_transform())

    ax.legend(loc='upper right')
    ax.set_ylim((0, 10))
    fig.savefig(os.path.join(args.reg_dir,
                             'lesion_centers.pdf'),
                bbox_inches='tight')

    fig.savefig(os.path.join(args.reg_dir,
                             'lesion_centers.jpg'),
                bbox_inches='tight')

    # Check outlier FN
    max_id = np.argmax(fn_distances)
    print('Max distance  = {}'.format(np.max(fn_distances)))
    print('Patient {}, lesion ids {}'.format(pids[max_id],
                                             lesion_ids[max_id]))

    # FN Analysis
    for idx in range(fn_distances.shape[0]):
        print('Patient {} Fixed Lesion ID {} Moving Lesion ID {} Distance = {}'.format(pids[idx],
                                                                                       lesion_ids[idx][0],
                                                                                       lesion_ids[idx][1],
                                                                                       fn_distances[idx]))

   # FP Analysis
    for idx in range(fp_distances.shape[0]):
        print('Patient {} Fixed Lesion ID {} Moving Lesion ID {} Distance = {}'.format(fp_pids[idx],
                                                                                       fp_lesion_ids[idx][0],
                                                                                       fp_lesion_ids[idx][1],
                                                                                       fp_distances[idx]))



