"""

Script to compute landmark corr. metrics -- TPs, FPs, FNs, and spatial error


@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from lesionmatching.analysis.metrics import *
from argparse import ArgumentParser
import os

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='umc')

    args = parser.parse_args()

    if args.dataset == 'umc':
        voxel_spacing = (1.543, 1.543, 1.543)
    elif args.dataset == 'dirlab':
        voxel_spacing = (1.0, 1.0, 1.0)

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
                                                                     voxel_spacing=voxel_spacing)

    create_spatial_error_boxplot(tp_spatial_errors=tp_spatial_errors,
                                 fp_spatial_errors=fp_spatial_errors,
                                 fname=os.path.join(args.result_dir, 'spatial_errors'))


    # Compute spatial matching errors for true and false positives
    tp_spatial_errors, fp_spatial_errors = create_spatial_error_dict(result_dir=args.result_dir,
                                                                     mode='norm',
                                                                     voxel_spacing=voxel_spacing)

    create_spatial_error_boxplot(tp_spatial_errors=tp_spatial_errors,
                                 fp_spatial_errors=fp_spatial_errors,
                                 fname=os.path.join(args.result_dir, 'spatial_errors_norm'))
