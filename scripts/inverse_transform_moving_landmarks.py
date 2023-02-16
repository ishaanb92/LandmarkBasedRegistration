"""

Script to transform moving landmarks to the affine-fixed domain using the inverse of the (estimated) affine transformation

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from lesionmatching.util_scripts.utils import *
import numpy as np
from argparse import ArgumentParser


COPD_DIR = '/home/ishaan/COPDGene'
DIRLAB_DIR = '/home/ishaan/DIR-Lab'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='copd')
    parser.add_argument('--affine_reg_dir', type=str, required=True)


    args = parser.parse_args()

    if args.dataset == 'copd':
        points_dir = os.path.join(COPD_DIR, 'points')
    elif args.dataset == 'dirlab':
        points_dir = os.path.join(DIRLAB_DIR, 'points')

    pat_dirs = [f.path for f in os.scandir(points_dir) if f.is_dir()]

    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]
        patient_affine_dir = os.path.join(args.affine_reg_dir, pid)

        if args.dataset == 'dirlab':
            moving_image_landmarks_path = os.path.join(pdir,
                                                       '{}_4D-75_T50_world_elx.txt'.format(pid))
        elif args.dataset == 'copd':
            moving_image_landmarks_path = os.path.join(pdir,
                                                       '{}_300_eBH_world_r1_elx.txt'.format(pid))

        # Parse landmark text file to get array of shape (N, 3)
        moving_image_landmarks = parse_points_file(fpath=moving_image_landmarks_path)

        # Get affine transform parameters
        A, t, c = get_affine_transform_parameters(fpath=os.path.join(patient_affine_dir,
                                                                     'TransformParameters.0.txt'))

        # Inverse transform moving image landmarks
        transformed_moving_image_landmarks = inverse_affine_transform(points_arr=moving_image_landmarks,
                                                                      A=A,
                                                                      t=t,
                                                                      c=c)

        # Save the landmarks array
        create_landmarks_file(landmarks=transformed_moving_image_landmarks,
                              world=True,
                              fname=os.path.join(patient_affine_dir, 'transformed_moving_landmarks_elx.txt'))





