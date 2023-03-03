"""

To check the "goodness" of our point correspondences, we fit a thin-plate spline (TPS) through all the point correspondences s.t. the TPS defines the transformation T : F->M. This transfomation can be then be analyzed to reason about performance or even filter a subset of point correspondences

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import SimpleITK as sitk
import numpy as np
from argparse import ArgumentParser
from lesionmatching.util_scripts.utils import *
from lesionmatching.data.deformations import construct_tps_defromation

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='copd')


    args = parser.parse_args()

    if args.dataset == 'copd':
        points_dir = '/home/ishaan/COPDGene/points'
    elif args.dataset == 'dirlab':
        points_dir = '/home/ishaan/DIR-Lab/points'

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]

    for pdir in pdirs:

        pid = pdir.split(os.sep)[-1]

        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        shape = fixed_image_itk.GetSize()

        fixed_points_arr = parse_points_file(os.path.join(pdir,
                                                          'fixed_landmarks_elx.txt'))

        moving_points_arr = parse_points_file(os.path.join(pdir,
                                                           'moving_landmarks_elx.txt'))

        # Scale the coordinates between [0, 1]

        fixed_points_scaled = np.divide(fixed_points_arr,
                                        np.expand_dims(np.array(shape),
                                                       axis=0))

        moving_points_scaled = np.divide(moving_points_arr,
                                         np.expand_dims(np.array(shape),
                                                        axis=0))


        print('Fitting TPS for landmark correspondences for Patient {}'.format(pid))

        # 1. Fit thin-plate spline to define DVF based on point correspondences
        T = construct_tps_defromation(p1=fixed_points_scaled,
                                      p2=moving_points_scaled,
                                      shape=np.array(shape))

        print(T.shape)

        # TODO
        # 2. Construct Jacobian of T

        # 3. Compute determinant of Jacobian


