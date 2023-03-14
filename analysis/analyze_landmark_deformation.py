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
from lesionmatching.util_scripts.image_utils import *
from lesionmatching.data.deformations import construct_tps_defromation, calculate_jacobian_determinant

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
        # Smoothing term is set to 0 for now, we want exact interpolation to study the
        # properties of the deformations defined by the (predicted/GT) landmark correspondences
        T = construct_tps_defromation(p1=fixed_points_scaled,
                                      p2=moving_points_scaled,
                                      shape=np.array(shape))

        # 2. Compute determinant of Jacobian
        jac_det = calculate_jacobian_determinant(deformed_grid=T)

        print('Jacobian determinant :: Min = {}, Max = {}'.format(np.amin(jac_det),
                                                                  np.amax(jac_det)))
        # 3. Save the jac_det as an ITK image
        save_ras_as_itk(img=jac_det,
                        metadata={'spacing':fixed_image_itk.GetSpacing(),
                                  'origin':fixed_image_itk.GetOrigin(),
                                  'direction':fixed_image_itk.GetDirection()},
                        fname=os.path.join(pdir, 'jac_det.mha'))







