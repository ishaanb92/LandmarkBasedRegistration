"""

Script to match lesions in the baseline and follow-up scans.
Lesion correspondence graphs help to visualize the effects of registration
directly in terms of matching lesions in baseline and follow-up scans

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

def merge_lesions_masks(dir_list=None):

    for idx, lesion_dir in enumerate(dir_list):
        lesion_mask_itk = sitk.ReadImage(os.path.join(lesion_dir, 'result.nii'))
        if idx == 0:
            lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk)
        else:
            lesion_mask_np += sitk.GetArrayFromImage(lesion_mask_itk)

    lesion_mask_np = np.where(lesion_mask_np > 1, 1, lesion_mask_np)
    _, n_lesions = return_lesion_coordinates(lesion_mask_np)

    return lesion_mask_np


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str)

    args = parser.parse_args()

    pat_dirs  = [f.path for f in os.scandir(args.out_dir) if f.is_dir()]

    print('Number of patients = {}'.format(len(pat_dirs)))

    for pat_dir in pat_dirs:
        pat_id = pat_dir.split(os.sep)[-1]

        m_lesion_dirs = [d for d in glob.glob(os.path.join(pat_dir, 'moving_lesion_*')) if os.path.isdir(d)]
        if len(m_lesion_dirs) == 0:
            print('No lesions present in moving image for patient {}'.format(pat_id))
            continue

        moving_lesion_mask_resampled = merge_lesions_masks(m_lesion_dirs)

        fixed_lesion_mask_itk = sitk.ReadImage(os.path.join(pat_dir, 'fixed_lesion_mask.nii.gz'))
        fixed_lesion_mask_np = sitk.GetArrayFromImage(fixed_lesion_mask_itk)

        # Use fixed lesion mask metadata to create ITK-compatible resampled moving lesion mask
        moving_lesion_mask_resample_itk = sitk.GetImageFromArray(moving_lesion_mask_resampled)
        moving_lesion_mask_resample_itk.SetSpacing(fixed_lesion_mask_itk.GetSpacing())
        moving_lesion_mask_resample_itk.SetOrigin(fixed_lesion_mask_itk.GetOrigin())
        moving_lesion_mask_resample_itk.SetDirection(fixed_lesion_mask_itk.GetDirection())
        sitk.WriteImage(moving_lesion_mask_resample_itk,
                        os.path.join(pat_dir, 'moving_lesion_mask_resampled.nii.gz'))

        print('Create correspondence graph for Patient {}'.format(pat_id))
        # Create the correspondence graph
        dgraph = create_correspondence_graph(seg=moving_lesion_mask_resampled,
                                             gt=fixed_lesion_mask_np,
                                             seg_prefix='Moving',
                                             gt_prefix='Fixed',
                                             min_overlap=0.5,
                                             verbose=True)


        # Visualize correspondence graph
        fname = os.path.join(pat_dir, 'lesion_correspondence.pdf')
        visualize_lesion_correspondences(dgraph=dgraph,
                                         fname=fname)








