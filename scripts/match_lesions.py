"""

Script to match lesions in the baseline and follow-up scans.
Lesion correspondence graphs help to visualize the effects of registration
directly in terms of matching lesions in baseline and follow-up scans

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
from lesionmatching.util_scripts.image_utils import *
import joblib


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, help='Directory containing registration results')

    args = parser.parse_args()

    pat_dirs  = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    print('Number of patients = {}'.format(len(pat_dirs)))

    review_patients = joblib.load(os.path.join(args.reg_dir, 'patients_to_review.pkl'))
    print('{} patients need to be reviewed'.format(len(review_patients)))
    n_review = len(review_patients)

    failed_registrations = joblib.load(os.path.join(args.reg_dir, 'failed_registrations.pkl'))
    print('Registration failed for {} patients'.format(len(failed_registrations)))
    missing_lesion_masks = joblib.load(os.path.join(args.reg_dir, 'missing_lesion_masks.pkl'))
    print('Patients with (at least one) lesion mask(s) missing = {}'.format(len(missing_lesion_masks)))

    # How many patients are we missing?
    missing_patients = failed_registrations
    missing_patients.extend(review_patients)
    missing_patients.extend(missing_lesion_masks)

    n_missing_patients = len(list(set(missing_patients)))
    print('Lesion correspondences cannot be established for {} patients'.format(n_missing_patients))

    for pat_dir in pat_dirs:
        pat_id = pat_dir.split(os.sep)[-1]

        if pat_id in review_patients or pat_id in missing_lesion_masks:
            continue

        m_lesion_dirs = [d for d in glob.glob(os.path.join(pat_dir, 'moving_lesion_*')) if os.path.isdir(d)]
        f_lesion_dirs = [d for d in glob.glob(os.path.join(pat_dir, 'fixed_lesion_*')) if os.path.isdir(d)]

        # Order the directories so their position in the list matches the 'ID' which was determined by the ordering
        # of slices returned by the function return_lesion_coordinates()
        m_lesion_dirs_ordered = [os.path.join(pat_dir, 'moving_lesion_{}'.format(idx)) for idx in range(len(m_lesion_dirs))]
        f_lesion_dirs_ordered = [os.path.join(pat_dir, 'fixed_lesion_{}'.format(idx)) for idx in range(len(f_lesion_dirs))]

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
        # To maintain consistency of 'Lesion ID' (which might get shuffled)
        # if find_objects() is called AFTER resampling, we manually construct
        # a list of Lesion objects to create a graph
        # Create a list of lesions in the moving image

        fixed_lesion_slices = get_lesion_slices(dir_list=f_lesion_dirs_ordered,
                                                fixed=True)

        moving_lesion_slices = get_lesion_slices(dir_list=m_lesion_dirs_ordered,
                                                 fixed=False)

        if fixed_lesion_slices is None or moving_lesion_slices is None:
            # Some problem, add patient ID to the review list
            review_patients.append(pat_id)
            handle_lesion_separation_error(pat_dir=pat_dir)
            continue

        fixed_lesions = []
        for idx, f_lesion_slice in enumerate(fixed_lesion_slices):
            fixed_lesions.append(Lesion(coordinates=f_lesion_slice,
                                        idx=idx,
                                        prefix='Fixed'))

        moving_lesions = []
        for idx, m_lesion_slice in enumerate(moving_lesion_slices):
            moving_lesions.append(Lesion(coordinates=m_lesion_slice,
                                        idx=idx,
                                        prefix='Moving'))

        dgraph = create_correspondence_graph_from_list(pred_lesions=moving_lesions,
                                                       gt_lesions=fixed_lesions,
                                                       seg=moving_lesion_mask_resampled,
                                                       gt=fixed_lesion_mask_np,
                                                       min_overlap=0.5,
                                                       verbose=False)

        # Save the graph object
        # NetworkX documentation states that a "graph" is just a dictionaries all the way down
        # so joblib should work fine to save it
        joblib.dump(value=dgraph,
                   filename=os.path.join(pat_dir, 'corr_graph.pkl'))

        # Visualize correspondence graph
        fname = os.path.join(pat_dir, 'lesion_correspondence.pdf')
        visualize_lesion_correspondences(dgraph=dgraph,
                                         fname=fname,
                                         min_overlap=0.5)

    # Check if more patients have been added to the review list
    if n_review < len(review_patients):
        # Overwrite old list with the updated one
        joblib.dump(value=review_patients,
                    filename=os.path.join(args.reg_dir, 'patients_to_review.pkl'))


