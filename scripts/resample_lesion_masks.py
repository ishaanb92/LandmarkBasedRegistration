"""

Use the transformation parameters obtain from image/liver mask registration
to resample lesion masks using transformix. Lesion masks of the moving image (follow-up)
need to be resampled on the fixed image domain so that matching can take place.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from elastix.transform_parameter_editor import TransformParameterFileEditor
from elastix.transformix_interface import TransformixInterface
from argparse import ArgumentParser
import joblib
import shutil
from lesionmatching.util_scripts.image_utils import *
from lesionmatching.util_scripts.utils import *
import SimpleITK as sitk
import numpy as np
import shutil

ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'
TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='Folder containing patient images and masks')
    parser.add_argument('--reg_dir', type=str, help='Folder containing registration results', default='registraion_results')
    parser.add_argument('--mode', type=str, default='image')

    args = parser.parse_args()



    add_library_path(ELASTIX_LIB)

    review_patients = []
    failed_registrations = []
    missing_lesion_masks = []

    pat_reg_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    for pat_reg_dir in pat_reg_dirs:

        pat_id = pat_reg_dir.split(os.sep)[-1]

        pat_dir = os.path.join(args.data_dir, pat_id)
        scan_dirs  = [f.name for f in os.scandir(pat_dir) if f.is_dir()]
        dt_obj_0 = create_datetime_object_from_str(scan_dirs[0])
        dt_obj_1 = create_datetime_object_from_str(scan_dirs[1])

        if dt_obj_0 > dt_obj_1: # Scan[0] is the moving image, Scan[1] is the fixed image
            moving_image_idx = 0
            fixed_image_idx = 1
        else: # Scan[1] is the moving image, Scan[0] is the fixed image
            moving_image_idx = 1
            fixed_image_idx = 0

        fixed_lesion_mask_path = os.path.join(pat_dir, scan_dirs[fixed_image_idx], '3DLesionAnnotations.nii')
        moving_lesion_mask_path = os.path.join(pat_dir, scan_dirs[moving_image_idx], '3DLesionAnnotations.nii')

        if os.path.exists(fixed_lesion_mask_path) is False or os.path.exists(moving_lesion_mask_path) is False:
            print('Lesion mask(s) missing for patient {}'.format(pat_id))
            missing_lesion_masks.append(pat_id)
            continue

        # Copy the fixed and moving lesion masks to the registration output directory
        fixed_lesion_mask_itk = sitk.ReadImage(fixed_lesion_mask_path)
        moving_lesion_mask_itk = sitk.ReadImage(moving_lesion_mask_path)

        fixed_lesion_mask_itk = check_and_fix_masks(fixed_lesion_mask_itk)
        moving_lesion_mask_itk = check_and_fix_masks(moving_lesion_mask_itk)

        if fixed_lesion_mask_itk is None or moving_lesion_mask_itk is None:
            print('Problem with either fixed or moving mask for Patient {}'.format(pat_id))
            review_patients.append(pat_id)
            continue

        # Save the lesion masks in the registration dir
        sitk.WriteImage(fixed_lesion_mask_itk,
                        os.path.join(pat_reg_dir, 'fixed_lesion_mask.nii.gz'))

        sitk.WriteImage(moving_lesion_mask_itk,
                        os.path.join(pat_reg_dir, 'moving_lesion_mask.nii.gz'))

        if args.mode == 'image':
            # Edit transform parameters to make resampling order 0
            for t_stage in range(3): # Rigid, affine, non-rigid
                t_file_path = os.path.join(pat_reg_dir, 'TransformParameters.{}.txt'.format(t_stage))

                # Edit transform paramater files to change resampling to order 0
                t_file_path_new = os.path.join(pat_reg_dir, 'TransformParameters_mask.{}.txt'.format(t_stage))
                t_file_editor = TransformParameterFileEditor(transform_parameter_file_path=t_file_path,
                                                             output_file_name=t_file_path_new)
                t_file_editor.modify_transform_parameter_file()

            transform_file_path = os.path.join(pat_reg_dir, 'TransformParameters_mask.2.txt')
        elif args.mode == 'mask':
            transform_file_path = os.path.join(pat_reg_dir, 'TransformParameters.2.txt')

        # Transformix interface used to resample (moving) lesion masks
        tr = TransformixInterface(parameters=transform_file_path,
                                  transformix_path=TRANSFORMIX_BIN)

        # Before we proceed, check if folding has occured!
        jac_det_path = tr.jacobian_determinant(output_dir=pat_reg_dir)
        jac_det_itk = sitk.ReadImage(jac_det_path)
        jac_det_np = sitk.GetArrayFromImage(jac_det_itk)

        # Check if registration has been successful before resampling the moving lesion mask
        if np.amin(jac_det_np) < 0:
            print('Registration has failed for patient {} since folding has occured'.format(pat_id))
            # Save the folding map
            folding_map_np = np.where(jac_det_np < 0, 1, 0).astype(np.uint8)
            folding_map_itk = sitk.GetImageFromArray(folding_map_np)
            folding_map_itk.SetOrigin(jac_det_itk.GetOrigin())
            folding_map_itk.SetSpacing(jac_det_itk.GetSpacing())
            folding_map_itk.SetDirection(jac_det_itk.GetDirection())
            sitk.WriteImage(folding_map_itk, os.path.join(pat_reg_dir, 'folding_map.nii.gz'))
            failed_registrations.append(pat_id)


        # Save each lesion as a separate mask
        try:
            n_fixed_lesions = create_separate_lesion_masks(os.path.join(pat_reg_dir, 'fixed_lesion_mask.nii.gz'))
            if n_fixed_lesions < 0:
                print('Error encountered while processing fixed lesion mask for Patient {}. Skipping'.format(pat_id))
                review_patients.append(pat_id)
                handle_lesion_separation_error(pat_dir=pat_reg_dir)
                continue
        except RuntimeError:
            print('Lesion annotations for patient {} need to reviewed'.format(pat_id))
            review_patients.append(pat_id)
            handle_lesion_separation_error(pat_dir=pat_reg_dir)
            continue

        try:
            n_moving_lesions = create_separate_lesion_masks(os.path.join(pat_reg_dir, 'moving_lesion_mask.nii.gz'))
            if n_moving_lesions < 0:
                print('Error encountered while processing moving lesion mask for Patient {}. Skipping'.format(pat_id))
                review_patients.append(pat_id)
                handle_lesion_separation_error(pat_dir=pat_reg_dir)
                continue
        except RuntimeError:
            print('Lesion annotations for patient {} need to reviewed'.format(pat_id))
            review_patients.append(pat_id)
            handle_lesion_separation_error(pat_dir=pat_reg_dir)
            continue

        # Resample each lesion in the moving image separately
        for m_lesion_idx in range(n_moving_lesions):

            moving_lesion_dir = os.path.join(pat_reg_dir,
                                             'moving_lesion_{}'.format(m_lesion_idx))

            lesion_fpath = os.path.join(moving_lesion_dir,
                                        'lesion.nii.gz')

            tr.transform_image(image_path=lesion_fpath,
                               output_dir=moving_lesion_dir)



    # Save list of patients whose lesion annotations need to be re-examined
    joblib.dump(value=review_patients,
                filename=os.path.join(args.reg_dir, 'patients_to_review.pkl'))

    joblib.dump(value=failed_registrations,
                filename=os.path.join(args.reg_dir, 'failed_registrations.pkl'))

    joblib.dump(value=missing_lesion_masks,
                filename=os.path.join(args.reg_dir, 'missing_lesion_masks.pkl'))
