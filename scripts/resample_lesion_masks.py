"""

Use the transformation parameters obtain from image/liver mask registration
to resample lesion masks using transformix

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

ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'
TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'




if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--out_dir', type=str, help='Path to directory to dump elastix results', default='registraion_results')
    parser.add_argument('--mode', type=str, default='image')

    args = parser.parse_args()


    pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients_umc.pkl'))
    pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients_umc.pkl')))
    pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'test_patients_umc.pkl')))

    add_library_path(ELASTIX_LIB)

    review_patients = []
    failed_registrations = []
    missing_lesion_masks = []

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split(os.sep)[-1]

        scan_dirs  = [f.name for f in os.scandir(pat_dir) if f.is_dir()]
        dt_obj_0 = create_datetime_object_from_str(scan_dirs[0])
        dt_obj_1 = create_datetime_object_from_str(scan_dirs[1])

        if dt_obj_0 > dt_obj_1: # Scan[0] is the moving image, Scan[1] is the fixed image
            moving_image_idx = 0
            fixed_image_idx = 1
        else: # Scan[1] is the moving image, Scan[0] is the fixed image
            moving_image_idx = 1
            fixed_image_idx = 0

        fixed_lesion_mask = os.path.join(pat_dir, scan_dirs[fixed_image_idx], '3DLesionAnnotations.nii')
        moving_lesion_mask = os.path.join(pat_dir, scan_dirs[moving_image_idx], '3DLesionAnnotations.nii')

        if os.path.exists(fixed_lesion_mask) is False or os.path.exists(moving_lesion_mask) is False:
            print('Lesion mask(s) missing for patient {}'.format(pat_id))
            missing_lesion_masks.append(pat_id)
            continue


        reg_dir = os.path.join(args.out_dir, pat_id)

        # The registration result for a patient may be absent
        # because it may have been deleted in a previous run
        # (because of a failed registraion, missing lesionmask etc.)
        if os.path.exists(reg_dir) is False:
            print('Registration directory not found for Patient {}'.format(pat_id))
            continue

        # Copy the fixed and moving lesion masks to the registration output directory
        shutil.copyfile(fixed_lesion_mask, os.path.join(reg_dir, 'fixed_lesion_mask.nii.gz'))
        shutil.copyfile(moving_lesion_mask, os.path.join(reg_dir, 'moving_lesion_mask.nii.gz'))

        if args.mode == 'image':
            # Edit transform parameters to make resampling order 0
            for t_stage in range(3): # Rigid, affine, non-rigid
                t_file_path = os.path.join(reg_dir, 'TransformParameters.{}.txt'.format(t_stage))

                # Edit transform paramater files to change resampling to order 0
                t_file_path_new = os.path.join(reg_dir, 'TransformParameters_mask.{}.txt'.format(t_stage))
                t_file_editor = TransformParameterFileEditor(transform_parameter_file_path=t_file_path,
                                                             output_file_name=t_file_path_new)
                t_file_editor.modify_transform_parameter_file()

            transform_file_path = os.path.join(reg_dir, 'TransformParameters_mask.2.txt')
        elif args.mode == 'mask':
            transform_file_path = os.path.join(reg_dir, 'TransformParameters.2.txt')

        # Transformix interface used to resample (moving) lesion masks
        tr = TransformixInterface(parameters=transform_file_path,
                                  transformix_path=TRANSFORMIX_BIN)

        # Before we proceed, check if folding has occured!
        jac_det_path = tr.jacobian_determinant(output_dir=reg_dir)
        jac_det_itk = sitk.ReadImage(jac_det_path)
        jac_det_np = sitk.GetArrayFromImage(jac_det_itk)

        # Save each lesion as a separate mask
        try:
            n_fixed_lesions = create_separate_lesion_masks(os.path.join(reg_dir, 'fixed_lesion_mask.nii.gz'))
            if n_fixed_lesions < 0:
                print('Error encountered while processing fixed lesion mask for Patient {}. Skipping'.format(pat_id))
                review_patients.append(pat_id)
                continue
        except RuntimeError:
            print('Lesion annotations for patient {} need to reviewed'.format(pat_id))
            review_patients.append(pat_id)
            continue

        try:
            n_moving_lesions = create_separate_lesion_masks(os.path.join(reg_dir, 'moving_lesion_mask.nii.gz'))
            if n_moving_lesions < 0:
                print('Error encountered while processing moving lesion mask for Patient {}. Skipping'.format(pat_id))
                review_patients.append(pat_id)
                continue
        except RuntimeError:
            print('Lesion annotations for patient {} need to reviewed'.format(pat_id))
            review_patients.append(pat_id)
            continue

        # Check if registration has been successful before resampling the moving lesion mask
        if np.amin(jac_det_np) < 0:
            print('Registration has failed for patient {} since folding has occured'.format(pat_id))
            # Save the folding map
            folding_map_np = np.where(jac_det_np < 0, 1, 0).astype(np.uint8)
            folding_map_itk = sitk.GetImageFromArray(folding_map_np)
            folding_map_itk.SetOrigin(jac_det_itk.GetOrigin())
            folding_map_itk.SetSpacing(jac_det_itk.GetSpacing())
            folding_map_itk.SetDirection(jac_det_itk.GetDirection())
            sitk.WriteImage(folding_map_itk, os.path.join(reg_dir, 'folding_map.nii.gz'))
            failed_registrations.append(pat_id)
            continue


        # Resample each lesion in the moving image separately
        for m_lesion_idx in range(n_moving_lesions):

            moving_lesion_dir = os.path.join(reg_dir,
                                             'moving_lesion_{}'.format(m_lesion_idx))

            lesion_fpath = os.path.join(moving_lesion_dir,
                                        'lesion.nii.gz')

            tr.transform_image(image_path=lesion_fpath,
                               output_dir=moving_lesion_dir)


    # Save list of patients whose lesion annotations need to be re-examined
    joblib.dump(value=review_patients,
                filename=os.path.join(args.out_dir, 'patients_to_review.pkl'))

    joblib.dump(value=failed_registrations,
                filename=os.path.join(args.out_dir, 'failed_registrations.pkl'))

    joblib.dump(value=missing_lesion_masks,
                filename=os.path.join(args.out_dir, 'missing_lesion_masks.pkl'))
