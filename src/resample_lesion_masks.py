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
from register_dataset import create_datetime_object_from_str
from register_images import add_library_path
import joblib
import shutil
from utils.image_utils import return_lesion_coordinates
import SimpleITK as sitk
import numpy as np

ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'
TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'



def create_separate_lesion_masks(fname):
    """
    For ease of creating ground truth matches, we split the lesion annotation into separate
    masks, each mask contains a single lesion. This is done for both fixed and moving lesion
    masks.

    Return number of lesions so that each (moving) lesion can be resampled

    """
    # Is it fixed or moving?
    mask_type = fname.split(os.sep)[-1].split('_')[0]
    lesion_mask_itk = sitk.ReadImage(fname)
    lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk)

    if lesion_mask_np.ndim != 3:
        return -1

    # Handle non-binary masks!!!
    if np.amax(lesion_mask_np) > 1:
        lesion_mask_np = np.where(lesion_mask_np >=1, 1, lesion_mask_np)
        lesion_mask_itk_binary = sitk.GetImageFromArray(lesion_mask_np)
        lesion_mask_itk_binary.SetOrigin(lesion_mask_itk.GetOrigin())
        lesion_mask_itk_binary.SetDirection(lesion_mask_itk.GetDirection())
        lesion_mask_itk_binary.SetSpacing(lesion_mask_itk.GetSpacing())
        # Overwrite old file
        sitk.WriteImage(lesion_mask_itk_binary, fname)
        # Convert the fixed binary ITK image to a numpy ndarray
        lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk_binary)

    lesion_slices, n_lesions = return_lesion_coordinates(lesion_mask_np)

    for idx, lesion_slice in enumerate(lesion_slices):

        # Create a new mask for a single lesion
        single_lesion_mask = np.zeros_like(lesion_mask_np)
        single_lesion_mask[lesion_slice] += lesion_mask_np[lesion_slice]
        single_lesion_mask_itk = sitk.GetImageFromArray(single_lesion_mask)

        # Add metadata
        single_lesion_mask_itk.SetOrigin(lesion_mask_itk.GetOrigin())
        single_lesion_mask_itk.SetDirection(lesion_mask_itk.GetDirection())
        single_lesion_mask_itk.SetSpacing(lesion_mask_itk.GetSpacing())

        lesion_dir = os.path.join(reg_dir, '{}_lesion_{}'.format(mask_type, idx))
        if os.path.exists(lesion_dir) is True:
            shutil.rmtree(lesion_dir)
        os.makedirs(lesion_dir)

        lesion_fpath = os.path.join(lesion_dir, 'lesion.nii.gz')
        sitk.WriteImage(single_lesion_mask_itk,
                        lesion_fpath)

    return n_lesions


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--out_dir', type=str, help='Path to directory to dump elastix results', default='registraion_results')
    parser.add_argument('--test', action='store_true', help='Use flag so that patients in the test-set are registered')
    parser.add_argument('--mode', type=str, default='image')

    args = parser.parse_args()

    if args.test is False:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients.pkl')))
    else:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'test_patients.pkl'))

    add_library_path(ELASTIX_LIB)

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
            continue

        reg_dir = os.path.join(args.out_dir, pat_id)

        # Copy the fixed and moving lesion masks to the registration output directory
        shutil.copyfile(fixed_lesion_mask, os.path.join(reg_dir, 'fixed_lesion_mask.nii.gz'))
        shutil.copyfile(moving_lesion_mask, os.path.join(reg_dir, 'moving_lesion_mask.nii.gz'))

        if args.mode == 'image':
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

        # Save each lesion as a separate mask
        n_fixed_lesions = create_separate_lesion_masks(os.path.join(reg_dir, 'fixed_lesion_mask.nii.gz'))
        if n_fixed_lesions < 0:
            print('Error encountered while processing fixed lesion mask for Patient {}. Skipping'.format(pat_id))
            continue

        n_moving_lesions = create_separate_lesion_masks(os.path.join(reg_dir, 'moving_lesion_mask.nii.gz'))
        if n_moving_lesions < 0:
            print('Error encountered while processing moving lesion mask for Patient {}. Skipping'.format(pat_id))
            continue

        # Resample each lesion in the moving image separately
        for m_lesion_idx in range(n_moving_lesions):

            moving_lesion_dir = os.path.join(reg_dir,
                                             'moving_lesion_{}'.format(m_lesion_idx))

            lesion_fpath = os.path.join(moving_lesion_dir,
                                        'lesion.nii.gz')

            tr.transform_image(image_path=lesion_fpath,
                               output_dir=moving_lesion_dir)







