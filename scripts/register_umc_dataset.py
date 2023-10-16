""""

Script to register a dataset of (paired) images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from lesionmatching.util_scripts.utils import create_datetime_object_from_str, add_library_path
import os
import shutil
from argparse import ArgumentParser
import joblib
from elastix.elastix_interface import *


N_DCE_CHANNELS = 6
N_DWI_CHANNELS = 3

ELASTIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/elastix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--registration_out_dir', type=str, help='Path to directory to dump elastix results', default='registraion_results')
    parser.add_argument('--params', type=str, help='Parameter file path(s)', nargs='+', default=None)
    parser.add_argument('--data_mode', type=str, default='test')
    parser.add_argument('--landmarks_dir', type=str, default=None)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--world', action='store_true')
    parser.add_argument('--multichannel', action='store_true')

    args = parser.parse_args()

    if args.data_mode == 'all':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients_umc.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients_umc.pkl')))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'test_patients_umc.pkl')))
    elif args.data_mode == 'test':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'test_patients_umc.pkl'))
    elif args.data_mode == 'train':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients_umc.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients_umc.pkl')))

    # DCE-based registration
    image_names = []
    if args.multichannel is False: # Mean DCE image
        image_names.append('DCE_mean.nii')
    else:
        for chidx in range(N_DCE_CHANNELS):
            image_names.append('DCE_channel_{}.nii'.format(chidx))

    if os.path.exists(args.registration_out_dir) is True:
        shutil.rmtree(args.registration_out_dir)

    os.makedirs(args.registration_out_dir)

    add_library_path(ELASTIX_LIB)
    el = ElastixInterface(elastix_path=ELASTIX_BIN)

    # Loop over all the patients in the dataset
    for pat_dir in pat_dirs:
        p_id = pat_dir.split(os.sep)[-1]
        scan_dirs  = [f.name for f in os.scandir(pat_dir) if f.is_dir()]

        dt_obj_0 = create_datetime_object_from_str(scan_dirs[0])
        dt_obj_1 = create_datetime_object_from_str(scan_dirs[1])
        if dt_obj_0 > dt_obj_1: # Scan[0] is the moving image, Scan[1] is the fixed image
            moving_image_idx = 0
            fixed_image_idx = 1
        else: # Scan[1] is the moving image, Scan[0] is the fixed image
            moving_image_idx = 1
            fixed_image_idx = 0

        fixed_image_path = []
        moving_image_path = []

        files_present = True
        for iname in image_names:
            fpath = os.path.join(pat_dir, scan_dirs[fixed_image_idx], iname)
            mpath = os.path.join(pat_dir, scan_dirs[moving_image_idx], iname)
            if os.path.exists(fpath) is False or os.path.exists(mpath) is False:
                files_present = False
                break

            fixed_image_path.append(fpath)
            moving_image_path.append(mpath)

        if files_present is False:
            print('Fixed/moving images are absent for Patient {}. Skipping this patient'.format(p_id))
            continue

        # Use liver mask to guide registration for image-based registration
        if args.use_mask is True:
            fixed_image_mask = os.path.join(pat_dir,
                                            scan_dirs[fixed_image_idx],
                                            'LiverMask_dilated.nii')

            moving_image_mask = os.path.join(pat_dir,
                                             scan_dirs[moving_image_idx],
                                             'LiverMask_dilated.nii')
        else:
            fixed_image_mask = None
            moving_image_mask = None

        if args.landmarks_dir is not None:
            if args.world is True:
                fixed_landmarks = os.path.join(args.landmarks_dir,
                                               p_id,
                                               'fixed_landmarks_world.txt')

                moving_landmarks = os.path.join(args.landmarks_dir,
                                                p_id,
                                                'moving_landmarks_world.txt')
            else:
                fixed_landmarks = os.path.join(args.landmarks_dir,
                                               p_id,
                                               'fixed_landmarks_voxels.txt')

                moving_landmarks = os.path.join(args.landmarks_dir,
                                                p_id,
                                                'moving_landmarks_voxels.txt')
        else:
            fixed_landmarks = None
            moving_landmarks = None

        # Create output directory
        reg_out_dir = os.path.join(args.registration_out_dir, '{}'.format(p_id))
        os.makedirs(reg_out_dir)

        # Copy the fixed and moving images/masks to the registration directories
        for idx, (fimage, mimage) in enumerate(zip(fixed_image_path, moving_image_path)):
            shutil.copyfile(fimage, os.path.join(reg_out_dir, 'fixed_image_{}.nii'.format(idx)))
            shutil.copyfile(mimage, os.path.join(reg_out_dir, 'moving_image_{}.nii'.format(idx)))

        if fixed_image_mask is not None:
            shutil.copyfile(fixed_image_mask, os.path.join(reg_out_dir, 'fixed_liver_mask.nii'))

        if moving_image_mask is not None:
            shutil.copyfile(moving_image_mask, os.path.join(reg_out_dir, 'moving_liver_mask.nii'))


        initial_transform = None # TODO
        # Register the images
        el.register(fixed_image=fixed_image_path,
                    moving_image=moving_image_path,
                    fixed_mask=fixed_image_mask,
                    moving_mask=moving_image_mask,
                    fixed_points=fixed_landmarks,
                    moving_points=moving_landmarks,
                    parameters=args.params,
                    initial_transform=initial_transform,
                    output_dir=reg_out_dir)

