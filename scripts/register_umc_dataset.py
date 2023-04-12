""""

Script to register a dataset of (paired) images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from lesionmatching.util_scripts.register_images import *
from lesionmatching.util_scripts.utils import create_datetime_object_from_str
import os
import shutil
from argparse import ArgumentParser
import joblib


N_DCE_CHANNELS = 6
N_DWI_CHANNELS = 3


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--out_dir', type=str, help='Path to directory to dump elastix results', default='registraion_results')
    parser.add_argument('--p', type=str, help='Parameter file path(s)', nargs='+', default=None)
    parser.add_argument('--multichannel', action='store_true', help='Use flag for (pairwise) multichannel registration')
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--input', type=str, default='dce')

    args = parser.parse_args()

    if args.mode == 'all':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients_umc.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients_umc.pkl')))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'test_patients_umc.pkl')))
    elif args.mode == 'test':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'test_patients_umc.pkl'))
    elif args.mode == 'train':
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients_umc.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients_umc.pkl')))


    if args.input == 'mask':
        image_name = ['LiverMask_dilated.nii']
    elif args.input == 'dce':
        if args.multichannel is False:
            image_name = ['DCE_mean.nii']
        else:
            image_name = []
            for channel in range(N_DCE_CHANNELS):
                image_name.append('DCE_channel_{}.nii'.format(channel))
    elif args.input == 'dwi':
        if args.multichannel is False:
            image_name = ['DWI_reg_mean.nii']
        else:
            image_name = []
            for channel in range(N_DCE_CHANNELS):
                image_name.append('DWI_channel_{}.nii'.format(channel))

    if os.path.exists(args.out_dir) is True:
        shutil.rmtree(args.out_dir)

    os.makedirs(args.out_dir)

    add_library_path(ELASTIX_LIB)

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
        for iname in image_name:
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
        if args.input == 'dce' or args.input == 'dwi':
            fixed_image_mask = os.path.join(pat_dir,
                                            scan_dirs[fixed_image_idx],
                                            'LiverMask_dilated.nii')

            moving_image_mask = os.path.join(pat_dir,
                                             scan_dirs[moving_image_idx],
                                             'LiverMask_dilated.nii')
        else:
            fixed_image_mask = None
            moving_image_mask = None

        # Create output directory
        out_dir = os.path.join(args.out_dir, '{}'.format(p_id))
        os.makedirs(out_dir)

        # Register the images
        register_image_pair(fixed_image=fixed_image_path,
                            moving_image=moving_image_path,
                            fixed_image_mask=fixed_image_mask,
                            moving_image_mask=moving_image_mask,
                            param_file_list=args.p,
                            out_dir=out_dir)

