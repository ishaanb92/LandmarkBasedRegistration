""""

Script to register a dataset of (paired) images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from register_images import *
import os
import shutil
from argparse import ArgumentParser
import datetime
import joblib



def create_datetime_object_from_str(dt_str):

    # The foldernames have the following format : yyyymmdd
    year = dt_str[0:4]
    month = dt_str[4:6]
    day = dt_str[-2:]

    dt_obj = datetime.datetime(int(year), int(month), int(day))
    return dt_obj

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_list_dir', type=str, help='Folder containing .pkl files with patient directories')
    parser.add_argument('--out_dir', type=str, help='Path to directory to dump elastix results', default='registraion_results')
    parser.add_argument('--p', type=str, help='Parameter file path(s)', nargs='+', default=None)
    parser.add_argument('--test', action='store_true', help='Use flag so that patients in the test-set are registered')
    parser.add_argument('--mode', type=str, default='image')

    args = parser.parse_args()

    if args.test is False:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'train_patients.pkl'))
        pat_dirs.extend(joblib.load(os.path.join(args.data_list_dir, 'val_patients.pkl')))
    else:
        pat_dirs = joblib.load(os.path.join(args.data_list_dir, 'test_patients.pkl'))

    if args.mode == 'mask':
        image_name = 'LiverMask_dilated.nii'
    elif args.mode == 'dce':
        image_name = 'DCE_mean.nii'
    elif args.mode == 'dwi':
        image_name = 'DWI_reg_mean.nii'

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


        fixed_image_path = os.path.join(pat_dir,
                                        scan_dirs[fixed_image_idx],
                                        image_name)

        moving_image_path = os.path.join(pat_dir,
                                         scan_dirs[moving_image_idx],
                                         image_name)

        # Use liver mask to guide registration for image-based registration
        if args.mode == 'dce' or args.mode == 'dwi':
            fixed_image_mask = os.path.join(pat_dir,
                                            scan_dirs[fixed_image_idx],
                                            'LiverMask_dilated.nii')

            moving_image_mask = os.path.join(pat_dir,
                                             scan_dirs[moving_image_idx],
                                             'LiverMask_dilated.nii')
        else:
            fixed_image_mask = None
            moving_image_mask = None

        if os.path.exists(fixed_image_path) is False or os.path.exists(moving_image_path) is False:
            print('One (or both) of the images cannot be found. Abort for patient {} '.format(p_id))
            continue

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

