"""

Script to register DIR-Lab images

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from argparse import ArgumentParser
import os
import shutil
from elastix.elastix_interface import *
from lesionmatching.util_scripts.utils import add_library_path
import joblib


ELASTIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/elastix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'

COPD_DIR = '/home/ishaan/COPDGene/mha'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--params', type=str, help='Parameter file path(s)', nargs='+')
    parser.add_argument('--registration_out_dir', type=str, help='Output directory')
    parser.add_argument('--dataset', type=str, help='dirlab or copd')
    parser.add_argument('--landmarks_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--affine_reg_dir', type=str, default=None)
    args = parser.parse_args()

    if os.path.exists(args.registration_out_dir) is True:
        shutil.rmtree(args.registration_out_dir)

    os.makedirs(args.registration_out_dir)

    add_library_path(ELASTIX_LIB)
    el = ElastixInterface(elastix_path=ELASTIX_BIN)

    # Collect all the patient directories
    # Dataset == 'dirlab'
    if args.mode == 'all':
        pat_dirs = joblib.load('train_patients_dirlab.pkl')
        pat_dirs.extend(joblib.load('val_patients_dirlab.pkl'))
    elif args.mode == 'val':
        pat_dirs = joblib.load('val_patients_dirlab.pkl')
    # Dataset == 'copd'
    elif args.mode == 'test':
        pat_dirs = [f.path for f in os.scandir(COPD_DIR) if f.is_dir()]

    if args.dataset == 'dirlab':
        im_types = ['T00', 'T50']
    elif args.dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']

    for pdir in pat_dirs:
        image_prefix = pdir.split(os.sep)[-1]
        reg_out_dir = os.path.join(args.registration_out_dir, image_prefix)

        # Use affine pre-registered moving image if directory is supplied
        if args.affine_reg_dir is not None:
            affine_pdir = os.path.join(args.affine_reg_dir, image_prefix)
        else:
            affine_pdir = None

        os.makedirs(reg_out_dir)

        fixed_image_path = os.path.join(pdir, '{}_{}_iso.mha'.format(image_prefix,
                                                                     im_types[0]))

        fixed_mask_path = os.path.join(pdir, 'lung_mask_{}_dl_iso.mha'.format(im_types[0]))

        if args.affine_reg_dir is None:
            moving_image_path = os.path.join(pdir, '{}_{}_iso.mha'.format(image_prefix,
                                                                          im_types[1]))
            moving_mask_path = os.path.join(pdir, 'lung_mask_{}_dl_iso.mha'.format(im_types[1]))
        else: # Use the result of the affine registration as the moving image (and mask)
            moving_image_path = os.path.join(affine_pdir, 'result.0.mha')
            moving_mask_path = os.path.join(affine_pdir, 'moving_lung_mask_affine', 'result.mha')

        # Copy files to the output directory for convinient copying+viz
        shutil.copyfile(fixed_image_path, os.path.join(reg_out_dir, 'fixed_image.mha'))
        shutil.copyfile(moving_image_path, os.path.join(reg_out_dir, 'moving_image.mha'))

        shutil.copyfile(fixed_mask_path, os.path.join(reg_out_dir, 'fixed_mask.mha'))
        shutil.copyfile(moving_mask_path, os.path.join(reg_out_dir, 'moving_mask.mha'))

        # Landmark pairs are predicted using fixed and affine registered moving image
        if args.landmarks_dir is not None:
            fixed_landmarks = os.path.join(args.landmarks_dir,
                                           image_prefix,
                                           'fixed_landmarks_elx.txt')

            moving_landmarks = os.path.join(args.landmarks_dir,
                                            image_prefix,
                                            'moving_landmarks_elx.txt')
        else:
            fixed_landmarks = None
            moving_landmarks = None

        el.register(fixed_image=fixed_image_path,
                    moving_image=moving_image_path,
                    fixed_mask=fixed_mask_path,
                    moving_mask=moving_mask_path,
                    fixed_points=fixed_landmarks,
                    moving_points=moving_landmarks,
                    parameters=args.params,
                    initial_transform=None,
                    output_dir=reg_out_dir)



