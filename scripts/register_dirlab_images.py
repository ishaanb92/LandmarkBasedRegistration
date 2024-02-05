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

COPD_POINTS_DIR = '/home/ishaan/COPDGene/points'
DIRLAB_POINTS_DIR = '/home/ishaan/DIR-Lab/points'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--params', type=str, help='Parameter file path(s)', nargs='+')
    parser.add_argument('--registration_out_dir', type=str, help='Output directory')
    parser.add_argument('--dataset', type=str, help='dirlab or copd')
    parser.add_argument('--landmarks_dir', type=str, default=None)
    parser.add_argument('--initial_transform_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--affine_reg_dir', type=str, default=None)
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--smoothing_term', type=float, default=0.0)
    parser.add_argument('--use_lung_mask', action='store_true')
    parser.add_argument('--use_threshold', action='store_true')
    parser.add_argument('--use_threshold_local', action='store_true')
    parser.add_argument('--use_sift_landmarks', action='store_true')

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
        points_dir = DIRLAB_POINTS_DIR
    elif args.dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']
        points_dir = COPD_POINTS_DIR

    for pdir in pat_dirs:
        image_prefix = pdir.split(os.sep)[-1]
        reg_out_dir = os.path.join(args.registration_out_dir, image_prefix)

        # Use affine pre-registered moving image if directory is supplied
        if args.affine_reg_dir is not None:
            affine_pdir = os.path.join(args.affine_reg_dir, image_prefix)
        else:
            affine_pdir = None

        os.makedirs(reg_out_dir)

        if args.use_sift_landmarks is False:
            fixed_image_path = os.path.join(pdir, '{}_{}_iso.mha'.format(image_prefix,
                                                                         im_types[0]))
        else:
            fixed_image_path = os.path.join(pdir, '{}_{}.mha'.format(image_prefix,
                                                                     im_types[0]))

        fixed_mask_path = os.path.join(pdir, 'lung_mask_{}_dl_iso.mha'.format(im_types[0]))

        if args.affine_reg_dir is None:
            if args.use_sift_landmarks is False:
                moving_image_path = os.path.join(pdir, '{}_{}_iso.mha'.format(image_prefix,
                                                                              im_types[1]))
            else:
                moving_image_path = os.path.join(pdir, '{}_{}.mha'.format(image_prefix,
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
            if args.use_threshold is True: # Lenient threshold for global alignment
                assert(args.use_threshold_local is False)
                fixed_landmarks = os.path.join(args.landmarks_dir,
                                               image_prefix,
                                               'fixed_landmarks_elx_threshold.txt')

                if args.smoothing_term == 0:
                    moving_landmarks = os.path.join(args.landmarks_dir,
                                                    image_prefix,
                                                    'moving_landmarks_elx_threshold.txt')
                else:
                    moving_landmarks = os.path.join(args.landmarks_dir,
                                                    image_prefix,
                                                    'moving_landmarks_elx_threshold_{}.txt'.format(args.smoothing_term))
            else:
                if args.use_threshold_local is True: # Stricter threshold for "local" displacements
                    fixed_landmarks = os.path.join(args.landmarks_dir,
                                                   image_prefix,
                                                   'fixed_landmarks_elx_threshold_local.txt')

                    moving_landmarks = os.path.join(args.landmarks_dir,
                                                    image_prefix,
                                                    'moving_landmarks_elx_threshold_local.txt')

                else: # No threshold => Raw landmark corr. predicted by DL model
                    fixed_landmarks = os.path.join(args.landmarks_dir,
                                                   image_prefix,
                                                   'fixed_landmarks_elx.txt')

                    if args.smoothing_term == 0:
                        moving_landmarks = os.path.join(args.landmarks_dir,
                                                        image_prefix,
                                                        'moving_landmarks_elx.txt')
                    else:
                        moving_landmarks = os.path.join(args.landmarks_dir,
                                                        image_prefix,
                                                        'moving_landmarks_elx_{}.txt'.format(args.smoothing_term))
        else:
            if args.sanity is False:
                fixed_landmarks = None
                moving_landmarks = None
            else:
                if args.dataset == 'dirlab':
                    fixed_landmarks = os.path.join(points_dir,
                                                   image_prefix,
                                                   '{}_4D-75_T00_world_elx.txt'.format(image_prefix))
                elif args.dataset == 'copd':
                    fixed_landmarks = os.path.join(points_dir,
                                                   image_prefix,
                                                   '{}_300_iBH_world_r1_elx.txt'.format(image_prefix))

                if affine_pdir is not None:
                    moving_landmarks = os.path.join(affine_pdir, 'transformed_moving_landmarks_elx.txt')
                else:
                    if args.dataset == 'dirlab':
                        moving_landmarks = os.path.join(points_dir,
                                                        image_prefix,
                                                        '{}_4D-75_T50_world_elx.txt'.format(image_prefix))
                    elif args.dataset == 'copd':
                        moving_landmarks = os.path.join(points_dir,
                                                        image_prefix,
                                                        '{}_300_eBH_world_r1_elx.txt'.format(image_prefix))


        if args.use_lung_mask is False:
            fixed_mask_path = None
            moving_mask_path = None

        if args.initial_transform_dir is not None:
            initial_transform = os.path.join(args.initial_transform_dir,
                                             image_prefix,
                                             'TransformParameters.0.txt')
        else:
            initial_transform = None

        el.register(fixed_image=fixed_image_path,
                    moving_image=moving_image_path,
                    fixed_mask=fixed_mask_path,
                    moving_mask=moving_mask_path,
                    fixed_points=fixed_landmarks,
                    moving_points=moving_landmarks,
                    parameters=args.params,
                    initial_transform=initial_transform,
                    output_dir=reg_out_dir)



