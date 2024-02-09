"""

Script to compute overlap between lung masks after registration

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from argparse import ArgumentParser
from elastix.transform_parameter_editor import TransformParameterFileEditor
from elastix.transformix_interface import TransformixInterface
import shutil
from lesionmatching.util_scripts.utils import add_library_path
import SimpleITK as sitk
import numpy as np

ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'
TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--registration_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, help='dirlab or copd', default='copd')
    parser.add_argument('--sift', action='store_true')
    args = parser.parse_args()


    pat_dirs = [f.path for f in os.scandir(args.registration_dir) if f.is_dir()]
    add_library_path(ELASTIX_LIB)

    if args.dataset == 'dirlab':
        im_types = ['T00', 'T50']
    elif args.dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']

    dice_scores = np.zeros((len(pat_dirs),),
                           dtype=np.float32)

    for idx, pdir in enumerate(pat_dirs):

        pid = pdir.split(os.sep)[-1]

        if args.sift is False:
            moving_mask_path = os.path.join(args.data_dir,
                                            pid,
                                            'lung_mask_{}_dl_iso.mha'.format(im_types[1]))
        else:
            moving_mask_path = os.path.join(args.data_dir,
                                            pid,
                                            'lung_mask_{}_dl.mha'.format(im_types[1]))

        resampled_mask_dir = os.path.join(pdir, 'transformed_lung_mask')

        if os.path.exists(resampled_mask_dir) is False:
            transform_file = os.path.join(pdir, 'TransformParameters.2.txt')
            # Edit transform parameters to make resampling order 0
            for t_stage in range(3): # Affine, B-Spline-1, B-Spline-2
                t_file_path = os.path.join(pdir, 'TransformParameters.{}.txt'.format(t_stage))

                # Edit transform paramater files to change resampling to order 0
                t_file_path_new = os.path.join(pdir, 'TransformParameters_mask.{}.txt'.format(t_stage))
                t_file_editor = TransformParameterFileEditor(transform_parameter_file_path=t_file_path,
                                                             output_file_name=t_file_path_new)
                t_file_editor.modify_transform_parameter_file()

            tr_obj = TransformixInterface(parameters=os.path.join(pdir, 'TransformParameters_mask.2.txt'),
                                          transformix_path=TRANSFORMIX_BIN)


            if os.path.exists(resampled_mask_dir) is True:
                shutil.rmtree(resampled_mask_dir)

            os.makedirs(resampled_mask_dir)

            # Resample moving lung mask
            resampled_mask_path = tr_obj.transform_image(image_path=moving_mask_path,
                                                         output_dir=resampled_mask_dir)
        else:
            resampled_mask_path = os.path.join(resampled_mask_dir,
                                               'result.mha')
        # Compute Dice Overlap
        if args.sift is False:
            fixed_mask_itk = sitk.ReadImage(os.path.join(args.data_dir,
                                                         pid,
                                                         'lung_mask_{}_dl_iso.mha'.format(im_types[0])))
        else:
            fixed_mask_itk = sitk.ReadImage(os.path.join(args.data_dir,
                                                         pid,
                                                         'lung_mask_{}_dl.mha'.format(im_types[0])))


        resampled_mask_itk = sitk.ReadImage(resampled_mask_path)

        fixed_mask_np = sitk.GetArrayFromImage(fixed_mask_itk)
        resampled_mask_np = sitk.GetArrayFromImage(resampled_mask_itk).astype(fixed_mask_np.dtype)

        dice = (np.sum(resampled_mask_np[fixed_mask_np==1.0])*2.0)/(np.sum(fixed_mask_np) + np.sum(resampled_mask_np))

        dice_scores[idx] = dice

    print('Dice scores :: {} +/- {}'.format(np.mean(dice_scores),
                                            np.std(dice_scores)))
