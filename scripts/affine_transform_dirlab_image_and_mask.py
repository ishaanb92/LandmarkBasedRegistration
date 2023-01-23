"""
To compare DIR-Lab (non-rigid) registration with and without landmarks, we use the affine registration as the starting point. Therefore, the moving lung mask is resampled using the affine transformation parameters before landmarks (and matches) are predicted using the trained NN

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
from elastix.transform_parameter_editor import TransformParameterFileEditor
from elastix.transformix_interface import TransformixInterface
import shutil
from lesionmatching.util_scripts.utils import add_library_path

ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'
TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--registration_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.registration_dir) if f.is_dir()]
    add_library_path(ELASTIX_LIB)

    for pdir in pat_dirs:

        p_id = pdir.split(os.sep)[-1]
        affine_transform_file = os.path.join(pdir, 'TransformParameters.0.txt')

        # Copy affinely registered image to data directory
        shutil.copy(os.path.join(pdir, 'result.0.mhd'),
                    os.path.join(args.data_dir, p_id, '{}_T50_iso_affine.mha'.format(p_id)))

        # Modify transform file to resample lung mask
        tr_editor = TransformParameterFileEditor(transform_parameter_file_path=affine_transform_file,
                                                 output_file_name=os.path.join(pdir, 'affine_transform_mask.txt'))
        tr_editor.modify_transform_parameter_file()

        tr_obj = TransformixInterface(parameters=os.path.join(pdir, 'affine_transform_mask.txt'),
                                      transformix_path=TRANSFORMIX_BIN)

        affine_resampled_mask_dir = os.path.join(pdir, 'moving_lung_mask_affine')

        if os.path.exists(affine_resampled_mask_dir) is True:
            shutil.rmtree(affine_resampled_mask_dir)

        os.makedirs(affine_resampled_mask_dir)

        resampled_mask_path = tr_obj.transform_image(image_path=os.path.join(pdir, 'moving_mask.mha'),
                                                     output_dir=affine_resampled_mask_dir)

        shutil.copy(resampled_mask_path,
                    os.path.join(args.data_dir, p_id, 'lung_mask_T50_dl_iso_affine.mha'))









