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

image_types = ['T00', 'T50']

ELASTIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/elastix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--params', type=str, help='Parameter file path(s)', nargs='+')
    parser.add_argument('--out_dir', type=str, help='Output directory')

    args = parser.parse_args()

    if os.path.exists(args.out_dir) is True:
        shutil.rmtree(args.out_dir)

    os.makedirs(args.out_dir)

    add_library_path(ELASTIX_LIB)
    el = ElastixInterface(elastix_path=ELASTIX_BIN)

    # Collect all the patient directories
    pat_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    for pdir in pat_dirs:
        image_prefix = pdir.split(os.sep)[-1]
        reg_out_dir = os.path.join(args.out_dir, image_prefix)
        os.makedirs(reg_out_dir)

        fixed_image_path = os.path.join(pdir, '{}_T00_smaller.mha'.format(image_prefix))
        moving_image_path = os.path.join(pdir, '{}_T50_smaller.mha'.format(image_prefix))

        # Copy files to the output directory for convinient copying+viz

        shutil.copyfile(fixed_image_path, os.path.join(reg_out_dir, 'fixed_image.mha'))
        shutil.copyfile(moving_image_path, os.path.join(reg_out_dir, 'moving_image.mha'))

        el.register(fixed_image=fixed_image_path,
                    moving_image=moving_image_path,
                    fixed_mask=None,
                    moving_mask=None,
                    parameters=args.params,
                    initial_transform=None,
                    output_dir=reg_out_dir)



