""""

Script to register images using the python wrapper for elastic

"""

import os
from elastix.elastix_interface import *
from argparse import ArgumentParser
import shutil
import sys

ELASTIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/elastix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'

# Update LD_LIBRARY_PATH so that the elastix binary can find the .so file
def add_library_path(path):
    old_path = os.environ.get('LD_LIBRARY_PATH')

    if old_path is not None:
        os.environ['LD_LIBRARY_PATH'] = old_path + ":" + path
    else:
        os.environ['LD_LIBRARY_PATH'] = path

def register_image_pair(fixed_image=None,
                        moving_image=None,
                        param_file_list=None,
                        initial_transform=None,
                        out_dir=None):

    el = ElastixInterface(elastix_path=ELASTIX_BIN)

    el.register(fixed_image=fixed_image,
                moving_image=moving_image,
                parameters=param_file_list,
                initial_transform=initial_transform,
                output_dir=out_dir)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--f', type=str, help='fixed image file path')
    parser.add_argument('--m', type=str, help='moving image file path')
    parser.add_argument('--p', type=str, help='Parameter file path')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--t', type=str, help='Initial transformation', default=None)

    args = parser.parse_args()

    if os.path.exists(args.out_dir) is True:
        shutil.rmtree(args.out_dir)

    os.makedirs(args.out_dir)

    add_library_path(ELASTIX_LIB)

    register_image_pair(fixed_image=args.f,
                        moving_image=args.m,
                        param_file_list=[args.p],
                        initial_transform=args.t,
                        out_dir=args.out_dir)

