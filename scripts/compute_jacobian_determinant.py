"""

Script to compute Jacobian determininant for a given registration configuration

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
from argparse import ArgumentParser
from elastix.transformix_interface import *
from lesionmatching.util_scripts.utils import *
from lesionmatching.analysis.metrics import *

TRANSFORMIX_BIN = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/bin/transformix'
ELASTIX_LIB = '/user/ishaan/elastix_binaries/elastix-5.0.1-linux/lib'


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, default=None)

    args = parser.parse_args()


    # Patients in the registration directory
    pat_dirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    add_library_path(ELASTIX_LIB)


    for pdir in pat_dirs:

        transform_param_file = os.path.join(pdir,
                                            'TransformParameters.2.txt')
        tr_if = TransformixInterface(parameters=transform_param_file,
                                     transformix_path=TRANSFORMIX_BIN)

        jac_det_path = tr_if.jacobian_determinant(output_dir=pdir)


