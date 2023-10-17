""""

To avoid confusion about points being transformed between stages, perform cascaded registration

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
