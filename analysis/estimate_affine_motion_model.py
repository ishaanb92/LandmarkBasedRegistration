"""

Using the results of the affine registration (with lung masks) of the DIR-Lab dataset, estimate parameters to model plausible affine transformations to train the landmarks DL-model

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lesionmatching.util_scripts.utils import get_dti_affine_transform_parameters
import pandas as pd

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--affine_reg_dir', type=str, required=True)

    args = parser.parse_args()


    pdirs = [f.path for f in os.scandir(args.affine_reg_dir) if f.is_dir()]

    affine_dict = {}
    affine_dict['theta_x'] = []
    affine_dict['theta_y'] = []
    affine_dict['theta_z'] = []
    affine_dict['gx'] = []
    affine_dict['gy'] = []
    affine_dict['gz'] = []
    affine_dict['sx'] = []
    affine_dict['sy'] = []
    affine_dict['sz'] = []
    affine_dict['tx'] = []
    affine_dict['ty'] = []
    affine_dict['tz'] = []

    for pdir in pdirs:
        # Step 1. Get R, G, S from Affine(DTI) transform output
        R, G, S, t, c = get_dti_affine_transform_parameters(fpath=os.path.join(pdir,
                                                                               'TransformParameters.0.txt'))

        affine_dict['theta_x'].append(R[0])
        affine_dict['theta_y'].append(R[1])
        affine_dict['theta_z'].append(R[2])
        affine_dict['gx'].append(G[0])
        affine_dict['gy'].append(G[1])
        affine_dict['gz'].append(G[2])
        affine_dict['sx'].append(S[0])
        affine_dict['sy'].append(S[1])
        affine_dict['sz'].append(S[2])
        affine_dict['tx'].append(t[0])
        affine_dict['ty'].append(t[1])
        affine_dict['tz'].append(t[2])

    # Convert dict to DF and save as a pickle file
    affine_df = pd.DataFrame.from_dict(affine_dict)
    affine_df.to_pickle(os.path.join(args.affine_reg_dir,
                                     'affine_transform_parameters.pkl'))
