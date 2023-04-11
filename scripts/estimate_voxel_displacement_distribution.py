"""

Using the thin-plate spline defined by manual landmarks, we have displacements for *EACH* voxel for *EVERY* patient in the DIR-Lab dataset.
We use these displacements to estimate a distribution, from which the parameters for the synthetic deformation are sampled

Note: The displacements are measured in number of voxels

See https://www.notion.so/Data-driven-generation-of-synthetic-deformations-52db56698d664edbb36f8778c69fd2bc?pvs=4
for detailed notes

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--displacement_dir', type=str, required=True)

    args = parser.parse_args()

    # 1. Read the data-frame with all the displacements
    disp_df = pd.read_pickle(os.path.join(args.displacement_dir,
                                          'displacement_df.pkl'))

    # Note: The order is now Z-Y-X to remain torch compliant
    zyx_displacements = disp_df[['Z-disp', 'Y-disp', 'X-disp']].to_numpy() # Shape: [N, 3]
    zyx_displacements = zyx_displacements.T # Shape: [3, N]

    # 2. Estimate PDF using Gaussian KDE
    disp_pdf = gaussian_kde(dataset=zyx_displacements)

    # 3. Save the PDF
    joblib.dump(value=disp_pdf,
                filename=os.path.join(args.displacement_dir,
                                      'disp_pdf.pkl'))

