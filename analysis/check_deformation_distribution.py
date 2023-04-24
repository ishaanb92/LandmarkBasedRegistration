"""

Script to check if the deformation distribution matches our design

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

from lesionmatching.data.deformations import *
from lesionmatching.data.datapipeline import *
from lesionmatching.util_scripts.image_utils import save_ras_as_itk
from lesionmatching.util_scripts.utils import detensorize_metadata
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import os
import shutil
from argparse import ArgumentParser
import monai
from monai.utils.misc import set_determinism
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DEFORM_DIR = '/home/ishaan/paper4/kp_detection/dirlab'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umc')
    parser.add_argument('--displacement_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    # Set the (global) seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_determinism(seed=args.seed)

    disp_dict = {}
    disp_dict['X-disp'] = []
    disp_dict['Y-disp'] = []
    disp_dict['Z-disp'] = []

    # Load all patient paths
    if args.dataset == 'umc':
        train_patients = joblib.load('train_patients_umc.pkl')
        train_dict = create_data_dicts_lesion_matching([train_patients[0]])


        data_loader, transforms = create_dataloader_lesion_matching(data_dicts=train_dict,
                                                                    train=True,
                                                                    batch_size=1,
                                                                    data_aug=False,
                                                                    patch_size=(128, 128, 64),
                                                                    seed=args.seed)

        coarse_displacements = (4, 8, 8)
        fine_displacements = (2, 4, 4)
        coarse_grid_resolution = (3, 3, 3)
        fine_grid_resolution = (6, 6, 6)

    elif args.dataset == 'dirlab':
        train_patients = joblib.load('train_patients_dirlab.pkl')

        train_patients.extend(joblib.load('val_patients_dirlab.pkl'))

        train_dict = create_data_dicts_dir_lab(train_patients[0:2])

        data_loader = create_dataloader_dir_lab(data_dicts=train_dict,
                                                test=False,
                                                batch_size=1,
                                                patch_size=(128, 128, 96),
                                                num_workers=1)

        disp_pdf = joblib.load(os.path.join(args.displacement_dir,
                                            'bspline_motion_pdf.pkl'))

        affine_df = pd.read_pickle(os.path.join(args.displacement_dir,
                                                'affine_transform_parameters.pkl'))

        coarse_grid_resolution = (2, 2, 2)
        fine_grid_resolution = (3, 3, 3)

    print('Length of dataloader = {}'.format(len(data_loader)))

    for b_id, batch_data_list in enumerate(data_loader):
        print('Processing batch {}'.format(b_id+1))

        if isinstance(batch_data_list, dict):
            batch_data_list = [batch_data_list]

        for sid, batch_data in enumerate(batch_data_list):
            images = batch_data['image']
            mask = batch_data['lung_mask']

            assert(isinstance(images, monai.data.meta_tensor.MetaTensor))

            if args.dataset == 'dirlab':
                metadata_list = detensorize_metadata(metadata=batch_data['metadata'],
                                                     batchsz=images.shape[0])

            deformed_images = torch.zeros_like(images)

            if args.dataset == 'umc':
                batch_deformation_grid, _ = create_batch_deformation_grid(shape=images.shape,
                                                                          coarse=True,
                                                                          fine=True,
                                                                          coarse_displacements=coarse_displacements,
                                                                          fine_displacements=fine_displacements,
                                                                          coarse_grid_resolution=coarse_grid_resolution,
                                                                          fine_grid_resolution=fine_grid_resolution)
            elif args.dataset == 'dirlab':
                batch_deformation_grid, jac_det = create_batch_deformation_grid_from_pdf(shape=images.shape,
                                                                                         non_rigid=True,
                                                                                         coarse=True,
                                                                                         fine=True,
                                                                                         disp_pdf=disp_pdf,
                                                                                         affine_df=None,
                                                                                         coarse_grid_resolution=coarse_grid_resolution,
                                                                                         fine_grid_resolution=fine_grid_resolution)

            # Shape : [B, H, W, D, 3]
            batch_deformation_grid_np = batch_deformation_grid.numpy()[0]


            # Shape: [3, H, W, D]
            batch_deformation_grid_np = np.transpose(batch_deformation_grid_np,
                                                     (3, 0, 1, 2))

            _, x, y, z = batch_deformation_grid_np.shape

            grid = np.array(np.meshgrid(np.linspace(-1, 1, x),
                                        np.linspace(-1, 1, y),
                                        np.linspace(-1, 1, z),
                                        indexing="ij"),
                            dtype=np.float32)


            grid_torch = np.copy(grid)
            grid_torch[0, :, :, :] = grid[2, :, :, :]
            grid_torch[2, :, :, :] = grid[0, :, :, :]

            print('Deformed grid shape: {}'.format(batch_deformation_grid_np.shape))
            print('Grid shape: {}'.format(grid_torch.shape))

            diff_grid = np.subtract(batch_deformation_grid_np,
                                    grid_torch)

            scaling_array = np.array([z/2, y/2, x/2],
                                     dtype=np.float32)

            scaling_array = scaling_array[:, None, None, None]

            diff_grid_scaled_displacements = np.multiply(diff_grid,
                                                         scaling_array)

            disp_z = diff_grid_scaled_displacements[0, ...]
            disp_y = diff_grid_scaled_displacements[1, ...]
            disp_x = diff_grid_scaled_displacements[2, ...]

            # Use only relevant deformations (within lungs)

            mask_hat = F.grid_sample(input=mask,
                                     grid=batch_deformation_grid,
                                     align_corners=True,
                                     mode="nearest",
                                     padding_mode="border")

            # Get rid of batch and channel axes and convert torch Tensor to np
            mask_np = mask[0, 0, ...].numpy()
            mask_hat_np = mask_hat[0, 0, ...].numpy()

            displacement_mask = np.where(mask_np+mask_hat_np>=1,
                                         1,
                                         0).astype(np.float32)


            disp_dict['X-disp'].extend(list(disp_x[displacement_mask==1].flatten()))
            disp_dict['Y-disp'].extend(list(disp_y[displacement_mask==1].flatten()))
            disp_dict['Z-disp'].extend(list(disp_z[displacement_mask==1].flatten()))

    print(len(disp_dict['X-disp']))
    print(len(disp_dict['Y-disp']))
    print(len(disp_dict['Z-disp']))

    # Construct pandas DF for easy plotting
    disp_df = pd.DataFrame.from_dict(disp_dict)

    fig, axs = plt.subplots(3,
                            figsize=(20, 20))

    sns.histplot(data=disp_df,
                 x='X-disp',
                 ax=axs[0])


    sns.histplot(data=disp_df,
                 x='Y-disp',
                 ax=axs[1])


    sns.histplot(data=disp_df,
                 x='Z-disp',
                 ax=axs[2])

    fig.savefig(os.path.join(args.displacement_dir,
                             'synthetic_displacements.png'),
                bbox_inches='tight')

    disp_df.to_pickle(os.path.join(args.displacement_dir,
                                   'synthetic_displacement_df.pkl'))
