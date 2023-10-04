"""

Script to test dataset visualization/deformations

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from lesionmatching.data.deformations import *
from lesionmatching.data.datapipeline import *
from lesionmatching.util_scripts.image_utils import *
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

COPD_DIR = '/home/ishaan/COPDGene/mha'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umc')
    parser.add_argument('--displacement_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--multichannel', action='store_true')
    args = parser.parse_args()

    # Set the (global) seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_determinism(seed=args.seed)

    # Load all patient paths
    if args.dataset == 'umc':
        umc_dataset_stats = joblib.load('umc_dataset_stats.pkl')
        train_patients = joblib.load('train_patients_umc.pkl')
        train_dict = create_data_dicts_lesion_matching([train_patients[0]],
                                                       multichannel=args.multichannel)


        data_loader, transforms = create_dataloader_lesion_matching(data_dicts=train_dict,
                                                                    train=True,
                                                                    batch_size=1,
                                                                    data_aug=False,
                                                                    patch_size=(128, 128, 128),
                                                                    seed=args.seed)

        # Patient positioning
        translation_max = [0, 0, 20]
        rotation_max = [0, 0, (np.pi*10)/180]

        # Respiratory motion
        coarse_grid_resolution = (4, 4, 4)
        coarse_displacements = (1.2, 12.4, 8.4)


    elif args.dataset == 'dirlab':
        train_patients = joblib.load('train_patients_dirlab.pkl')

        train_dict = create_data_dicts_dir_lab(train_patients[3:4])

        for tdict in train_dict:
            print(tdict['patient_id'])

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


    elif args.dataset == 'copd':
        train_patients = [f.path for f in os.scandir(COPD_DIR) if f.is_dir()]

        train_dict = create_data_dicts_dir_lab(train_patients[3:4],
                                               dataset='copd')

        data_loader = create_dataloader_dir_lab(data_dicts=train_dict,
                                                test=False,
                                                batch_size=1,
                                                patch_size=(128, 128, 96),
                                                num_workers=1)

        disp_pdf = joblib.load(os.path.join(args.displacement_dir,
                                            'bspline_motion_pdf_copd.pkl'))

        affine_df = pd.read_pickle(os.path.join(args.displacement_dir,
                                                'affine_transform_parameters_copd.pkl'))

        coarse_grid_resolution = (2, 2, 2)
        fine_grid_resolution = (3, 3, 3)



    print('Length of dataloader = {}'.format(len(data_loader)))

    if args.dataset == 'dirlab':
        save_dir = 'dirlab_viz'
    elif args.dataset  == 'copd':
        save_dir = 'copd_viz'
    elif args.dataset == 'umc':
        save_dir = 'umc_viz'

    if os.path.exists(save_dir) is True:
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    for b_id, batch_data_list in enumerate(data_loader):
        print('Processing batch {}'.format(b_id+1))

        if isinstance(batch_data_list, dict):
            batch_data_list = [batch_data_list]

        for sid, batch_data in enumerate(batch_data_list):
            images = batch_data['image']

            assert(isinstance(images, monai.data.meta_tensor.MetaTensor))

            if args.dataset == 'dirlab' or args.dataset == 'copd':
                metadata_list = detensorize_metadata(metadata=batch_data['metadata'],
                                                     batchsz=images.shape[0])

            print('Images shape: {}'.format(images.shape))

            deformed_images = torch.zeros_like(images)

            if args.dataset == 'umc':
                batch_deformation_grid, _ = create_batch_deformation_grid(shape=images.shape,
                                                                          coarse_displacements=coarse_displacements,
                                                                          fine_displacements=None,
                                                                          coarse_grid_resolution=coarse_grid_resolution,
                                                                          fine_grid_resolution=None,
                                                                          translation_max=translation_max,
                                                                          rotation_max=rotation_max)
                if args.multichannel is True:
                    images = min_max_rescale_umc(images=images,
                                                 max_value=umc_dataset_stats['max_multichannel'],
                                                 min_value=umc_dataset_stats['min_multichannel'])
                else:
                    images = min_max_rescale_umc(images=images,
                                                 max_value=umc_dataset_stats['mean_max'],
                                                 min_value=umc_dataset_stats['mean_min'])

            elif args.dataset == 'dirlab' or args.dataset == 'copd':
                batch_deformation_grid, jac_det = create_batch_deformation_grid_from_pdf(shape=images.shape,
                                                                                         non_rigid=True,
                                                                                         coarse=True,
                                                                                         fine=True,
                                                                                         disp_pdf=disp_pdf,
                                                                                         affine_df=affine_df,
                                                                                         coarse_grid_resolution=coarse_grid_resolution,
                                                                                         fine_grid_resolution=fine_grid_resolution)
            if batch_deformation_grid is not None:
                deformed_images = F.grid_sample(input=images,
                                                grid=batch_deformation_grid,
                                                align_corners=True,
                                                mode="bilinear",
                                                padding_mode="zeros")


            if args.dataset == 'umc':
                images = gamma_transformation(images,
                                              gamma=[0.5, 1.5])

                deformed_images = gamma_transformation(deformed_images,
                                                       gamma=[0.5, 1.5])

            for batch_idx in range(images.shape[0]):
                # Sanity checks
                if torch.max(images[batch_idx, ...]) == torch.min(images[batch_idx, ...]):
                    print('Min and max values of image are the same!!!!')
                if torch.max(deformed_images[batch_idx, ...]) == torch.min(deformed_images[batch_idx, ...]):
                    print('Min and max values of deformed image are the same!!!!')

                # Handle metadata for NifTi
                if args.dataset == 'umc':

                    image = images[batch_idx, ...]
                    dimage = deformed_images[batch_idx, ...]
                    image_array = torch.squeeze(image, dim=0).numpy() # Get rid of channel axis
                    dimage_array = torch.squeeze(dimage, dim=0).numpy()
                    image_fname = os.path.join(save_dir, 'image_{}.nii'.format(batch_idx))
                    dimage_fname = os.path.join(save_dir, 'deformed_image_{}.nii'.format(batch_idx))

                    image_nib = create_nibabel_image(image_array=image_array,
                                                     metadata_dict=image.meta,
                                                     affine=image.meta['affine'])

                    dimage_nib = create_nibabel_image(image_array=dimage_array,
                                                      metadata_dict=dimage.meta,
                                                      affine=dimage.meta['affine'])
                    print('About to save images')
                    save_nib_image(image_nib,
                                   image_fname)

                    save_nib_image(dimage_nib,
                                   dimage_fname)

                # Handle metadata for DIR-Lab (DICOM/ITK)
                elif args.dataset == 'dirlab' or args.dataset == 'copd':
                    print('Saving elem {} from batch {}'.format(batch_idx+1, b_id+1))
                    save_ras_as_itk(img=images[batch_idx, ...],
                                    metadata=metadata_list[batch_idx],
                                    fname=os.path.join(save_dir, 'image_{}_{}.mha'.format(b_id, batch_idx)))

                    save_ras_as_itk(img=deformed_images[batch_idx, ...],
                                    metadata=metadata_list[batch_idx],
                                    fname=os.path.join(save_dir, 'dimage_{}_{}.mha'.format(b_id, batch_idx)))
