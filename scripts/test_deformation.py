"""

Script to test dataset visualization/deformations

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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umc')
    args = parser.parse_args()

    # Load all patient paths
    if args.dataset == 'umc':
        train_patients = joblib.load('train_patients.pkl')
        train_dict = create_data_dicts_lesion_matching([train_patients[0]])

        data_loader, transforms = create_dataloader_lesion_matching(data_dicts=train_dict,
                                                                    train=True,
                                                                    batch_size=1,
                                                                    data_aug=False,
                                                                    patch_size=(128, 128, 64))
    elif args.dataset == 'dirlab':
        train_patients = joblib.load('train_patients_dirlab.pkl')

        train_dict = create_data_dicts_dir_lab(train_patients[0:2])

        for tdict in train_dict:
            print(tdict['patient_id'])

        data_loader = create_dataloader_dir_lab(data_dicts=train_dict,
                                                test=False,
                                                batch_size=2,
                                                data_aug=False)

    print('Length of dataloader = {}'.format(len(data_loader)))

    for b_id, batch_data in enumerate(data_loader):

        print('Processing batch {}'.format(b_id+1))

        images = batch_data['image']

        metadata_list = detensorize_metadata(metadata=batch_data['metadata'],
                                             batchsz=images.shape[0])

        print('Images shape: {}'.format(images.shape))

        deformed_images = torch.zeros_like(images)

        batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                               coarse=True,
                                                               fine=True,
                                                               coarse_displacements=(12.8, 6.4, 3.2),
                                                               fine_displacements=(3.2, 3.2, 3.2),
                                                               coarse_grid_resolution=(4, 4, 4),
                                                               fine_grid_resolution=(8, 8, 8))

        if batch_deformation_grid is not None:
            deformed_images = F.grid_sample(input=images,
                                            grid=batch_deformation_grid,
                                            align_corners=True,
                                            mode="bilinear",
                                            padding_mode="border")

        if args.dataset == 'dirlab':
            save_dir = 'dirlab_viz'

            if os.path.exists(save_dir) is True:
                shutil.rmtree(save_dir)

            os.makedirs(save_dir)

        for batch_idx in range(images.shape[0]):
            # Sanity checks
            if torch.max(images[batch_idx, ...]) == torch.min(images[batch_idx, ...]):
                print('Min and max values of image are the same!!!!')
            if torch.max(deformed_images[batch_idx, ...]) == torch.min(deformed_images[batch_idx, ...]):
                print('Min and max values of deformed image are the same!!!!')

            # Handle metadata for NifTi
            if args.dataset == 'umc':

                save_dir = 'images_viz_{}_{}'.format(b_id, batch_idx)

                if os.path.exists(save_dir) is True:
                    shutil.rmtree(save_dir)

                os.makedirs(save_dir)

                image = images[batch_idx, ...]
                dimage = deformed_images[batch_idx, ...]
                image_array = torch.squeeze(image, dim=0).numpy() # Get rid of channel axis
                dimage_array = torch.squeeze(dimage, dim=0).numpy()
                image_fname = os.path.join(save_dir, 'image.nii')
                dimage_fname = os.path.join(save_dir, 'deformed_image.nii')
                image_nib = create_nibabel_image(image_array=image_array,
                                                 metadata_dict=batch_data['image_meta_dict'],
                                                 affine=batch_data['image_meta_dict']['affine'][batch_idx])

                dimage_nib = create_nibabel_image(image_array=dimage_array,
                                                  metadata_dict=batch_data['image_meta_dict'],
                                                  affine=batch_data['image_meta_dict']['affine'][batch_idx])

                save_nib_image(image_nib,
                               image_fname)

                save_nib_image(dimage_nib,
                               dimage_fname)

            # Handle metadata for DIR-Lab (DICOM/ITK)
            elif args.dataset == 'dirlab':
                print('Saving elem {} from batch {}'.format(batch_idx+1, b_id+1))
                save_ras_as_itk(img=images[batch_idx, ...],
                                metadata=metadata_list[batch_idx],
                                fname=os.path.join(save_dir, 'image_{}_{}.mha'.format(b_id, batch_idx)))

                save_ras_as_itk(img=deformed_images[batch_idx, ...],
                                metadata=metadata_list[batch_idx],
                                fname=os.path.join(save_dir, 'dimage_{}_{}.mha'.format(b_id, batch_idx)))
