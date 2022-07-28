"""

Script with code related to image transformations

A image is transformed in two steps:
    1. Creating a deformation grid (pixel map from original to transformed)
    2. Interpolation (using the inverse map)

Since the deformations will be a part of the loss function, they need to be backprop friendly
so all functions will use torch primitives.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import torch
import torch.nn.functional as F
import gryds
import joblib
import os
from helper_functions import *
from datapipeline import *
import SimpleITK as sitk


def create_affine_transform(ndim=3,
                            center=[0, 0, 0],
                            angles=[0, 0, 0],
                            scaling=None,
                            shear_matrix=None,
                            translation=None):

    aff_transform = gryds.AffineTransformation(ndim=ndim,
                                               center=center,
                                               angles=angles,
                                               scaling=scaling,
                                               shear_matrix=shear_matrix,
                                               translation=translation)
    return aff_transform


def create_deformation_grid(shape,
                            transforms=[]):
    ndim = len(shape)
    assert(isinstance(transforms, list))
    assert(len(transforms)>=1)

    if ndim == 3:
        grid = np.array(np.meshgrid(np.linspace(-1, 1, shape[0]),
                                    np.linspace(-1, 1, shape[1]),
                                    np.linspace(-1, 1, shape[2]),
                                    indexing="ij"),
                        dtype=np.float32)
        center = [0, 0, 0]
    elif ndim == 2:
        grid = np.array(np.meshgrid(np.linspace(-1, 1, shape[0]),
                                    np.linspace(-1, 1, shape[1]),
                                    indexing="ij"),
                        dtype=np.float32)
        center = [0, 0]


    image_grid = gryds.Grid(grid=grid)


    deformed_grid = image_grid.transform(*transforms)

    flow_grid = deformed_grid.grid
    # Rearrange axes to make the deformation grid torch-friendly
    ndim, k, j, i = flow_grid.shape
    flow_grid = np.reshape(flow_grid, (ndim, -1)).T
    flow_grid = np.reshape(flow_grid, (k, j, i, ndim))

    return flow_grid


# Test deformations
if __name__ == '__main__':

    # Load all patient paths
    train_patients = joblib.load('../train_patients.pkl')
    train_dict = create_data_dicts([train_patients[0]],
                                   n_channels=1,
                                   channel_id=3)

    data_loader, transforms = create_dataloader(data_dicts=train_dict,
                                      train=False,
                                      batch_size=1)

    post_transforms = Compose([EnsureTyped(keys=['d_image']),
                               Invertd(keys=['d_image', 'label', 'image'],
                                       transform=transforms,
                                       orig_keys='image',
                                       meta_keys=['d_image_meta_dict', 'label_meta_dict', 'image_meta_dict'],
                                       nearest_interp=False,
                                       to_tensor=True)])

    for b_id, batch_data in enumerate(data_loader):
        images = batch_data['image']

        b, c, i, j, k = images.shape
        # To see why the deformation grid and image have different axes ordering
        # See: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/6
        # (i, j, k) --> (k, j, i) [i: z, j: y, k: x]
        batch_deformation_grid = np.zeros((b, k, j, i, 3),
                                          dtype=np.float32)

        # Loop over batch and generated a unique deformation grid for each batch element
        for batch_idx in range(images.shape[0]):
            image = images[batch_idx, ...]
            aff_transform = create_affine_transform(ndim=3,
                                                    angles=[np.pi/4, 0, 0],
                                                    center=[0, 0, 0])

            deformed_grid = create_deformation_grid(shape=[k, j, i],
                                                    transforms=[aff_transform])


            batch_deformation_grid[batch_idx, ...] = deformed_grid

        #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
        batch_deformation_grid = torch.Tensor(batch_deformation_grid)

        deformed_images = F.grid_sample(input=images,
                                        grid=batch_deformation_grid,
                                        align_corners=False,
                                        mode="bilinear")

        # (x, y, z) -> (i, j, k)
        deformed_images = deformed_images.permute(0, 1, 4, 3, 2)

        save_dir = 'images_b_{}'.format(b_id)

        batch_data['d_image'] = deformed_images


        # Save as ITK images with meta-data
        batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

        save_op_image = SaveImaged(keys='image',
                                   output_postfix='image',
                                   output_dir=save_dir,
                                   separate_folder=False)

        save_op_d_image = SaveImaged(keys='d_image',
                                   output_postfix='d_image',
                                   output_dir=save_dir,
                                   separate_folder=False)

        for batch_dict in batch_data:
            save_op_image(batch_dict)
            save_op_d_image(batch_dict)

