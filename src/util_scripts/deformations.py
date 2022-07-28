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

def create_displacement_grid_affine(shape,
                                    angles=[0.0, 0.0, 0.0],
                                    scaling=None,
                                    shear_matrix=None,
                                    translation=None):

    ndim = len(shape)

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

    aff_transform = gryds.AffineTransformation(ndim=ndim,
                                               center=center,
                                               angles=angles,
                                               scaling=scaling,
                                               shear_matrix=shear_matrix,
                                               translation=translation)

    deformed_grid = image_grid.transform(aff_transform)

    flow_grid = deformed_grid.grid

    # FIXME!! REMOVE!!!!
#    np.testing.assert_array_equal(flow_grid, image_grid.grid)

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

        # Shape: N, C, D, H, W
        images = images.permute(0, 1, 4, 3, 2)
        b, c, d, h, w = images.shape
        batch_deformation_grid = np.zeros((b, h, w, 2),
                                          dtype=np.float32)
        images = images[:, :, 60, ...]

        # Loop over batch and generated a unique deformation grid for each batch element
        for batch_idx in range(images.shape[0]):
            image = images[batch_idx, ...]
            deformed_grid = create_displacement_grid_affine(shape=list(torch.squeeze(image, dim=0).shape),
                                                            angles=[np.pi/4])

            ndim , y, x = deformed_grid.shape
            deformed_grid = np.reshape(deformed_grid, (ndim, -1)).T
            deformed_grid = np.reshape(deformed_grid, (y, x, ndim))
            # Transpose to torch-friendly axes ordering (H, W, D, 3)
            batch_deformation_grid[batch_idx, ...] = np.transpose(deformed_grid, (1, 0, 2))


        #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
        batch_deformation_grid = torch.Tensor(batch_deformation_grid)

        print(images.shape)
        print(batch_deformation_grid.shape)

        deformed_images = F.grid_sample(input=images,
                                        grid=batch_deformation_grid,
                                        align_corners=False,
                                        mode="bilinear")


        save_dir = 'images_b_{}'.format(b_id)

        if os.path.exists(os.path.join(save_dir)) is False:
            os.makedirs(save_dir)

        for batch_idx in range(deformed_images.shape[0]):
            d_image = np.squeeze(deformed_images[batch_idx, ...].numpy(), axis=0)
            image = np.squeeze(images[batch_idx, ...].numpy(), axis=0)

            d_image_itk = sitk.GetImageFromArray(d_image)
            image_itk = sitk.GetImageFromArray(image)

            sitk.WriteImage(d_image_itk, os.path.join(save_dir, 'd_image.nii.gz'))
            sitk.WriteImage(image_itk, os.path.join(save_dir, 'image.nii.gz'))



#        print(images.shape)
#        print(deformed_images.shape)
#        print(torch.max(images))
#        print(torch.max(deformed_images))
#
#        print(torch.min(images))
#        print(torch.min(deformed_images))
#
#
#        batch_data['d_image'] = deformed_images
#
#
#        # Save as ITK images with meta-data
#        batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]
#
#        save_op_image = SaveImaged(keys='image',
#                                   output_postfix='image',
#                                   output_dir=save_dir,
#                                   separate_folder=False)
#
#        save_op_d_image = SaveImaged(keys='d_image',
#                                   output_postfix='d_image',
#                                   output_dir=save_dir,
#                                   separate_folder=False)
#
#        for batch_dict in batch_data:
#            save_op_image(batch_dict)
#            save_op_d_image(batch_dict)
#
