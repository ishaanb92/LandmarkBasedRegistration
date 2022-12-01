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
from datapipeline import *
import SimpleITK as sitk
import shutil

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


def create_bspline_transform(coarse=True,
                             shape=None,
                             displacements=(5, 3, 3)):
    """

    Displacements are in terms of voxels
    dispacement[k] = m => Along the kth axis, displacements will be in the range (-m , m)

    """

    x, y, z = shape

    if coarse is True:
        random_grid = np.random.rand(3, 4, 4, 4).astype(np.float32) # Make a random 3D 3 x 3 x 3 grid
        random_grid = random_grid*2 - 1 # Shift range to [-1, 1]
        random_grid[0, ...] = random_grid[0, ...]*(1/x)*displacements[0]
        random_grid[1, ...] = random_grid[1, ...]*(1/y)*displacements[1]
        random_grid[2, ...] = random_grid[2, ...]*(1/z)*displacements[2]
    else: # Fine with a larger control grid
        random_grid = np.random.rand(3, 8, 8, 8).astype(np.float32) # Make a random 3D 4 x 4 x 4 grid
        random_grid = random_grid*2 - 1 # Shift range to [-1, 1]
        random_grid[0, ...] = random_grid[0, ...]*(1/x)*displacements[0]
        random_grid[1, ...] = random_grid[1, ...]*(1/y)*displacements[1]
        random_grid[2, ...] = random_grid[2, ...]*(1/z)*displacements[2]

    bspline_transform = gryds.BSplineTransformation(grid=random_grid,
                                                    order=3)

    return bspline_transform

def create_deformation_grid(grid=None,
                            shape=None,
                            transforms=[]):

    if grid is None:
        ndim = len(shape)
        assert(isinstance(transforms, list))

        if ndim == 3:
            grid = np.array(np.meshgrid(np.linspace(0, 1, shape[0]),
                                        np.linspace(0, 1, shape[1]),
                                        np.linspace(0, 1, shape[2]),
                                        indexing="ij"),
                            dtype=np.float32)

            center = [0, 0, 0]
        elif ndim == 2:
            grid = np.array(np.meshgrid(np.linspace(0, 1, shape[0]),
                                        np.linspace(0, 1, shape[1]),
                                        indexing="ij"),
                            dtype=np.float32)
            center = [0, 0]


    image_grid = gryds.Grid(grid=grid)

    if len(transforms) == 0: # DEBUG
        return image_grid.grid

    deformed_grid = image_grid.transform(*transforms)

    jac_det = deformed_grid.jacobian_det(*transforms)

    # Check for folding
    if np.amin(jac_det) < 0:
        print('Folding has occured!. Skip this batch')
        return None

    dgrid = deformed_grid.grid

    # Rescale to [-1, 1] so that this is compliant with torch.nn.functional.grid_sample()
    dgrid = 2*dgrid - 1

    return dgrid


def create_batch_deformation_grid(shape,
                                  device='cpu',
                                  dummy=False,
                                  non_rigid=True,
                                  coarse=True,
                                  fine=False,
                                  coarse_displacements=(4, 4, 3),
                                  fine_displacements=(0.75, 0.75, 0.75)):

    b, c, i, j, k = shape

    # See: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/6
    # grid[:, i, j, k, 0] = new_x (k')
    # grid[:, i, j, k, 1] = new_y (j')
    # grid[:, i, j, k, 2] = new_z (i')
    batch_deformation_grid = np.zeros((b, i, j, k, 3),
                                      dtype=np.float32)

    # Loop over batch and generated a unique deformation grid for each image in the batch
    for batch_idx in range(b):
        transforms = []
        if dummy is False:
            if non_rigid is True:
                if coarse is True:
                    elastic_transform_coarse = create_bspline_transform(coarse=True,
                                                                        shape=[k, j, i],
                                                                        displacements=coarse_displacements)
                    transforms.append(elastic_transform_coarse)
                if fine is True:
                    elastic_transform_fine = create_bspline_transform(coarse=False,
                                                                      shape=[k, j, i],
                                                                      displacements=fine_displacements)
                    transforms.append(elastic_transform_fine)

            else: # Translation only
                translation_transform = create_affine_transform(translation=[0, 0, 0.1])
                transforms.append(translation_transform)

        # Create deformation grid by composing transforms
        deformed_grid = create_deformation_grid(shape=[k, j, i],
                                                transforms=transforms)

        if deformed_grid is None:
            return None

        # Rearrange axes to make the deformation grid torch-friendly
        ndim, k, j, i = deformed_grid.shape
        deformed_grid = np.reshape(deformed_grid, (ndim, -1)).T
        deformed_grid = np.reshape(deformed_grid, (k, j, i, ndim))

        # Re-order axes to HWD (ijk) for grid_sample
        deformed_grid = np.transpose(deformed_grid, (2, 1, 0, 3))

        batch_deformation_grid[batch_idx, ...] = deformed_grid

    #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
    batch_deformation_grid = torch.Tensor(batch_deformation_grid).to(device)
    return batch_deformation_grid



# Test deformations
if __name__ == '__main__':

    # Load all patient paths
    train_patients = joblib.load('../train_patients.pkl')
    train_dict = create_data_dicts_lesion_matching(train_patients[0:2])

    data_loader, transforms = create_dataloader_lesion_matching(data_dicts=train_dict,
                                                                train=True,
                                                                batch_size=2,
                                                                data_aug=False,
                                                                patch_size=(96, 96, 48))

    print('Length of dataloader = {}'.format(len(data_loader)))

    for b_id, batch_data in enumerate(data_loader):
        images = batch_data['image']
        print('Images shape: {}'.format(images.shape))

        deformed_images = torch.zeros_like(images)

        batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                               dummy=False,
                                                               non_rigid=True,
                                                               coarse=True,
                                                               fine=True,
                                                               coarse_displacements=(4, 4, 4),
                                                               fine_displacements=(1, 1, 1))

        if batch_deformation_grid is not None:
            deformed_images = F.grid_sample(input=images,
                                            grid=batch_deformation_grid,
                                            align_corners=True,
                                            mode="bilinear",
                                            padding_mode="border")


        for batch_idx in range(images.shape[0]):
            if torch.max(images[batch_idx, ...]) == torch.min(images[batch_idx, ...]):
                print('Min and max values of image are the same!!!!')

            if torch.max(deformed_images[batch_idx, ...]) == torch.min(deformed_images[batch_idx, ...]):
                print('Min and max values of deformed image are the same!!!!')

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
