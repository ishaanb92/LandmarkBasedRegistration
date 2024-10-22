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
import SimpleITK as sitk
import shutil
from argparse import ArgumentParser
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from lesionmatching.util_scripts.utils import add_library_path

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


def create_affine_transform_from_df(ndim=3,
                                    param_df=None,
                                    shape=None):
    """
    Create affine transform using motion model estimated using DIR-Lab data

    """

    x, y, z = shape

    angles = [np.random.uniform(low=param_df['theta_x'].min(),
                                high=param_df['theta_x'].max()),
              np.random.uniform(low=param_df['theta_y'].min(),
                                high=param_df['theta_y'].max()),
              np.random.uniform(low=param_df['theta_z'].min(),
                                high=param_df['theta_z'].max())]


    scaling = [np.random.uniform(low=param_df['sx'].min(),
                                high=param_df['sx'].max()),
               np.random.uniform(low=param_df['sy'].min(),
                                high=param_df['sy'].max()),
               np.random.uniform(low=param_df['sz'].min(),
                                high=param_df['sz'].max())]

    translation = [np.random.uniform(low=param_df['tx'].min(),
                                     high=param_df['tx'].max()),
                   np.random.uniform(low=param_df['ty'].min(),
                                     high=param_df['ty'].max()),
                   np.random.uniform(low=param_df['tz'].min(),
                                     high=param_df['tz'].max())]

    # Scale to [0, 1]
    translation = np.divide(translation, shape)

    # Shear factors cannot be randomly sampled because a "valid" shear matrix
    # must have a determinant = 1 (shears preserve volume)
    shear_idx = np.random.randint(low=0,
                                  high=param_df.shape[0])
    gx = param_df['gx'][shear_idx]
    gy = param_df['gy'][shear_idx]
    gz = param_df['gz'][shear_idx]

    Gx = np.array([[1, 0 , 0], [gx, 1, 0], [gx, 0, 1]],
                  dtype=np.float32)
    Gy = np.array([[1, gy , 0], [0, 1, 0], [0, gy, 1]],
                  dtype=np.float32)
    Gz = np.array([[1, 0 , gz], [0, 1, gz], [0, 0, 1]],
                  dtype=np.float32)

    G = np.dot(np.dot(Gx, Gy), Gz)

    assert(np.isclose(np.linalg.det(G), 1))

    # Define the affine transform
    aff_transform = gryds.AffineTransformation(ndim=3,
                                               center=[0.5, 0.5, 0.5],
                                               angles=angles,
                                               scaling=scaling,
                                               shear_matrix=G,
                                               translation=translation)
    return aff_transform

def create_bspline_transform(shape=None,
                             displacements=(5, 3, 3),
                             grid_resolution=(4, 4, 4)):
    """

    Displacements are in terms of voxels
    dispacement[k] = m => Along the kth axis, displacements will be in the range (-m , m)

    """

    x, y, z = shape

    random_grid = np.random.rand(3,
                                 grid_resolution[0],
                                 grid_resolution[1],
                                 grid_resolution[2]).astype(np.float32)

    random_grid = random_grid*2 - 1 # Shift range to [-1, 1]
    random_grid[0, ...] = random_grid[0, ...]*(1/x)*displacements[0]
    random_grid[1, ...] = random_grid[1, ...]*(1/y)*displacements[1]
    random_grid[2, ...] = random_grid[2, ...]*(1/z)*displacements[2]


    bspline_transform = gryds.BSplineTransformation(grid=random_grid,
                                                    order=3)

    return bspline_transform

def create_bspline_transform_from_pdf(shape=None,
                                      disp_pdf=None,
                                      grid_resolution=(4, 4, 4)):
    """

    Displacements are in terms of voxels
    dispacement[k] = m => Along the kth axis, displacements will be in the range (-m , m)

    """

    x, y, z = shape

    # 1. Sample displacements from the PDF
    # Since the numpy seed is set at the start, the order of seeds (and hence sampled displacements)
    # will be fixed for a given "main" seed
    random_grid = disp_pdf.resample(size=grid_resolution[0]*grid_resolution[1]*grid_resolution[2],
                                    seed=np.random.randint(low=1000, high=1000000))

    # 2. Reshape random grid
    random_grid = np.reshape(random_grid, (3,
                                           grid_resolution[0],
                                           grid_resolution[1],
                                           grid_resolution[2]))


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

        elif ndim == 2:
            grid = np.array(np.meshgrid(np.linspace(0, 1, shape[0]),
                                        np.linspace(0, 1, shape[1]),
                                        indexing="ij"),
                            dtype=np.float32)

    image_grid = gryds.Grid(grid=grid)

    if len(transforms) == 0: # DEBUG
        return image_grid.grid


    jac_det = image_grid.jacobian_det(*transforms)

    # Check for folding
    if np.amin(jac_det) < 0:
        print('Folding has occured!. Skip this batch')
        return None, None

    deformed_grid = image_grid.transform(*transforms)

    dgrid = deformed_grid.grid

    # Rescale to [-1, 1] so that this is compliant with torch.nn.functional.grid_sample()
    dgrid = 2*dgrid - 1

    return dgrid, jac_det


def create_batch_deformation_grid(shape,
                                  device='cpu',
                                  coarse_displacements=(4, 4, 3),
                                  fine_displacements=(0.75, 0.75, 0.75),
                                  coarse_grid_resolution=(4, 4, 4),
                                  fine_grid_resolution=(8, 8, 8),
                                  translation_max=None,
                                  rotation_max=None,
                                  shear=None,
                                  scaling=None):

    b, c, i, j, k = shape

    batch_deformation_grid = np.zeros((b, i, j, k, 3),
                                      dtype=np.float32)

    jac_det = np.zeros((b, i, j, k, 3),
                      dtype=np.float32)

    # Loop over batch and generated a unique deformation grid for each image in the batch
    for batch_idx in range(b):
        transforms = []

        # Affine/Rigid motion
        if translation_max is not None or rotation_max is not None:
            tx = np.random.uniform(-translation_max[0], translation_max[0])
            ty = np.random.uniform(-translation_max[1], translation_max[1])
            tz = np.random.uniform(-translation_max[2], translation_max[2])
            translation = [tx, ty, tz]
            # Scale to [0, 1]
            scaled_translation = np.divide(np.array(translation),
                                           np.array([i, j, k]))

            angles = [np.random.uniform(-rotation_max[0], rotation_max[0]),
                      np.random.uniform(-rotation_max[1], rotation_max[1]),
                      np.random.uniform(-rotation_max[2], rotation_max[2])]

            affine_transform = create_affine_transform(center=[0.5, 0.5, 0.5],
                                                       angles=angles,
                                                       translation=scaled_translation)
            transforms.append(affine_transform)

        # Deformable motion
        if coarse_displacements is not None:
            assert(coarse_grid_resolution is not None)
            elastic_transform_coarse = create_bspline_transform(shape=[i, j, k],
                                                                displacements=coarse_displacements,
                                                                grid_resolution=coarse_grid_resolution)
            transforms.append(elastic_transform_coarse)
        if fine_displacements is not None:
            assert(fine_grid_resolution is not None)
            elastic_transform_fine = create_bspline_transform(shape=[i, j, k],
                                                              displacements=fine_displacements,
                                                              grid_resolution=fine_grid_resolution)
            transforms.append(elastic_transform_fine)


        # Create deformation grid by composing transforms
        deformed_grid, jac_det = create_deformation_grid(shape=[i, j, k],
                                                         transforms=transforms)

        if deformed_grid is None:
            return None, None

        ndim, i, j, k = deformed_grid.shape
        deformed_grid = np.transpose(deformed_grid,
                                     (1, 2, 3, 0)) # Dimension at the last

        deformed_grid_torch = np.copy(deformed_grid)
        # See: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/6
        # grid[:, i, j, k, 0] = new_x (k')
        # grid[:, i, j, k, 1] = new_y (j')
        # grid[:, i, j, k, 2] = new_z (i')
        deformed_grid_torch[:, :, :, 0] = deformed_grid[:, :, :, 2]
        deformed_grid_torch[:, :, :, 2] = deformed_grid[:, :, :, 0]


        batch_deformation_grid[batch_idx, ...] = deformed_grid_torch

    #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
    batch_deformation_grid = torch.Tensor(batch_deformation_grid).to(device)
    jac_det = torch.Tensor(jac_det).to(device)
    return batch_deformation_grid, jac_det


def create_batch_deformation_grid_from_pdf(shape,
                                           device='cpu',
                                           dummy=False,
                                           non_rigid=True,
                                           coarse=True,
                                           fine=False,
                                           coarse_grid_resolution=(4, 4, 4),
                                           fine_grid_resolution=(8, 8, 8),
                                           disp_pdf=None,
                                           affine_df=None):

    b, c, i, j, k = shape

    batch_deformation_grid = np.zeros((b, i, j, k, 3),
                                      dtype=np.float32)

    jac_det = np.zeros((b, i, j, k, 3),
                      dtype=np.float32)

    # Loop over batch and generated a unique deformation grid for each image in the batch
    for batch_idx in range(b):
        transforms = []
        if dummy is False:
            if affine_df is not None:
                affine_transform = create_affine_transform_from_df(shape=[i, j, k],
                                                                   ndim=3,
                                                                   param_df=affine_df)
                transforms.append(affine_transform)

            if non_rigid is True:
                if coarse is True:
                    elastic_transform_coarse = create_bspline_transform_from_pdf(shape=[i, j, k],
                                                                                 disp_pdf=disp_pdf,
                                                                                 grid_resolution=coarse_grid_resolution)

                    transforms.append(elastic_transform_coarse)
                if fine is True:
                    elastic_transform_fine = create_bspline_transform_from_pdf(shape=[i, j, k],
                                                                               disp_pdf=disp_pdf,
                                                                               grid_resolution=fine_grid_resolution)
                    transforms.append(elastic_transform_fine)

        # Create deformation grid by composing transforms
        deformed_grid, jac_det = create_deformation_grid(shape=[i, j, k],
                                                         transforms=transforms)

        if deformed_grid is None:
            return None, None

        # Rearrange axes to make the deformation grid torch-friendly
        ndim, i, j, k = deformed_grid.shape
        deformed_grid = np.transpose(deformed_grid,
                                     (1, 2, 3, 0)) # Dimension at the last


        deformed_grid_torch = np.copy(deformed_grid)
        # See: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/6
        # grid[:, i, j, k, 0] = new_x (k')
        # grid[:, i, j, k, 1] = new_y (j')
        # grid[:, i, j, k, 2] = new_z (i')
        deformed_grid_torch[:, :, :, 0] = deformed_grid[:, :, :, 2]
        deformed_grid_torch[:, :, :, 1] = deformed_grid[:, :, :, 1]
        deformed_grid_torch[:, :, :, 2] = deformed_grid[:, :, :, 0]

        batch_deformation_grid[batch_idx, ...] = deformed_grid_torch

    #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
    batch_deformation_grid = torch.Tensor(batch_deformation_grid).to(device)
    jac_det = torch.Tensor(jac_det).to(device)

    return batch_deformation_grid, jac_det


def construct_tps_defromation(p1=None,
                              p2=None,
                              smoothing=0.0,
                              shape=None,
                              gpu_id=-1):

    """

    Function f that fits a thin-plate spline given a set of point correspondences

        s.t. f(x) = x', x \in p1 and x' \in p2

        and f = min_f [\lambda*\int |D^2f|^2 dX] (minimum bending energy)

    Input:
        p1, p2 are 3-D coordinates scaled between [0, 1]
        p1: (np.ndarray) Px3 points from the fixed image
        p2: (np.ndarray) Px3 points from the moving image
        shape: (np.ndarray) Grid dims from which p1 and p2 have been sampled

    Returns:
        transformed_grid: (np.ndarray) Regular grid transformed using tps defined by p1-p2 correspondences

    """

    # Shape: [3, X, Y, Z]
    grid = np.array(np.meshgrid(np.linspace(0, 1, shape[0]),
                                np.linspace(0, 1, shape[1]),
                                np.linspace(0, 1, shape[2]),
                                indexing="ij"),
                    dtype=np.float32)

    ndim, X, Y, Z = grid.shape

    # Reshape the grid so that it's compatible with the __call__ method
    # Shape: [3, X*Y*Z]
    grid = np.reshape(grid, (ndim, X*Y*Z))

    # Shape: [X*Y*Z, 3]
    grid = grid.T


    if gpu_id < 0:
        tps_interpolator = RBFInterpolator(y=p1,
                                           d=p2,
                                           smoothing=smoothing,
                                           kernel='thin_plate_spline',
                                           degree=1)
        # Shape: [X*Y*Z, 3]
        transformed_grid = tps_interpolator(grid)

    else:
        with cp.cuda.Device(gpu_id):
            p1 = cp.asarray(p1)
            p2 = cp.asarray(p2)
            grid = cp.asarray(grid)

            tps_interpolator = GPURBFInterpolator(y=p1,
                                                  d=p2,
                                                  smoothing=smoothing,
                                                  kernel='thin_plate_spline',
                                                  degree=1)

            transformed_grid = tps_interpolator(grid)

        # Move it back to the CPU
        transformed_grid = cp.asnumpy(transformed_grid)

    # Re-shape this grid
    transformed_grid = transformed_grid.T
    transformed_grid = np.reshape(transformed_grid,
                                  (ndim, X, Y, Z))

    return transformed_grid.astype(np.float32)


def transform_grid(transform=None,
                   shape=None):

    """

    Given a transformation, deform the grid (used to resample later)

    """

    # Shape: [3, X, Y, Z]
    grid = np.array(np.meshgrid(np.linspace(0, 1, shape[0]),
                                np.linspace(0, 1, shape[1]),
                                np.linspace(0, 1, shape[2]),
                                indexing="ij"),
                    dtype=np.float32)

    ndim, X, Y, Z = grid.shape

    # Reshape the grid so that it's compatible with the __call__ method
    # Shape: [3, X*Y*Z]
    grid = np.reshape(grid, (ndim, X*Y*Z))

    # Shape: [X*Y*Z, 3]
    grid = grid.T

    transformed_grid = transform(grid)

    transformed_grid = transformed_grid.T
    transformed_grid = np.reshape(transformed_grid,
                                  (ndim, X, Y, Z))

    return transformed_grid.astype(np.float32)


def calculate_jacobian(deformed_grid:np.ndarray=None):
    """

    Compute Jacobian of the deformed grid. Adapted from Koen's gryds code
    See: ~/gryds/gryds/gryds/interpolators/grid.py

    """

    ndim, X, Y, Z = deformed_grid.shape

    # Scale [0,1] ->[0, X/Y/Z] depending on dimension
    scaled_deformed_grid = np.zeros_like(deformed_grid)
    scaled_deformed_grid[0, ...] = X*deformed_grid[0, ...]
    scaled_deformed_grid[1, ...] = Y*deformed_grid[1, ...]
    scaled_deformed_grid[2, ...] = Z*deformed_grid[2, ...]

    jacobian = np.zeros((ndim, ndim, X, Y, Z),
                        dtype=scaled_deformed_grid.dtype)

    for i in range(ndim):
        for j in range(ndim):
            padding = ndim*[(0, 0)]
            padding[j] = (0, 1) # Only pad after since np.diff uses a[i+1]-a[i]
            jacobian[i, j, ...] = np.pad(np.diff(scaled_deformed_grid[i, ...],
                                                 n=1,
                                                 axis=j),
                                         padding,
                                         mode='edge')
    return jacobian

def calculate_jacobian_determinant(deformed_grid:np.ndarray=None):
    """

    Compute determinant of Jacobian

    Input:
        deformed_grid: (np.ndarray, (3, X, Y, Z)) Generated by applying the transformation to a grid
    Returns:
        jac_det: (np.ndarray, (X, Y, Z)) Determinant of the Jacobian for each point on the grid

    """

    jacobian = calculate_jacobian(deformed_grid) # Shape: (3, 3, X, Y, Z)

    jacobian = np.transpose(jacobian, (2, 3, 4, 0, 1))

    jac_det = np.linalg.det(jacobian)

    return jac_det

def resample_image(image:np.ndarray,
                   transformed_coordinates:np.ndarray):

    """

    Function to resample image on a transformed grid.
    Example usage: Given a transformation (defined from fixed -> moving domains), resample the moving image

    """

    # Scale the coordinates [0, 1] -> [0, X/Y/Z]
    scaled_transformed_coordinates = np.zeros_like(transformed_coordinates)
    scaled_transformed_coordinates[0, ...] = image.shape[0]*transformed_coordinates[0, ...]
    scaled_transformed_coordinates[1, ...] = image.shape[1]*transformed_coordinates[1, ...]
    scaled_transformed_coordinates[2, ...] = image.shape[2]*transformed_coordinates[2, ...]

    resampled_image = map_coordinates(input=image,
                                      coordinates=scaled_transformed_coordinates,
                                      order=3,
                                      mode='constant',
                                      cval=np.amin(image))

    return resampled_image

