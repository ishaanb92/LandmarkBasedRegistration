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

def create_displacement_grid_affine(image,
                                    angles=[0.0, 0.0, 0.0],
                                    scaling=None,
                                    shear_matrix=None,
                                    translation=None):

    image_grid = gryds.Grid(image.shape)

    print(image_grid.grid.shape)

    aff_transform = gryds.AffineTransformation(ndim=3,
                                                center=[0.5, 0.5, 0.5],
                                                angles=angles,
                                                scaling=scaling,
                                                shear_matrix=shear_matrix,
                                                translation=translation)

    deformed_grid = image_grid.transform(aff_transform)

    print(deformed_grid.grid.shape)

    return deformed_grid.grid




# Test deformations
if __name__ == '__main__':

    # Load all patient paths
    train_patients = joblib.load('../train_patients.pkl')
    train_dict = create_data_dicts([train_patients[0]],
                                   n_channels=1,
                                   channel_id=3)

    data_loader, _ = create_dataloader(data_dicts=train_dict,
                                      train=False,
                                      batch_size=1)

    for batch_data in data_loader:
        images = batch_data['image']
        print(images.shape)

        b, c, h, w, d = images.shape
        batch_deformation_grid = np.zeros((b, h, w, d, 3),
                                          dtype=np.float32)
        for batch_idx in range(images.shape[0]):
            image = images[batch_idx]
            deformed_grid = create_displacement_grid_affine(image=torch.squeeze(image, dim=0),
                                                            angles=[0.0, 0.0, 30])
            batch_deformation_grid[batch_idx, ...] = deformed_grid

        # Deform the whole batch
        deformed_images = F.grid_sample(input=images,
                                        grid=batch_deformation_grid)
