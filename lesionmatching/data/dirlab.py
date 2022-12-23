"""

While quite impressive and user-friendly, the MONAI data APIs are quite opaque when it comes to meta-data
The SimpleITK/ITK interface is quite simple with spacing, direction, and origin. Therefore, we define a Pytorch
Dataset combining fine-grain control over resampling and meta-data propagation and off-the-shelf implementations
of patch-sampling and intensity/spation transforms


@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from monai.transforms import *
import SimpleITK as sitk
import numpy as np
import random

class DIRLab(Dataset):

    def __init__(self,
                 data_dicts:dict,
                 test:bool=False,
                 data_aug:bool=False):
        """
        Args:
            data_dicts(list) : List of dictionaries. Each dictionary contains is two keys 'image' and 'lung mask'
                               while are populated with paths to the CT image and binary lung mask respectively

        """
        super().__init__()
        self.data_dicts = data_dicts
        self.test = test
        self.data_aug = data_aug


    def resample(self, image, mask):
        """
        Reference: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/popi_utilities_setup.py

        """
        assert(isinstance(image, sitk.Image))
        assert(isinstance(mask, sitk.Image))

        # Get default pixel value
        default_pixel_value = int(np.amin(sitk.GetArrayFromImage(image)).astype(np.float32))

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(round(original_size[0]*(original_spacing[0]/self.new_spacing[0]))),
                    int(round(original_size[1]*(original_spacing[1]/self.new_spacing[1]))),
                    int(round(original_size[2]*(original_spacing[2]/self.new_spacing[2])))]

        resampled_image = sitk.Resample(image,
                                        new_size,
                                        sitk.Transform(),
                                        sitk.sitkLinear,
                                        image.GetOrigin(),
                                        self.new_spacing,
                                        image.GetDirection(),
                                        default_pixel_value,
                                        image.GetPixelID()
                                        )

        resampled_mask = sitk.Resample(mask,
                                       new_size,
                                       sitk.Transform(),
                                       sitk.sitkNearestNeighbor,
                                       mask.GetOrigin(),
                                       self.new_spacing,
                                       mask.GetDirection(),
                                       0,
                                       mask.GetPixelID()
                                    )

        return resampled_image, resampled_mask


    def __len__(self):
        return len(self.data_dicts)

    @staticmethod
    def convert_itk_to_ras_numpy(image):

        assert(isinstance(image, sitk.Image))

        im_np = sitk.GetArrayFromImage(image)

        # Convert to RAS axis ordering : [z, y, x] -> [x, y, z]
        im_np = np.transpose(im_np, (2, 1, 0))

        # Add 'fake' channel axis
        im_np = np.expand_dims(im_np, axis=0)

        return im_np


    def __getitem__(self, idx):

        data_dict = self.data_dicts[idx]
        impath = data_dict['image']
        maskpath = data_dict['lung_mask']

        batch_dict = {}
        batch_dict['patient_id'] = data_dict['patient_id']
        batch_dict['type'] = data_dict['type']

        # Step 1. Load the images (ITK)
        image_itk = sitk.ReadImage(impath)
        mask_itk  = sitk.ReadImage(maskpath)

        # Step 2. Store the original metadata
        batch_dict['metadata'] = {}
        batch_dict['metadata']['spacing'] = mask_itk.GetSpacing()
        batch_dict['metadata']['direction'] = mask_itk.GetDirection()
        batch_dict['metadata']['origin'] = mask_itk.GetOrigin()


        # Step 3. Convert ITK image to numpy array (in RAS axis ordering)
        image_np = self.convert_itk_to_ras_numpy(image_itk)
        mask_np = self.convert_itk_to_ras_numpy(mask_itk)

        # Step 4. Convert numpy ndarrays to torch Tensors
        image_t = torch.from_numpy(image_np)
        mask_t = torch.from_numpy(mask_np)

        # Step 5. Min-max normalization (over full image)
        image_t = ScaleIntensity(minv=0.0,
                                 maxv=1.0)(image_t)

        batch_dict['image'] = image_t
        batch_dict['lung_mask'] = mask_t.float()

        return batch_dict
