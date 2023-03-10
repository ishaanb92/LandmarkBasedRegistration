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
from lesionmatching.util_scripts.image_utils import convert_itk_to_ras_numpy

class DIRLab(Dataset):

    def __init__(self,
                 data_dicts:dict,
                 test:bool=False,
                 patch_size=(128, 128, 64),
                 seed=1234):
        """
        Args:
            data_dicts(list) : List of dictionaries. Each dictionary contains is two keys 'image' and 'lung mask'
                               while are populated with paths to the CT image and binary lung mask respectively

        """
        super().__init__()
        self.data_dicts = data_dicts
        self.test = test
        self.patch_size = patch_size
        self.crop_transform = RandCropByPosNegLabel(spatial_size=self.patch_size,
                                                    pos=1.0,
                                                    neg=0.0,
                                                    num_samples=10)
        self.crop_transform.set_random_state(seed)

    def __len__(self):
        return len(self.data_dicts)

    @staticmethod
    def scale_image_intensities(image, mask):

        lung_max = np.max(image[np.where(mask==1)])
        lung_min = np.min(image[np.where(mask==1)])

        intensity_range = lung_max-lung_min

        # Avoid NaN's by restricting intensity range
        image = np.where(image>lung_max, lung_max, image)
        image = np.where(image<lung_min, lung_min, image)

        image = np.divide(np.subtract(image, lung_min), (lung_max-lung_min))

        if np.isnan(np.sum(image)) is True:
            raise RuntimeError('NaN encountered while rescaling voxel intensities')

        return image

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
        image_np = convert_itk_to_ras_numpy(image_itk)
        mask_np = convert_itk_to_ras_numpy(mask_itk)

        assert(np.amax(mask_np) == 1)

        # Step 4. Min-max normalization (only over the lung)
        image_np = self.scale_image_intensities(image=image_np,
                                                mask=mask_np)

        # Add "fake" channel axis
        image_np = np.expand_dims(image_np, axis=0)
        mask_np = np.expand_dims(mask_np, axis=0)

        # Step 5. Convert numpy ndarrays to torch Tensors
        image_t = torch.from_numpy(image_np)
        mask_t = torch.from_numpy(mask_np)


        # Step 6. Use the Gamma correction (Eppenhof and Pluim, TMI, (2019))
        gamma_factor = np.random.uniform(low=0.5, high=1.5)
        image_t = torch.pow(image_t, gamma_factor)

        # Step 7. Sample patch from image
        if self.test is False:
            image_and_mask_cat_t = torch.cat([image_t, mask_t], dim=0)

            image_and_mask_patch_list = self.crop_transform(img=image_and_mask_cat_t,
                                                            label=mask_t)

            # Shuffle list to ensure different patches are selected!
            random.shuffle(image_and_mask_patch_list)

            image_t = torch.unsqueeze(image_and_mask_patch_list[0][0, ...],
                                      dim=0)

            mask_t = torch.unsqueeze(image_and_mask_patch_list[0][1, ...],
                                     dim=0)

        batch_dict['image'] = image_t
        batch_dict['lung_mask'] = mask_t.float()

        return batch_dict



class DIRLabPaired(Dataset):

    def __init__(self,
                 data_dicts:dict=None,
                 rescaling_stats:dict=None):

        super().__init__()
        self.data_dicts = data_dicts
        self.rescaling_stats = rescaling_stats

    def __len__(self):
        return len(self.data_dicts)


    @staticmethod
    def scale_image_intensities(image, mask, lung_max=None, lung_min=None):

        if lung_max is None and lung_min is None:
            lung_max = np.amax(image[np.where(mask==1)])
            lung_min = np.amin(image[np.where(mask==1)])

        image = (image - lung_min)/(lung_max-lung_min)

        return image.astype(np.float32)

    def preprocess_image_and_mask(self,
                                  data_dict:dict,
                                  image_type:str='fixed'):


        impath = data_dict['{}_image'.format(image_type)]
        maskpath = data_dict['{}_lung_mask'.format(image_type)]

        batch_dict = {}

        # Step 1. Load the images (ITK)
        image_itk = sitk.ReadImage(impath)
        mask_itk  = sitk.ReadImage(maskpath)

        # Step 2. Store the original metadata
        batch_dict['{}_metadata'.format(image_type)] = {}
        batch_dict['{}_metadata'.format(image_type)]['spacing'] = mask_itk.GetSpacing()
        batch_dict['{}_metadata'.format(image_type)]['direction'] = mask_itk.GetDirection()
        batch_dict['{}_metadata'.format(image_type)]['origin'] = mask_itk.GetOrigin()

        # Step 3. Convert ITK image to numpy array (in RAS axis ordering)
        image_np = convert_itk_to_ras_numpy(image_itk)
        mask_np = convert_itk_to_ras_numpy(mask_itk)
        assert(np.amax(mask_np) == 1)

        # Step 4. Min-max normalization (only over the lung)
        if self.rescaling_stats is None:
            image_np = self.scale_image_intensities(image=image_np,
                                                    mask=mask_np)
        else:
            image_np = self.scale_image_intensities(image=image_np,
                                                    mask=mask_np,
                                                    lung_max=self.rescaling_stats['{}_image_max'.format(image_type)],
                                                    lung_min=self.rescaling_stats['{}_image_min'.format(image_type)])

        # Add "fake" channel axis
        image_np = np.expand_dims(image_np, axis=0)
        mask_np = np.expand_dims(mask_np, axis=0)

        # Step 5. Convert numpy ndarrays to torch Tensors
        image_t = torch.from_numpy(image_np)
        mask_t = torch.from_numpy(mask_np)

        batch_dict['{}_image'.format(image_type)] = image_t
        batch_dict['{}_lung_mask'.format(image_type)] = mask_t.float()

        return batch_dict


    def __getitem__(self, idx):

        data_dict = self.data_dicts[idx]

        fixed_image_dict = self.preprocess_image_and_mask(data_dict=data_dict,
                                                          image_type='fixed')

        moving_image_dict = self.preprocess_image_and_mask(data_dict=data_dict,
                                                           image_type='moving')

        # Merge the two dictionaries
        batch_dict = {**fixed_image_dict, **moving_image_dict}
        batch_dict['patient_id'] = data_dict['patient_id']

        return batch_dict
