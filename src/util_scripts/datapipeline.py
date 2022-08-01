"""

Construct a data pipeline using the MONAI API

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from monai.transforms import *
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import joblib
from helper_functions import create_data_dicts

def create_dataloader_lesion_matching(data_dicts=None, train=True, batch_size=4):

    if train is True:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              # Make sure image is channel first
                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              # Isotropic spacing
                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              # Extract 128x128x48 3-D patches
                              RandSpatialCropd(keys=["image", "label"],
                                               roi_size=[128, 128, 48],
                                               random_size=False),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,


                              EnsureTyped(keys=["image", "label"])
                              ])

    else:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              EnsureTyped(keys=["image", "label"])
                              ])

    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=4)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=4)

    if train is True:
        return loader
    else:
        return loader, transforms


def create_dataloader_liver_seg(data_dicts=None, train=True, batch_size=4):

    if train is True:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              # Make sure image is channel first
                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              # Isotropic spacing
                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              # Extract 128x128x48 3-D patches
                              RandSpatialCropd(keys=["image", "label"],
                                               roi_size=[128, 128, 48],
                                               random_size=False),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),

                              RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

                              EnsureTyped(keys=["image", "label"])
                              ])

    else:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              EnsureTyped(keys=["image", "label"])
                              ])

    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=4)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=4)

    if train is True:
        return loader
    else:
        return loader, transforms

def debug_dataloader_liver_seg(data_dicts=None, batch_size=4):
    """
    Create a bare-bones transform pipeline to flush out bugs in the pipeline

    """

    transforms = Compose([LoadImaged(keys=["image", "label"]),

                          # Add fake channel to the label
                          AddChanneld(keys=["label"]),

                          EnsureChannelFirstd(keys=["image"]),

                          Orientationd(keys=["image", "label"], axcodes="RAS"),

                          Spacingd(keys=["image", "label"],
                                   pixdim=(1.543, 1.543, 1.543),
                                   mode=("bilinear", "nearest")),

                          NormalizeIntensityd(keys=["image"],
                                              nonzero=True,
                                              channel_wise=True),

                          RandSpatialCropd(keys=["image", "label"],
                                           roi_size=[128, 128, 48],
                                           random_size=False),

                          EnsureTyped(keys=["image", "label"])
                          ])


    print('Instanced dataset class')

    ds = Dataset(data=data_dicts,
                 transform=transforms)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4)

    print('Created data loader')
    return loader

if __name__ == '__main__':

    train_patients = joblib.load('train_patients.pkl')
    train_dicts = create_data_dicts(train_patients)

    loader = debug_dataloader(train_dicts, batch_size=4)

    for batch_data in loader:
        images = batch_data['image']
        print(images.shape)
