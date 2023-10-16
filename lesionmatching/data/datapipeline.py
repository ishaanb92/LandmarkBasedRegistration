"""

Construct a data pipeline using the MONAI API

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from monai.transforms import *
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import joblib
import torch
from monai.transforms import StdShiftIntensity, ScaleIntensityRange
from monai.data.utils import no_collation
import numpy as np
import nibabel as nib
from lesionmatching.data.dirlab import *
from lesionmatching.util_scripts.utils import create_datetime_object_from_str

def create_data_dicts_liver_seg(patient_dir_list=None, n_channels=6, channel_id=3):


    data_dicts = []

    for p_dir in patient_dir_list:
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            s_id = s_dir.split(os.sep)[-1]
            data_dict = {}
            if n_channels > 1:
                data_dict['image'] = []
                for chidx in range(n_channels):
                    data_dict['image'].append(os.path.join(s_dir, 'DCE_channel_{}.nii'.format(chidx)))
            else:
                data_dict['image'] = os.path.join(s_dir, 'DCE_channel_{}.nii'.format(channel_id))

            data_dict['label'] = os.path.join(s_dir, 'LiverMask.nii')
            data_dict['patient_id'] = p_id
            data_dict['scan_id'] = s_id
            data_dicts.append(data_dict)

    return data_dicts


# Data dicts for synthetic transforms (eg: training/evaluation w.r.t repeatability)
def create_data_dicts_lesion_matching(patient_dir_list=None,
                                      multichannel=False):

    data_dicts = []

    for p_dir in patient_dir_list:
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            s_id = s_dir.split(os.sep)[-1]
            data_dict = {}

            if multichannel is False:
                data_dict['image'] = os.path.join(s_dir, 'DCE_mean_zscore.nii')
            else:
                data_dict['image'] = []
                for chidx in range(6):
                    data_dict['image'].append(os.path.join(s_dir, 'DCE_channel_{}_zscore.nii'.format(chidx)))

            data_dict['liver_mask'] = os.path.join(s_dir, 'Liver_mask_padded.nii')
            data_dict['vessel_mask'] = os.path.join(s_dir, 'vessel_mask_padded.nii')

            if os.path.exists(os.path.join(s_dir, 'vessel_mask.nii')) is False:
                print('Vessel mask does not exist for Patient {}, scan-ID : {}'.format(p_id, s_dir))
                data_dict['vessel_mask'] = os.path.join(s_dir, 'Liver_mask_padded.nii')

            data_dict['patient_id'] = p_id
            data_dict['scan_id'] = s_id
            data_dicts.append(data_dict)

    return data_dicts

# Data dicts for "real" paired data
def create_data_dicts_lesion_matching_inference(patient_dir_list=None,
                                                multichannel=False):

    data_dicts = []

    for p_dir in patient_dir_list:
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]

        # Figure out which idx is baseline and which is follow-up
        s_id_0 = scan_dirs[0].split(os.sep)[-1]
        s_id_1 = scan_dirs[1].split(os.sep)[-1]
        timestamp_0 = create_datetime_object_from_str(s_id_0)
        timestamp_1 = create_datetime_object_from_str(s_id_1)
        if timestamp_0 > timestamp_1: # Scan0 occurs after Scan1
            baseline_idx = 1
            followup_idx = 0
        else:
            baseline_idx=0
            followup_idx=1

        data_dict = {}
        data_dict['patient_id'] = p_id
        for idx, s_dir in enumerate(scan_dirs):
            s_id = s_dir.split(os.sep)[-1]
            if idx == baseline_idx:
                suffix = 'baseline'
            elif idx ==  followup_idx:
                suffix = 'followup'
            else:
                raise ValueError('Scan dir idx {} is not 0 or 1. Check'.format(idx))

            if multichannel is True:
                data_dict['image_{}'.format(suffix)] = []
                for chidx in range(6):
                    data_dict['image_{}'.format(suffix)].append(os.path.join(s_dir,
                                                                             'DCE_channel_{}_zscore.nii'.format(chidx)))
            else:
                data_dict['image_{}'.format(suffix)] = os.path.join(s_dir,
                                                                    'DCE_mean_zscore.nii')

            data_dict['liver_mask_{}'.format(suffix)] = os.path.join(s_dir, 'LiverMask.nii')
            data_dict['vessel_mask_{}'.format(suffix)] = os.path.join(s_dir, 'vessel_mask.nii')

            if os.path.exists(os.path.join(s_dir, 'vessel_mask.nii')) is False:
                print('Vessel mask does not exist for Patient {}, scan-ID : {}'.format(p_id, s_dir))
                data_dict['vessel_mask_{}'.format(suffix)] = os.path.join(s_dir, 'LiverMask.nii')
            data_dict['scan_id_{}'.format(suffix)] = s_id

            # FIXME: Ugly hack but I can't work with MONAI metadata
            liver_mask = sitk.ReadImage(os.path.join(s_dir, 'LiverMask.nii'))
            itk_metadata_dict = {'spacing':liver_mask.GetSpacing(),
                                 'origin': liver_mask.GetOrigin(),
                                 'direction':liver_mask.GetDirection()}

            data_dict['itk_metadata_dict_{}'.format(suffix)] = itk_metadata_dict

        data_dicts.append(data_dict)

    return data_dicts

def create_data_dicts_dir_lab(patient_dir_list=None, dataset='dirlab'):

    data_dicts = []

    if dataset == 'dirlab':
        im_types = ['T00', 'T50']
    elif dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']

    for p_dir in patient_dir_list:
        im_str = p_dir.split(os.sep)[-1]
        for im_type in im_types:
            data_dict = {}
            data_dict['patient_id'] = im_str
            data_dict['type'] = im_type # Inhale or exhale
            data_dict['image'] = os.path.join(p_dir, '{}_{}_iso.mha'.format(im_str, im_type))
            data_dict['lung_mask'] = os.path.join(p_dir, 'lung_mask_{}_dl_iso.mha'.format(im_type))
            data_dicts.append(data_dict)

    return data_dicts

def create_data_dicts_dir_lab_paired(patient_dir_list=None,
                                     affine_reg_dir = None,
                                     dataset='dirlab',
                                     soft_masking=False):
    data_dicts = []

    if dataset == 'dirlab':
        im_types = ['T00', 'T50']
    elif dataset == 'copd':
        im_types = ['iBHCT', 'eBHCT']

    for p_dir in patient_dir_list:
        im_str = p_dir.split(os.sep)[-1]
        data_dict = {}
        data_dict['fixed_image'] = os.path.join(p_dir, '{}_{}_iso.mha'.format(im_str, im_types[0]))

        if soft_masking is False:
            data_dict['fixed_lung_mask'] = os.path.join(p_dir, 'lung_mask_{}_dl_iso.mha'.format(im_types[0]))
        else:
            data_dict['fixed_lung_mask'] = None

        if affine_reg_dir is None:
            data_dict['moving_image'] = os.path.join(p_dir, '{}_{}_iso.mha'.format(im_str, im_types[1]))
        else:
            data_dict['moving_image'] = os.path.join(affine_reg_dir, im_str, 'result.0.mha')

        if soft_masking is False:
            if affine_reg_dir is None:
                data_dict['moving_lung_mask'] = os.path.join(p_dir, 'lung_mask_{}_dl_iso.mha'.format(im_types[1]))
            else:
                data_dict['moving_lung_mask'] = os.path.join(affine_reg_dir,
                                                             im_str,
                                                             'moving_lung_mask_affine',
                                                             'result.mha')
        else:
            data_dict['moving_lung_mask'] = None


        data_dict['patient_id'] = im_str
        data_dicts.append(data_dict)

    return data_dicts


def create_dataloader_dir_lab(data_dicts=None,
                              test=False,
                              batch_size=4,
                              num_workers=4,
                              patch_size=(128, 128, 128),
                              seed=1234):

    ds = DIRLab(data_dicts=data_dicts,
                test=test,
                patch_size=patch_size,
                seed=seed)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=not(test),
                        num_workers=num_workers)

    return loader


def create_dataloader_dir_lab_paired(data_dicts:dict=None,
                                     batch_size:int=1,
                                     num_workers:int=4,
                                     rescaling_stats:dict=None,
                                     patch_size=(128, 128, 96),
                                     overlap=0.0):


    ds = DIRLabPairedPatchWise(data_dicts=data_dicts,
                               rescaling_stats=rescaling_stats,
                               patch_size=patch_size,
                               overlap=overlap)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)
    return loader



def create_dataloader_lesion_matching(data_dicts=None,
                                      train=True,
                                      batch_size=4,
                                      num_workers=4,
                                      patch_size=(128, 128, 64),
                                      seed=1234,
                                      num_samples=1):

    if train is True:
        transforms = Compose([LoadImaged(keys=["image", "liver_mask", "vessel_mask"]),

                              # Add fake channel to the liver_mask
                              EnsureChannelFirstd(keys=["image", "liver_mask", "vessel_mask"]),

                              Orientationd(keys=["image", "liver_mask", "vessel_mask"], axcodes="RAS"),

                              # Extract 128x128x64 3-D patches
                              RandCropByPosNegLabeld(keys=["image", "liver_mask", "vessel_mask"],
                                                     label_key="liver_mask",
                                                     spatial_size=patch_size,
                                                     pos=1.0,
                                                     neg=0.0,
                                                     num_samples=num_samples),

                              EnsureTyped(keys=["image", "liver_mask", "vessel_mask"])
                              ])

    else:
        transforms = Compose([LoadImaged(keys=["image", "liver_mask", "vessel_mask"]),

                              # Add fake channel to the liver_mask
                              EnsureChannelFirstd(keys=["image", "liver_mask", "vessel_mask"]),

                              Orientationd(keys=["image", "liver_mask", "vessel_mask"], axcodes="RAS"),

                              EnsureTyped(keys=["image", "liver_mask", "vessel_mask"])
                              ])

    # Set (local) random state to ensure reproducibility
    transforms = transforms.set_random_state(seed=seed)

    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=num_workers)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=num_workers)

    return loader, transforms


# With "real" paired data -- No resampling
def create_dataloader_lesion_matching_inference(data_dicts=None, batch_size=4, num_workers=4):

    transforms = Compose([LoadImaged(keys=["image_baseline",  "liver_mask_baseline", "vessel_mask_baseline",
                                           "image_followup",  "liver_mask_followup", "vessel_mask_followup"],
                                     reader="ITKReader"),

                          # Add fake channel to the liver_mask
                          EnsureChannelFirstd(keys=["image_baseline", "liver_mask_baseline", "vessel_mask_baseline",
                                            "image_followup",  "liver_mask_followup", "vessel_mask_followup"]),

                          Orientationd(keys=["image_baseline", "liver_mask_baseline", "vessel_mask_baseline",
                                             "image_followup",  "liver_mask_followup", "vessel_mask_followup"], axcodes="RAS"),

                          EnsureTyped(keys=["image_baseline", "liver_mask_baseline", "vessel_mask_baseline",
                                            "image_followup", "liver_mask_followup", "vessel_mask_followup"])
                          ])


    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=num_workers)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

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


def shift_intensity(images):

    assert(isinstance(images, torch.Tensor))

    factor = np.random.uniform(low=0.6, high=1.0)
    images_shifted = StdShiftIntensity(factor=factor)(images)
    return images_shifted


def create_nifti_header(metadata_dict):

    header = nib.nifti1.Nifti1Header()

    for key, value in metadata_dict.items():
        if "affine" in key:
            continue

        try:
            header[key] = value
        except ValueError:
            continue

    return header

def create_nibabel_image(image_array, affine, metadata_dict):

    assert(isinstance(image_array, np.ndarray))
    assert(isinstance(metadata_dict, dict))

    # Create Nifti header
    header = create_nifti_header(metadata_dict)

    # Create Nifti image
    nib_image = nib.nifti1.Nifti1Image(dataobj=image_array,
                                       affine=affine,
                                       header=header)
    return nib_image

def save_nib_image(img, filename):
    nib.save(img, filename)


def maybe_convert_tensor_to_array(arr):

    if isinstance(arr, torch.Tensor):
        if arr.device != torch.device('cpu'):
            arr = arr.cpu()

        arr = arr.numpy()

    return arr

def write_image_to_file(image_array,
                        affine,
                        metadata_dict,
                        filename):

    image_array = maybe_convert_tensor_to_array(image_array)

    image_nib = create_nibabel_image(image_array,
                                     affine,
                                     metadata_dict)

    save_nib_image(image_nib,
                   filename)

if __name__ == '__main__':

    train_patients = joblib.load('train_patients.pkl')
    train_dicts = create_data_dicts(train_patients)

    loader = debug_dataloader(train_dicts, batch_size=4)

    for batch_data in loader:
        images = batch_data['image']
        print(images.shape)
