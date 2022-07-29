""""

Misc. functions

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os

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


def create_data_dicts_lesion_matching(patient_dir_list=None):

    data_dicts = []

    for p_dir in patient_dir_list:
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            s_id = s_dir.split(os.sep)[-1]
            data_dict = {}
            data_dict['image'] = os.path.join(s_dir, 'DCE_vessel_image.nii')
            data_dict['liver_mask'] = os.path.join(s_dir, 'LiverMask.nii')
            data_dict['vessel_mask'] = os.path.join(s_dir, 'vessel_mask.nii')
            data_dict['patient_id'] = p_id
            data_dict['scan_id'] = s_id
            data_dict.append(data_dict)

    return data_dicts
