""""

Misc. functions

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os



def create_data_dicts(patient_dir_list=None):


    data_dicts = []

    for p_dir in patient_dir_list:
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        for s_dir in scan_dirs:
            data_dict = {}
            data_dict['image'] = []
            for chidx in range(6):
                data_dict['image'].append(os.path.join(s_dir, 'DCE_channel_{}.nii'.format(chidx)))
            data_dict['label'] = os.path.join(s_dir, 'LiverMask.nii')
            data_dicts.append(data_dict)

    return data_dicts
