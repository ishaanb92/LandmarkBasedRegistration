"""

Apply z-score normalization to each image in the UMC dataset. Use the liver mask to restrict computation of mean/std to the liver region

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy as np
import SimpleITK as sitk
import joblib
from lesionmatching.util_scripts.image_utils import *
from lesionmatching.util_scripts.utils import *

if __name__ == '__main__':

    patients = joblib.load('train_patients_umc.pkl')
    patients.extend(joblib.load('val_patients_umc.pkl'))
    patients.extend(joblib.load('test_patients_umc.pkl'))


    for pidx, p_dir in enumerate(patients):
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]

        for sidx, s_dir in enumerate(scan_dirs):
            liver_mask_itk = sitk.ReadImage(os.path.join(s_dir,
                                                         'LiverMask_dilated.nii'))

            liver_mask_np = sitk.GetArrayFromImage(liver_mask_itk)

            vessel_mask_path = os.path.join(s_dir,
                                            'vessel_mask.nii')

            if os.path.exists(vessel_mask_path) is True:
                vessel_mask_itk = sitk.ReadImage(vessel_mask_path)
                vessel_mask_np = sitk.GetArrayFromImage(vessel_mask_itk)
            else:
                vessel_mask_itk = None
                vessel_mask_np = None

            # Mean image
            mean_dce_itk = sitk.ReadImage(os.path.join(s_dir,
                                                       'DCE_mean.nii'))

            mean_dce_np = sitk.GetArrayFromImage(mean_dce_itk)

            # Zero-pad all images in the slice axis to 128 (for patching)
            z, y, x = liver_mask_np.shape
            pad_length = 128 - z
            if pad_length > 0:
                liver_mask_np = np.pad(liver_mask_np,
                                       pad_width = ((0, pad_length),
                                                    (0, 0),
                                                    (0, 0)))
                mean_dce_np = np.pad(mean_dce_np,
                                     pad_width=((0, pad_length),
                                                (0, 0),
                                                (0, 0)))

                if vessel_mask_np is not None:
                    vessel_mask_np = np.pad(vessel_mask_np,
                                           pad_width = ((0, pad_length),
                                                        (0, 0),
                                                        (0, 0)))

            mean_liver = np.mean(mean_dce_np[liver_mask_np==1])
            std_liver = np.mean(mean_dce_np[liver_mask_np==1])

            mean_dce_np_z_score = (mean_dce_np-mean_liver)/std_liver

            # Save this image
            mean_dce_np_z_score_itk = sitk.GetImageFromArray(mean_dce_np_z_score)
            mean_dce_np_z_score_itk.SetSpacing(liver_mask_itk.GetSpacing())
            mean_dce_np_z_score_itk.SetOrigin(liver_mask_itk.GetOrigin())
            mean_dce_np_z_score_itk.SetDirection(liver_mask_itk.GetDirection())
            sitk.WriteImage(mean_dce_np_z_score_itk,
                            os.path.join(s_dir, 'DCE_mean_zscore.nii'))

            liver_mask_padded_itk = sitk.GetImageFromArray(liver_mask_np)
            liver_mask_padded_itk.SetSpacing(liver_mask_itk.GetSpacing())
            liver_mask_padded_itk.SetOrigin(liver_mask_itk.GetOrigin())
            liver_mask_padded_itk.SetDirection(liver_mask_itk.GetDirection())
            sitk.WriteImage(liver_mask_padded_itk,
                            os.path.join(s_dir, 'Liver_mask_padded.nii'))

            if vessel_mask_np is not None:
                vessel_mask_padded_itk = sitk.GetImageFromArray(vessel_mask_np)
                vessel_mask_padded_itk.SetSpacing(liver_mask_itk.GetSpacing())
                vessel_mask_padded_itk.SetOrigin(liver_mask_itk.GetOrigin())
                vessel_mask_padded_itk.SetDirection(liver_mask_itk.GetDirection())
                sitk.WriteImage(vessel_mask_padded_itk,
                                os.path.join(s_dir, 'vessel_mask_padded.nii'))

            # Multi-channel
            for chid in range(6):
                ch_dce_itk = sitk.ReadImage(os.path.join(s_dir,
                                                         'DCE_channel_{}.nii'.format(chid)))

                ch_dce_np = sitk.GetArrayFromImage(ch_dce_itk)

                if pad_length > 0:
                    ch_dce_np = np.pad(ch_dce_np,
                                       pad_width=((0, pad_length),
                                                   (0, 0),
                                                   (0, 0)))

                mean_liver = np.mean(ch_dce_np[liver_mask_np==1])
                std_liver = np.std(ch_dce_np[liver_mask_np==1])

                ch_dce_np_z_score = (ch_dce_np-mean_liver)/std_liver

                ch_dce_np_z_score_itk = sitk.GetImageFromArray(ch_dce_np_z_score)
                ch_dce_np_z_score_itk.SetSpacing(liver_mask_itk.GetSpacing())
                ch_dce_np_z_score_itk.SetOrigin(liver_mask_itk.GetOrigin())
                ch_dce_np_z_score_itk.SetDirection(liver_mask_itk.GetDirection())
                sitk.WriteImage(ch_dce_np_z_score_itk,
                                os.path.join(s_dir, 'DCE_channel_{}_zscore.nii'.format(chid)))

