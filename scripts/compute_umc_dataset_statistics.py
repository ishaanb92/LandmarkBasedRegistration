"""

Script to compute UMC dataset statistics (used to scale intensities)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import numpy as np
import joblib
import SimpleITK as sitk

if __name__ == '__main__':

    train_patients = joblib.load('train_patients_umc.pkl')

    max_mean_image_arr = np.zeros((2*len(train_patients),),
                                   dtype=np.float32)

    min_mean_image_arr = np.zeros((2*len(train_patients),),
                                   dtype=np.float32)

    max_multichannel_arr = np.zeros((2*len(train_patients), 6),
                                     dtype=np.float32)

    min_multichannel_arr = np.zeros((2*len(train_patients), 6),
                                     dtype=np.float32)



    for pidx, p_dir in enumerate(train_patients):
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]

        for sidx, s_dir in enumerate(scan_dirs):
            liver_mask_itk = sitk.ReadImage(os.path.join(s_dir,
                                                         'LiverMask_dilated.nii'))

            liver_mask_np = sitk.GetArrayFromImage(liver_mask_itk)

            # Mean image
            mean_dce_itk = sitk.ReadImage(os.path.join(s_dir,
                                                       'DCE_mean.nii'))

            mean_dce_np = sitk.GetArrayFromImage(mean_dce_itk)

            max_mean_image_arr[len(scan_dirs)*pidx+sidx] = np.amax(mean_dce_np[liver_mask_np==1])
            min_mean_image_arr[len(scan_dirs)*pidx+sidx] = np.amin(mean_dce_np[liver_mask_np==1])


            # Multi-channel
            for chid in range(6):
                ch_dce_itk = sitk.ReadImage(os.path.join(s_dir,
                                                         'DCE_channel_{}.nii'.format(chid)))

                ch_dce_np = sitk.GetArrayFromImage(mean_dce_itk)

                max_multichannel_arr[len(scan_dirs)*pidx+sidx, chid] = np.amax(ch_dce_np[liver_mask_np==1])
                min_multichannel_arr[len(scan_dirs)*pidx+sidx, chid] = np.amin(ch_dce_np[liver_mask_np==1])




    max_mean_image = np.amax(max_mean_image_arr)
    min_mean_image = np.amin(min_mean_image_arr)
    max_multichannel = np.amax(max_multichannel_arr, axis=0) # (6, )
    min_multichannel = np.amin(min_multichannel_arr, axis=0) # (6, )

    dataset_stats = {'mean_max':max_mean_image,
                     'mean_min':min_mean_image,
                     'max_multichannel': max_multichannel,
                     'min_multichannel': min_multichannel}

    joblib.dump(dataset_stats,
                'umc_dataset_stats.pkl')

