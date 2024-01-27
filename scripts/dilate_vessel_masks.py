"""

Script to dilate vessel masks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import generate_binary_structure, binary_dilation
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    pat_dirs = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    for pdir in pat_dirs:
        scan_dirs = [f.path for f in os.scandir(pdir) if f.is_dir()]
        for sdir in scan_dirs:
            try:
                vessel_mask_itk = sitk.ReadImage(os.path.join(sdir,
                                                              'vessel_mask.nii'))
            except:
                print('Vessel mask not found for patient {}/{}'.format(pdir.split(os.sep)[-1],
                                                                       sdir.split(os.sep)[-1]))
                continue

            vessel_mask_np = sitk.GetArrayFromImage(vessel_mask_itk)

            vessel_mask_np = vessel_mask_np.astype(np.uint8)

            # Generate binary structure
            #structure = generate_binary_structure(rank=3,
            #                                      connectivity=3)

            structure = np.ones((5, 5, 5),
                                dtype=np.uint8)

            # Dilation with a full 3x3 structure matrix
            vessel_mask_dilated_np = binary_dilation(input=vessel_mask_np,
                                                     structure=structure,
                                                     iterations=1).astype(vessel_mask_np.dtype)

            vessel_mask_dilated_itk = sitk.GetImageFromArray(vessel_mask_dilated_np)
            vessel_mask_dilated_itk.SetOrigin(vessel_mask_itk.GetOrigin())
            vessel_mask_dilated_itk.SetDirection(vessel_mask_itk.GetDirection())
            vessel_mask_dilated_itk.SetSpacing(vessel_mask_itk.GetSpacing())

            sitk.WriteImage(vessel_mask_dilated_itk,
                            os.path.join(sdir,
                                         'vessel_mask_dilated.nii'))


