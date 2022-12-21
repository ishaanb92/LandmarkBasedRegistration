"""

Create lung masks that can be used to focus landmark candidate selection

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import SimpleITK as sitk
import os
import scipy.ndimage as ndi


DATA_DIR = '/home/ishaan/DIR-Lab/mha'

def create_lung_mask(image_dir, imagename):

    imagefname = os.path.join(image_dir,
                              imagename)

    suffix = imagename.split('_')[-1].split('.')[0]

    img_itk = sitk.ReadImage(imagefname)

    img_np = sitk.GetArrayFromImage(img_itk)

    # Threshold the image
    lung_and_patient_ext = np.where(img_np < -250, 1, 0).astype(np.uint8)
    lung_and_surround = np.where(img_np < -900, 0, 1).astype(np.uint8)

    # Element-wise multiplication
    lung_only = lung_and_patient_ext*lung_and_surround

    # Find connected components
    label_im, nb_labels = ndi.label(lung_only)

    comp_sizes = ndi.sum_labels(lung_only,
                                labels=label_im,
                                index=range(nb_labels+1))


    sort_labels = np.argsort(comp_sizes)

    top_two_labels = sort_labels[-2:]

    lung_mask = np.where(np.logical_or((label_im == top_two_labels[0]),
                                       (label_im == top_two_labels[1])), 1, 0).astype(np.uint8)


    # Set the bottom of the mask (patient exterior) to 0
    lung_mask[:, 240:, :] = 0


    # Close the holes
    lung_mask = ndi.binary_fill_holes(lung_mask,
                                      structure=ndi.generate_binary_structure(rank=3,
                                                                              connectivity=8)).astype(np.uint8)

    # Convert lung mask into an ITK image
    lung_mask_itk = sitk.GetImageFromArray(lung_mask)

    lung_mask_itk.SetOrigin(img_itk.GetOrigin())
    lung_mask_itk.SetSpacing(img_itk.GetSpacing())
    lung_mask_itk.SetDirection(img_itk.GetDirection())

    sitk.WriteImage(lung_mask_itk,
                    os.path.join(image_dir, 'lung_mask_{}.mha'.format(suffix)))

if __name__ == '__main__':

    case_dirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

    for cdir in case_dirs:
        img_files = [f.name for f in os.scandir(cdir) if 'case' in f.name]
        for img_file in img_files:
            print('Creating mask for {}'.format(img_file))
            create_lung_mask(image_dir=cdir,
                             imagename=img_file)






