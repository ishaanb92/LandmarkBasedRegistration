"""

Script to compare intensity distributions (inside the lung) of the CT images belonging to the 4DCT and COPDGene datasets

The idea is to create 2 sets of histograms for each dataset: inhale and exhale

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import numpy
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DIRLAB_DIR = '/home/ishaan/DIR-Lab/mha'
COPD_DIR = '/home/ishaan/COPDGene/mha'



def create_histograms_for_dataset_and_image_type(dataset='dirlab',
                                                 image_type='inhale',
                                                 bins=15,
                                                 ax=None):

    if dataset == 'dirlab':
        data_dir = DIRLAB_DIR
        if image_type == 'inhale':
            im_str = 'T00'
        elif image_type == 'exhale':
            im_str = 'T50'

    elif dataset == 'copd':
        data_dir = COPD_DIR

        if image_type == 'inhale':
            im_str = 'iBHCT'
        elif image_type == 'exhale':
            im_str = 'eBHCT'


    # Find the patient dirs
    pat_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    # Find min and max (in the lung area) over all patients
    min_value = None
    max_value = None
    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]
        img_itk = sitk.ReadImage(os.path.join(pdir, '{}_{}_iso.mha'.format(pid, im_str)))
        mask_itk = sitk.ReadImage(os.path.join(pdir, 'lung_mask_{}_dl_iso.mha'.format(im_str)))

        img_np = sitk.GetArrayFromImage(img_itk)
        mask_np = sitk.GetArrayFromImage(mask_itk)

        img_min_value = np.amin(img_np[np.where(mask_np==1)])
        img_max_value = np.amax(img_np[np.where(mask_np==1)])

        if min_value is None:
            min_value = img_min_value
        else:
            if img_min_value < min_value:
                min_value = img_min_value

        if max_value is None:
            max_value = img_max_value
        else:
            if img_max_value > max_value:
                max_value = img_max_value

    # Start with the histogram now!
    counts = None
    prev_bins = None
    for pdir in pat_dirs:
        pid = pdir.split(os.sep)[-1]
        img_itk = sitk.ReadImage(os.path.join(pdir, '{}_{}_iso.mha'.format(pid, im_str)))
        mask_itk = sitk.ReadImage(os.path.join(pdir, 'lung_mask_{}_dl_iso.mha'.format(im_str)))

        img_np = sitk.GetArrayFromImage(img_itk)
        mask_np = sitk.GetArrayFromImage(mask_itk)

        img_counts, bins = np.histogram(a=img_np[np.where(mask_np == 1)],
                                        bins=bins,
                                        range=(min_value, max_value))

        # Sanity check to see if bin edges for all images are the same!
        if prev_bins is not None:
            assert(np.array_equal(bins, prev_bins))

        if counts is None:
            counts = img_counts
        else:
            counts += img_counts

        prev_bins = bins

    ax.bar(x=bins[:-1],
           height=counts,
           width=np.diff(bins),
           align='edge',
           fc='skyblue',
           ec='black')

    ax.set_ylabel('Counts')
    ax.set_xlabel('Voxel intensity bins')

    return ax




if __name__ == '__main__':

    fig, axs = plt.subplots(2, 2,
                            figsize=(10, 10))

    axs[0, 0] = create_histograms_for_dataset_and_image_type(dataset='dirlab',
                                                             image_type='inhale',
                                                             ax=axs[0, 0])

    axs[0, 1] = create_histograms_for_dataset_and_image_type(dataset='dirlab',
                                                             image_type='exhale',
                                                             ax=axs[0, 1])

    axs[1, 0] = create_histograms_for_dataset_and_image_type(dataset='copd',
                                                             image_type='inhale',
                                                             ax=axs[1, 0])

    axs[1, 1] = create_histograms_for_dataset_and_image_type(dataset='copd',
                                                             image_type='exhale',
                                                             ax=axs[1, 1])

    axs[0, 0].set_title('4DCT-Inhale')
    axs[0, 1].set_title('4DCT-Exhale')
    axs[1, 0].set_title('COPD-Inhale')
    axs[1, 1].set_title('COPD-Exhale')

    fig.savefig(os.path.join('/home/ishaan/ct_statistics.png'),
                bbox_inches='tight')
