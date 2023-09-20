"""

Script to test patching and (more importantly) re-assembling of the image

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

from lesionmatching.data.datapipeline import *
import os
import joblib
from lesionmatching.data.dirlab import *
from lesionmatching.util_scripts.image_utils import save_ras_as_itk
from lesionmatching.util_scripts.utils import detensorize_metadata
import monai

COPD_DIR = '/home/ishaan/COPDGene/mha'

if __name__ == '__main__':
    patients = [f.path for f in os.scandir(COPD_DIR) if f.is_dir()]
    rescaling_stats = joblib.load(os.path.join(COPD_DIR,
                                               'rescaling_stats.pkl'))


    data_dicts = create_data_dicts_dir_lab_paired([patients[0]],
                                                  affine_reg_dir=None,
                                                  dataset='copd',
                                                  soft_masking=True)

    data_loader = create_dataloader_dir_lab_paired(data_dicts=data_dicts,
                                                   batch_size=1,
                                                   num_workers=4,
                                                   rescaling_stats=rescaling_stats,
                                                   patch_size=(128, 128, 96),
                                                   overlap=0.0)

    for b_id, batch_data_list in enumerate(data_loader):
        print('Processing batch {}'.format(b_id+1))

        if isinstance(batch_data_list, dict):
            batch_data_list = [batch_data_list]

        for sid, batch_data in enumerate(batch_data_list):
            images = batch_data['fixed_image']

            metadata_list = detensorize_metadata(metadata=batch_data['fixed_metadata'],
                                                 batchsz=images.shape[0])

            print('Images shape: {}'.format(images.shape))
            print('Number of patches: {}'.format(len(batch_data['patch_dict']['origins'])))

            images_np = torch.squeeze(images[0], dim=0).numpy()
            empty_image = np.zeros_like(images_np)

            for patch_origin, patch_end in zip(batch_data['patch_dict']['origins'], batch_data['patch_dict']['ends']):
                empty_image[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]] = \
                images_np[patch_origin[0]:patch_end[0], patch_origin[1]:patch_end[1], patch_origin[2]:patch_end[2]]

            save_ras_as_itk(img=empty_image,
                            metadata=metadata_list[b_id],
                            fname='patch_assemble.mha')



