"""

Script to evaluate liver segmentation using 3-D UNet

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
import sys
import os
sys.path.append(os.path.join(os.getcwd(),  'src', 'util_scripts'))
from datapipeline import *
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from utils.utils import *


def test(args):

    # Set device
    device = torch.device('cuda:{}'.format(args.gpu_id))


    # Define model

    model = UNet(spatial_dims=3,
                 in_channels=6,
                 out_channels=1,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=1)

    # Load model

    load_dict = load_model(model=model,
                           checkpoint_dir=args.checkpoint_dir,
                           training=False)

    model = load_dict['model']
    model.eval()
    model.to(device)

    test_patients = joblib.load('test_patients.pkl')
    test_dicts = create_data_dicts_liver_seg(test_patients,
                                             n_channels=6)

    test_loader, transforms = create_dataloader_liver_seg(data_dicts=test_dicts,
                                                          train=False,
                                                          batch_size=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_transforms = Compose([EnsureTyped(keys=['pred']),
                               Invertd(keys=['pred', 'label', 'image'],
                                       transform=transforms,
                                       orig_keys='image',
                                       meta_keys=['pred_meta_dict', 'label_meta_dict', 'image_meta_dict'],
                                       nearest_interp=False,
                                       to_tensor=True),
                               Activationsd(keys='pred', sigmoid=True),
                               # After resampling some label voxels can get float values
                               AsDiscreted(keys=['pred', 'label'], threshold=0.5),
                               KeepLargestConnectedComponentd(keys='pred'),
                               FillHolesd(keys='pred')])

    metric_list = []

    with torch.no_grad():
        for test_data in test_loader:
            # Idx 0 => Assumes batch_size = 1
            patient_id = test_data['patient_id'][0]
            scan_id = test_data['scan_id'][0]
            test_inputs, test_labels = (test_data['image'].to(device),
                                      test_data['label'].to(device))

            roi_size = (128, 128, 48)
            sw_batch_size = 4

            test_data['pred'] = sliding_window_inference(test_inputs,
                                                         roi_size,
                                                         sw_batch_size,
                                                         model,
                                                         overlap=0.5)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]


            test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
            # Dice metric for current iteration
            dice_metric(y_pred=test_outputs, y=test_labels)
            print('Dice metric: {}'.format(dice_metric.aggregate().item()))
            metric_list.append(dice_metric.aggregate().item())
            # Save the image, label, and prediction (in the original spacing)
            save_dir = os.path.join(args.checkpoint_dir, 'results', patient_id, scan_id)
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            save_op_image = SaveImaged(keys='image',
                                       output_postfix='image',
                                       output_dir=save_dir,
                                       separate_folder=False)

            save_op_label = SaveImaged(keys='label',
                                       output_postfix='label',
                                       output_dir=save_dir,
                                       separate_folder=False)

            save_op_pred = SaveImaged(keys='pred',
                                       output_postfix='pred',
                                       output_dir=save_dir,
                                       separate_folder=False)

            for test_dict in test_data:
                save_op_image(test_dict)
                save_op_label(test_dict)
                save_op_pred(test_dict)

            dice_metric.reset()

    np.save(os.path.join(os.path.join(args.checkpoint_dir, 'results', 'dice.npy')),
            np.array(metric_list, dtype=np.float32))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)

    args = parser.parse_args()

    test(args)


