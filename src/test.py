"""

Script to evaluate trained models (with visualization)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from utils.utils import *
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'arch'))
from model import LesionMatchingModel
from deformations import *
from datapipeline import *


def test(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    checkpoint_dir = args.checkpoint_dir

    # Set up data pipeline
    if args.mode == 'val':
        patients = joblib.load('val_patients.pkl')
    elif args.mode == 'test':
        patients = joblib.load('test_patients.pkl')


    if args.synthetic is True:
        data_dicts = create_data_dicts_lesion_matching(patients)
        data_loader, _ = create_dataloader_lesion_matching(data_dicts=data_dicts,
                                                          train=True,
                                                          batch_size=args.batch_size,
                                                          num_workers=4)
    else: # "Real" data
        data_dicts = create_data_dicts_lesion_matching_inference(patients)
        data_loader, _ = create_dataloader_lesion_matching_inference(data_dicts=data_dicts,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=4)


    # Define the model
    model = LesionMatchingModel(K=512,
                                W=4)

    # Load the model
    load_dict = load_model(model=model,
                           checkpoint_dir=checkpoint_dir,
                           training=False)

    model = load_dict['model']

    model.to(device)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if args.synthetic is True:

                images, liver_mask, vessel_mask = (batch_data['image'], batch_data['liver_mask'], batch_data['vessel_mask'])

                batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                       device=images.device)

                if batch_deformation_grid is None:
                    continue

                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=False,
                                           mode="bilinear")

                landmarks_1, landmarks_2, matches = model.inference(images.to(device),
                                                                    images_hat.to(device))

                print(matches.shape)
                print(landmarks_1.shape)
                print(landmarks_2.shape)

            else: #TODO
                pass





if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--synthetic', action='store_true')

    args = parser.parse_args()

    test(args)
