"""

Script to train landmark correspondence model

See:
    Paper: http://arxiv.org/abs/2001.07434
    Code: https://github.com/monikagrewal/End2EndLandmarks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'arch'))
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
from monai.inferers import sliding_window_inference
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torchearlystopping.pytorchtools import EarlyStopping
from model import LesionMatchingModel
from loss import *
from deformations import *
from datapipeline import *
from tqdm import tqdm
import numpy as np
import torch
import random

# Required for CacheDataset
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

TRAINING_PATCH_SIZE = (96, 96, 48)

def train(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    if os.path.exists(args.checkpoint_dir) is False:
        os.makedirs(args.checkpoint_dir)

    checkpoint_dir = args.checkpoint_dir

    log_dir = os.path.join(args.checkpoint_dir, 'logs')

    if os.path.exists(log_dir) is True:
        shutil.rmtree(log_dir)

    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # Set the seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up data pipeline
    train_patients = joblib.load('train_patients.pkl')
    val_patients = joblib.load('val_patients.pkl')
    print('Number of patients in training set: {}'.format(len(train_patients)))
    print('Number of patients in validation set: {}'.format(len(val_patients)))


    train_dicts = create_data_dicts_lesion_matching(train_patients)
    val_dicts = create_data_dicts_lesion_matching(val_patients)

    # Patch-based training
    train_loader, _ = create_dataloader_lesion_matching(data_dicts=train_dicts,
                                                        train=True,
                                                        data_aug=args.data_aug,
                                                        batch_size=args.batch_size,
                                                        num_workers=4,
                                                        patch_size=TRAINING_PATCH_SIZE)

    # Validation on full image!
    val_loader, _ = create_dataloader_lesion_matching(data_dicts=val_dicts,
                                                      train=False,
                                                      data_aug=False,
                                                      batch_size=1,
                                                      num_workers=4)


    model = LesionMatchingModel(K=args.kpts_per_batch,
                                W=4)

    optimizer = torch.optim.Adam(model.parameters(),
                                 1e-4)

    early_stopper = EarlyStopping(patience=args.patience,
                                  checkpoint_dir=args.checkpoint_dir,
                                  delta=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.to(device)
    n_iter = 0
    n_iter_val = 0

    print('Start training')
    for epoch in range(10000):


        model.train()
        nbatches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), desc="training", total=nbatches, unit="batches")

        for batch_idx, batch_data in pbar:

            images, liver_mask, vessel_mask = (batch_data['image'], batch_data['liver_mask'], batch_data['vessel_mask'])

            batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                   device=images.device,
                                                                   dummy=args.dummy,
                                                                   non_rigid=True,
                                                                   coarse=True,
                                                                   coarse_displacements=(3, 3, 3))

            if batch_deformation_grid is None:
                continue

            if args.dummy is False:
                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="bilinear",
                                           padding_mode="border")

                # Image intensity augmentation
                images_hat = shift_intensity(images_hat)
            else:
                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="nearest",
                                           padding_mode="border")

                assert(torch.equal(images, images_hat))

            # Transform liver mask
            liver_mask_hat = F.grid_sample(input=liver_mask,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="nearest",
                                           padding_mode="border")

            # Check for empty liver masks!
            skip_batch = False
            for bid in range(liver_mask.shape[0]):
                if torch.max(liver_mask[bid, ...]) == 0 or torch.max(liver_mask_hat[bid, ...]) == 0:
                    skip_batch = True
                    break

            if skip_batch is True:
                continue

            assert(images.shape == images_hat.shape)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):

                outputs = model(x1=images.to(device),
                                x2=images_hat.to(device),
                                mask=liver_mask.to(device),
                                mask2=liver_mask_hat.to(device),
                                training=True)

                if outputs is None: # Too few keypoints found
                    continue

                if args.dummy is True:
                    assert(torch.equal(outputs['kpt_sampling_grid'][0], outputs['kpt_sampling_grid'][1]))


                gt1, gt2, matches, num_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid'][0],
                                                                                     kpts2=outputs['kpt_sampling_grid'][1],
                                                                                     deformation=batch_deformation_grid)

                loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                        landmark_logits2=outputs['kpt_logits'][1],
                                        desc_pairs_score=outputs['desc_score'],
                                        desc_pairs_norm=outputs['desc_norm'],
                                        gt1=gt1,
                                        gt2=gt2,
                                        match_target=matches,
                                        k=args.kpts_per_batch,
                                        device=device)
            # Backprop
            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'Training loss': loss_dict['loss'].item()})
            writer.add_scalar('train/loss', loss_dict['loss'].item(), n_iter)
            writer.add_scalar('train/landmark_1_loss', loss_dict['landmark_1_loss'].item(), n_iter)
            writer.add_scalar('train/landmark_2_loss', loss_dict['landmark_2_loss'].item(), n_iter)
            writer.add_scalar('train/desc_loss_ce', loss_dict['desc_loss_ce'].item(), n_iter)
            writer.add_scalar('train/desc_loss_hinge', loss_dict['desc_loss_hinge'].item(), n_iter)
            writer.add_scalar('train/gt_matches', num_matches, n_iter)
            n_iter += 1

        print('EPOCH {} done'.format(epoch))

        with torch.no_grad():
            print('Start validation')
            model.eval()
            val_loss = []
            nbatches = len(val_loader)
            pbar_val = tqdm(enumerate(val_loader), desc="validation", total=nbatches, unit="batches")
            for batch_val_idx, val_data in pbar_val:
                images, liver_mask, vessel_mask = (val_data['image'], val_data['liver_mask'], val_data['vessel_mask'])

                batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                       device=images.device,
                                                                       dummy=args.dummy,
                                                                       non_rigid=True,
                                                                       coarse=True,
                                                                       coarse_displacements=(4, 4, 4))
                # Folding may have occured
                if batch_deformation_grid is None:
                    continue

                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="bilinear",
                                           padding_mode="border")

                # Transform liver mask
                liver_mask_hat = F.grid_sample(input=liver_mask,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="nearest",
                                               padding_mode="border")

                assert(images.shape == images_hat.shape)

                # Concatenate along channel axis so that sliding_window_inference can
                # be used
                assert(images_hat.shape == images.shape)
                images_cat = torch.cat([images, images_hat], dim=1)


                # Pad the z-axis to make the image a cube : 256 x 256 x 256
                # Otherwise, sliding_window_inference complains
                depth = images_cat.shape[-1]
                pad = 256-depth

                images_cat = F.pad(images_cat, (0, pad), "constant", 0)
                liver_mask = F.pad(liver_mask, (0, pad), "constant", 0)
                liver_mask_hat = F.pad(liver_mask_hat, (0, pad), "constant", 0)


                # Keypoint logits
                kpts_logits_1, kpts_logits_2 = sliding_window_inference(inputs=images_cat,
                                                                        roi_size=TRAINING_PATCH_SIZE,
                                                                        sw_batch_size=4,
                                                                        sw_device=device,
                                                                        device='cpu',
                                                                        predictor=model.get_patch_keypoint_scores,
                                                                        overlap=0.5)

                # Feature maps
                features_1_low, features_1_high, features_2_low, features_2_high =\
                                                        sliding_window_inference(inputs=images_cat,
                                                                                 roi_size=TRAINING_PATCH_SIZE,
                                                                                 sw_batch_size=4,
                                                                                 sw_device=device,
                                                                                 device='cpu',
                                                                                 predictor=model.get_patch_feature_descriptors,
                                                                                 overlap=0.5)

                features_1 = (features_1_low.to(device), features_1_high.to(device))
                features_2 = (features_2_low.to(device), features_1_high.to(device))



                # Get (predicted) landmarks and matches on the full image
                # These landmarks are predicted based on L2-norm between feature descriptors
                # and predicted matching probability
                outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                          kpts_2=kpts_logits_2.to(device),
                                          features_1=features_1,
                                          features_2=features_2,
                                          conf_thresh=0.05,
                                          num_pts=args.kpts_per_batch,
                                          mask=liver_mask.to(device),
                                          mask2=liver_mask_hat.to(device))

                # Get ground truth matches based on projecting keypoints using the deformation grid
                gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                           kpts2=outputs['kpt_sampling_grid_2'],
                                                                                           deformation=batch_deformation_grid,
                                                                                           pixel_thresh=5)

                loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits_1'],
                                        landmark_logits2=outputs['kpt_logits_2'],
                                        desc_pairs_score=outputs['desc_score'],
                                        desc_pairs_norm=outputs['desc_norm'],
                                        gt1=gt1,
                                        gt2=gt2,
                                        match_target=gt_matches,
                                        k=args.kpts_per_batch,
                                        device=device)

                writer.add_scalar('val/loss', loss_dict['loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_1_loss', loss_dict['landmark_1_loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_2_loss', loss_dict['landmark_2_loss'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_ce', loss_dict['desc_loss_ce'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_hinge', loss_dict['desc_loss_hinge'].item(), n_iter_val)
                writer.add_scalar('val/GT matches', num_gt_matches, n_iter_val)
                writer.add_scalar('val/Predicted matches', torch.sum(outputs['matches']).item(), n_iter_val)
                val_loss.append(loss_dict['loss'].item())
                pbar_val.set_postfix({'Validation loss': loss_dict['loss'].item()})
                n_iter_val += 1

            mean_val_loss = np.mean(np.array(val_loss))

            early_stop_condition, best_epoch = early_stopper(val_loss=mean_val_loss,
                                                             curr_epoch=epoch,
                                                             model=model,
                                                             optimizer=optimizer,
                                                             scheduler=None,
                                                             scaler=scaler,
                                                             n_iter=n_iter,
                                                             n_iter_val=n_iter_val)
            if early_stop_condition is True:
                print('Best epoch = {}, stopping training'.format(best_epoch))
                return


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--kpts_per_batch', type=int, default=512)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--dummy', action='store_true')

    args = parser.parse_args()

    train(args)
