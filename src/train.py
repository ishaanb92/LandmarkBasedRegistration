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
from visualize import *
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

TRAINING_PATCH_SIZE = (128, 128, 64)

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

    viz_dir = os.path.join(args.checkpoint_dir, 'viz')

    if os.path.exists(viz_dir) is True:
        shutil.rmtree(viz_dir)

    os.makedirs(viz_dir)

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

    val_loader, _ = create_dataloader_lesion_matching(data_dicts=val_dicts,
                                                      train=True,
                                                      data_aug=False,
                                                      batch_size=args.batch_size,
                                                      num_workers=4,
                                                      patch_size=TRAINING_PATCH_SIZE)


    model = LesionMatchingModel(K=args.kpts_per_batch,
                                W=args.window_size)

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
                                                                   non_rigid=True,
                                                                   coarse=True,
                                                                   fine=args.fine_deform,
                                                                   coarse_displacements=(2, 4, 4),
                                                                   fine_displacements=(1, 2, 2),
                                                                   coarse_grid_resolution=(4, 4, 4),
                                                                   fine_grid_resolution=(8, 8, 8))
            if batch_deformation_grid is not None:
                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=True,
                                           mode="bilinear",
                                           padding_mode="border")
            else:
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

            # Check for empty liver masks and other issues!
            skip_batch = False
            for bid in range(liver_mask.shape[0]):
                if torch.max(liver_mask[bid, ...]) == 0 or torch.max(liver_mask_hat[bid, ...]) == 0:
                    skip_batch = True
                    break
                if torch.max(images[bid, ...]) == torch.min(images[bid, ...]):
                    print('Min and max values of image are the same!!!!')
                    skip_batch = True
                    break

                if torch.max(images_hat[bid, ...]) == torch.min(images_hat[bid, ...]):
                    print('Min and max values of deformed image are the same!!!!')
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
            writer.add_scalar('train/landmark_1_loss_max_p', loss_dict['landmark_1_loss_max_p'].item(), n_iter)
            writer.add_scalar('train/landmark_1_wce', loss_dict['landmark_1_loss_wce'].item(), n_iter)
            writer.add_scalar('train/landmark_2_loss_max_p', loss_dict['landmark_2_loss_max_p'].item(), n_iter)
            writer.add_scalar('train/landmark_2_wce', loss_dict['landmark_2_loss_wce'].item(), n_iter)
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
                                                                       non_rigid=True,
                                                                       coarse=True,
                                                                       fine=args.fine_deform,
                                                                       coarse_displacements=(2, 4, 4),
                                                                       fine_displacements=(1, 2, 2),
                                                                       coarse_grid_resolution=(4, 4, 4),
                                                                       fine_grid_resolution=(8, 8, 8))
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

                skip_batch = False
                for bid in range(liver_mask.shape[0]):
                    if torch.max(liver_mask[bid, ...]) == 0 or torch.max(liver_mask_hat[bid, ...]) == 0:
                        skip_batch = True
                        break
                    if torch.max(images[bid, ...]) == torch.min(images[bid, ...]):
                        print('Min and max values of image are the same!!!!')
                        skip_batch = True
                        break

                    if torch.max(images_hat[bid, ...]) == torch.min(images_hat[bid, ...]):
                        print('Min and max values of deformed image are the same!!!!')
                        skip_batch = True
                        break


                if skip_batch is True:
                    continue

                # Run inference block here
                images_cat = torch.cat([images, images_hat],
                                       dim=1)

                kpts_logits_1, kpts_logits_2 = model.get_patch_keypoint_scores(images_cat.to(device))

                features_1_low, features_1_high, features_2_low, features_2_high =\
                                                        model.get_patch_feature_descriptors(images_cat.to(device))

                features_1 = (features_1_low.to(device), features_1_high.to(device))
                features_2 = (features_2_low.to(device), features_1_high.to(device))

                outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                          kpts_2=kpts_logits_2.to(device),
                                          features_1=features_1,
                                          features_2=features_2,
                                          conf_thresh=0.05,
                                          num_pts=args.kpts_per_batch,
                                          mask=liver_mask.to(device),
                                          mask2=liver_mask_hat.to(device),
                                          test=False)

                if outputs is None: # Too few keypoints found
                    continue

                if args.dummy is True:
                    assert(torch.equal(outputs['kpt_sampling_grid'][0], outputs['kpt_sampling_grid'][1]))


                gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                           kpts2=outputs['kpt_sampling_grid_2'],
                                                                                           deformation=batch_deformation_grid)

                loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits_1'],
                                        landmark_logits2=outputs['kpt_logits_2'],
                                        desc_pairs_score=outputs['desc_score'],
                                        desc_pairs_norm=outputs['desc_norm'],
                                        gt1=gt1,
                                        gt2=gt2,
                                        match_target=gt_matches,
                                        k=args.kpts_per_batch,
                                        device=device)

                for batch_id in range(images.shape[0]):
                    im1 = images[batch_id, ...].squeeze(dim=0)
                    im2 = images_hat[batch_id, ...].squeeze(dim=0)
                    patient_id = val_data['patient_id'][batch_id]
                    scan_id = val_data['scan_id'][batch_id]

                    out_dir = os.path.join(viz_dir,
                                           patient_id,
                                           scan_id,
                                           'epoch_{}'.format(epoch))

                    if os.path.exists(out_dir) is False:
                        os.makedirs(out_dir)

                    visualize_keypoints_3d(im1=im1,
                                           im2=im2,
                                           landmarks1=outputs['landmarks_1'][batch_id, ...],
                                           landmarks2=outputs['landmarks_2'][batch_id, ...],
                                           pred_matches=outputs['matches'][batch_id, ...],
                                           gt_matches=gt_matches[batch_id, ...],
                                           out_dir=out_dir,
                                           neighbourhood=5,
                                           verbose=False)

                writer.add_scalar('val/loss', loss_dict['loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_1_loss', loss_dict['landmark_1_loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_2_loss', loss_dict['landmark_2_loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_1_loss_max_p', loss_dict['landmark_1_loss_max_p'].item(), n_iter_val)
                writer.add_scalar('val/landmark_1_wce', loss_dict['landmark_1_loss_wce'].item(), n_iter_val)
                writer.add_scalar('val/landmark_2_loss_max_p', loss_dict['landmark_2_loss_max_p'].item(), n_iter_val)
                writer.add_scalar('val/landmark_2_wce', loss_dict['landmark_2_loss_wce'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_ce', loss_dict['desc_loss_ce'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_hinge', loss_dict['desc_loss_hinge'].item(), n_iter_val)
                writer.add_scalar('val/GT matches', num_gt_matches, n_iter_val)
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
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--kpts_per_batch', type=int, default=512)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--fine_deform', action='store_true')

    args = parser.parse_args()

    train(args)
