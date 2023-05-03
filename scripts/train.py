"""

Script to train landmark correspondence model

See:
    Paper: http://arxiv.org/abs/2001.07434
    Code: https://github.com/monikagrewal/End2EndLandmarks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os, sys
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
from monai.utils.misc import set_determinism
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from lesionmatching.util_scripts.utils import *
from lesionmatching.analysis.visualize import *
from torch.utils.tensorboard import SummaryWriter
from torchearlystopping.pytorchtools import EarlyStopping
from lesionmatching.arch.model import LesionMatchingModel
from lesionmatching.arch.loss import *
from lesionmatching.data.deformations import *
from lesionmatching.data.datapipeline import *
from tqdm import tqdm
import numpy as np
import torch
import random
import warnings

# Required for CacheDataset
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

TRAINING_PATCH_SIZE = (128, 128, 64)
TRAINING_PATCH_SIZE_DIRLAB = (128, 128, 96)

def train(args):

    if args.soft_masking is False:
        proceed = input('Softmasking is set to False. Are you sure you want to continue? (y/n)')
        if proceed == 'n':
            return
        elif proceed == 'y':
            pass

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    log_dir = os.path.join(args.checkpoint_dir, 'logs')
    viz_dir = os.path.join(args.checkpoint_dir, 'viz')

    new_training = True

    # Case 1 : Checkpoint directory does not exist => New training
    if os.path.exists(args.checkpoint_dir) is False:
        os.makedirs(args.checkpoint_dir)
        os.makedirs(log_dir)
        os.makedirs(viz_dir)
    else: # Case 2 : Checkpoint directory exists
        if args.renew is True: # Case 2a : Delete existing checkpoint directory and create a new one
            shutil.rmtree(args.checkpoint_dir)
            os.makedirs(args.checkpoint_dir)
            os.makedirs(log_dir)
            os.makedirs(viz_dir)
        else: # Case 2b
            if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pt')) is True:
                new_training = False
            else:
                new_training = True

    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = args.checkpoint_dir

    # Set the (global) seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_determinism(seed=args.seed)

    # Set up data pipeline
    train_patients = joblib.load('train_patients_{}.pkl'.format(args.dataset))
    val_patients = joblib.load('val_patients_{}.pkl'.format(args.dataset))

    # For DIR-Lab, the validation set is only used to visually monitor training
    # in terms predicted landmarks and their correspondences!
    # During test-time, we use the COPDGene dataset
    if args.dataset == 'dirlab':
        train_patients.extend(val_patients)

    print('Number of patients in training set: {}'.format(len(train_patients)))
    print('Number of patients in validation set: {}'.format(len(val_patients)))

    if args.dataset == 'umc':
        train_dicts = create_data_dicts_lesion_matching(train_patients)
        val_dicts = create_data_dicts_lesion_matching(val_patients)
        # Patch-based training
        train_loader, _ = create_dataloader_lesion_matching(data_dicts=train_dicts,
                                                            train=True,
                                                            data_aug=args.data_aug,
                                                            batch_size=args.batch_size,
                                                            num_workers=4,
                                                            patch_size=TRAINING_PATCH_SIZE,
                                                            seed=args.seed,
                                                            num_samples=args.num_samples)

        val_loader, _ = create_dataloader_lesion_matching(data_dicts=val_dicts,
                                                          train=True,
                                                          data_aug=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=4,
                                                          patch_size=TRAINING_PATCH_SIZE,
                                                          seed=args.seed,
                                                          num_samples=args.num_samples)
    elif args.dataset == 'dirlab':
        train_dicts = create_data_dicts_dir_lab(train_patients)
        val_dicts = create_data_dicts_dir_lab(val_patients)

        train_loader = create_dataloader_dir_lab(data_dicts=train_dicts,
                                                 batch_size=args.batch_size,
                                                 num_workers=4,
                                                 test=False,
                                                 patch_size=TRAINING_PATCH_SIZE_DIRLAB,
                                                 seed=args.seed)

        val_loader = create_dataloader_dir_lab(data_dicts=val_dicts,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               test=False,
                                               patch_size=TRAINING_PATCH_SIZE_DIRLAB,
                                               seed=args.seed)


    model = LesionMatchingModel(K=args.kpts_per_batch,
                                W=args.window_size)


    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 1e-4)

    early_stopper = EarlyStopping(patience=args.patience,
                                  checkpoint_dir=args.checkpoint_dir,
                                  delta=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    if new_training is False: # Load model
        load_dict = load_model(model=model,
                               optimizer=optimizer,
                               scaler=scaler,
                               checkpoint_dir=args.checkpoint_dir,
                               training=True,
                               device=device)

        n_iter = load_dict['n_iter']
        n_iter_val = load_dict['n_iter_val']
        optimizer = load_dict['optimizer']
        model = load_dict['model']
        epoch_saved = load_dict['epoch']
        if epoch_saved is None:
            epoch_saved = n_iter//(len(train_loader)*args.num_samples)
        print('Resuming training from epoch {}'.format(epoch_saved+1))

    else: # Reset parameters and initialize all counters
        model.apply(resetModelWeights)
        n_iter = 0
        n_iter_val = 0
        epoch_saved = -1


    if args.dataset == 'umc':
        coarse_displacements = (4, 8, 8)
        coarse_grid_resolution = (3, 3, 3)
        fine_displacements = (2, 4, 4)
        fine_grid_resolution = (6, 6, 6)
        pixel_thresh = (2, 4, 4)
    elif args.dataset == 'dirlab':
        disp_pdf = joblib.load(os.path.join(args.displacement_dir,
                                            'bspline_motion_pdf.pkl'))

        affine_df = pd.read_pickle(os.path.join(args.displacement_dir,
                                                'affine_transform_parameters.pkl'))
        coarse_grid_resolution = (2, 2, 2)
        fine_grid_resolution = (3, 3, 3)
        pixel_thresh = (1, 2, 2)

    if args.loss_type == 'ce':
        desc_loss_comp_wt = torch.Tensor([1.0, 0.0])
    elif args.loss_type == 'hinge':
        desc_loss_comp_wt = torch.Tensor([0.0, 1.0])
    elif args.loss_type == 'aux':
        desc_loss_comp_wt = torch.Tensor([1.0, 1.0])

    print('Start training')
    for epoch in range(epoch_saved+1, args.epochs):

        model.train()
        nbatches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), desc="training", total=nbatches, unit="batches")

        for batch_idx, batch_data_list in pbar:

            if isinstance(batch_data_list, dict):
                batch_data_list = [batch_data_list]

            for sample_idx, batch_data in enumerate(batch_data_list):
                if args.dataset == 'umc':
                    images, mask, vessel_mask = (batch_data['image'], batch_data['liver_mask'], batch_data['vessel_mask'])
                elif args.dataset == 'dirlab':
                    images, mask = (batch_data['image'], batch_data['lung_mask'])

                    # Additional non-rigid deformation -- See Eppenhof and Pluim (2019), TMI
                    if args.data_aug is True:
                        augmentation_deformation_grid, _ = create_batch_deformation_grid_from_pdf(shape=images.shape,
                                                                                                  non_rigid=True,
                                                                                                  coarse=True,
                                                                                                  fine=False,
                                                                                                  disp_pdf=disp_pdf,
                                                                                                  coarse_grid_resolution=(2, 2, 2))
                        if augmentation_deformation_grid is not None:
                            images = F.grid_sample(input=images,
                                                   grid=augmentation_deformation_grid,
                                                   align_corners=True,
                                                   mode="bilinear",
                                                   padding_mode="border")

                            mask = F.grid_sample(input=images,
                                                 grid=augmentation_deformation_grid,
                                                 align_corners=True,
                                                 mode="nearest",
                                                 padding_mode="border")


                if args.dataset == 'umc':
                    batch_deformation_grid, jac_det = create_batch_deformation_grid(shape=images.shape,
                                                                                    non_rigid=True,
                                                                                    coarse=True,
                                                                                    fine=True,
                                                                                    coarse_displacements=coarse_displacements,
                                                                                    fine_displacements=fine_displacements,
                                                                                    coarse_grid_resolution=coarse_grid_resolution,
                                                                                    fine_grid_resolution=fine_grid_resolution)
                elif args.dataset == 'dirlab':
                    batch_deformation_grid, jac_det = create_batch_deformation_grid_from_pdf(shape=images.shape,
                                                                                             non_rigid=True,
                                                                                             coarse=True,
                                                                                             fine=True,
                                                                                             disp_pdf=disp_pdf,
                                                                                             affine_df=affine_df,
                                                                                             coarse_grid_resolution=coarse_grid_resolution,
                                                                                             fine_grid_resolution=fine_grid_resolution)
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

                    if args.dataset == 'umc':
                        images_hat = shift_intensity(images_hat)
                    elif args.dataset == 'copd':
                        if args.dry_sponge is True:
                            # See : H. Sokooti et al., 3D Convolutional Neural Networks Image Registration
                            # Based on Efficient Supervised Learning from Artificial Deformations
                            images_hat = dry_sponge_augmentation(images_hat,
                                                                 jac_det)
                        else:
                            pass

                else:
                    images_hat = F.grid_sample(input=images,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="nearest",
                                               padding_mode="border")

                    assert(torch.equal(images, images_hat))

                # Transform liver mask
                mask_hat = F.grid_sample(input=mask,
                                         grid=batch_deformation_grid,
                                         align_corners=True,
                                         mode="nearest",
                                         padding_mode="border")

                # Check for empty liver masks and other issues!
                skip_batch = False
                for bid in range(mask.shape[0]):
                    if torch.max(mask[bid, ...]) == 0 or torch.max(mask_hat[bid, ...]) == 0:
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
                                    mask=mask.to(device),
                                    mask2=mask_hat.to(device),
                                    training=True,
                                    soft_masking=args.soft_masking)

                    if outputs is None: # Too few keypoints found
                        continue

                    if args.dummy is True:
                        assert(torch.equal(outputs['kpt_sampling_grid'][0], outputs['kpt_sampling_grid'][1]))


                    gt1, gt2, matches, num_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid'][0],
                                                                                         kpts2=outputs['kpt_sampling_grid'][1],
                                                                                         deformation=batch_deformation_grid,
                                                                                         pixel_thresh=pixel_thresh)

                    loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                            landmark_logits2=outputs['kpt_logits'][1],
                                            desc_pairs_score=outputs['desc_score'],
                                            desc_pairs_norm=outputs['desc_norm'],
                                            gt1=gt1,
                                            gt2=gt2,
                                            match_target=matches,
                                            k=args.kpts_per_batch,
                                            mask_idxs_1 = outputs['mask_idxs_1'],
                                            mask_idxs_2 = outputs['mask_idxs_2'],
                                            device=device,
                                            desc_loss_comp_wt=desc_loss_comp_wt.to(device))
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
                writer.add_scalar('train/kpts_inside_lung_1', loss_dict['kpts_inside_lung_1'], n_iter)
                writer.add_scalar('train/kpts_inside_lung_2', loss_dict['kpts_inside_lung_2'], n_iter)
                writer.add_scalar('train/kpts_outside_lung_1', loss_dict['kpts_outside_lung_1'], n_iter)
                writer.add_scalar('train/kpts_outside_lung_2', loss_dict['kpts_outside_lung_2'], n_iter)
                n_iter += 1

        print('EPOCH {} done'.format(epoch))

        with torch.no_grad():
            print('Start validation')
            model.eval()
            val_loss = []
            nbatches = len(val_loader)
            pbar_val = tqdm(enumerate(val_loader), desc="validation", total=nbatches, unit="batches")
            for batch_val_idx, val_data_list in pbar_val:
                if isinstance(val_data_list, dict):
                    val_data_list = [val_data_list]
                for sample_val_idx, val_data in enumerate(val_data_list):
                    if args.dataset == 'umc':
                        images, mask, vessel_mask = (val_data['image'], val_data['liver_mask'], val_data['vessel_mask'])
                    elif args.dataset == 'dirlab':
                        images, mask = (val_data['image'], val_data['lung_mask'])

                    if args.dataset == 'umc':
                        batch_deformation_grid, jac_det = create_batch_deformation_grid(shape=images.shape,
                                                                                        non_rigid=True,
                                                                                        coarse=True,
                                                                                        fine=True,
                                                                                        coarse_displacements=coarse_displacements,
                                                                                        fine_displacements=fine_displacements,
                                                                                        coarse_grid_resolution=coarse_grid_resolution,
                                                                                        fine_grid_resolution=fine_grid_resolution)
                    elif args.dataset == 'dirlab':
                        batch_deformation_grid, jac_det = \
                            create_batch_deformation_grid_from_pdf(shape=images.shape,
                                                                   non_rigid=True,
                                                                   coarse=True,
                                                                   fine=True,
                                                                   disp_pdf=disp_pdf,
                                                                   affine_df=affine_df,
                                                                   coarse_grid_resolution=coarse_grid_resolution,
                                                                   fine_grid_resolution=fine_grid_resolution)


                    if batch_deformation_grid is None:
                        continue

                    images_hat = F.grid_sample(input=images,
                                               grid=batch_deformation_grid,
                                               align_corners=True,
                                               mode="bilinear",
                                               padding_mode="border")


                    # Transform liver mask
                    mask_hat = F.grid_sample(input=mask,
                                                   grid=batch_deformation_grid,
                                                   align_corners=True,
                                                   mode="nearest",
                                                   padding_mode="border")

                    assert(images.shape == images_hat.shape)

                    skip_batch = False
                    for bid in range(mask.shape[0]):
                        if torch.max(mask[bid, ...]) == 0 or torch.max(mask_hat[bid, ...]) == 0:
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

                    unet_outputs = model.get_unet_outputs(images_cat.to(device))

                    kpts_logits_1 = unet_outputs['kpts_logits_1']
                    kpts_logits_2 = unet_outputs['kpts_logits_2']
                    features_1_low = unet_outputs['features_1_low']
                    features_1_high = unet_outputs['features_1_high']
                    features_2_low = unet_outputs['features_2_low']
                    features_2_high = unet_outputs['features_2_high']

                    features_1 = (features_1_low.to(device), features_1_high.to(device))
                    features_2 = (features_2_low.to(device), features_2_high.to(device))

                    outputs = model.inference(kpts_1=kpts_logits_1.to(device),
                                              kpts_2=kpts_logits_2.to(device),
                                              features_1=features_1,
                                              features_2=features_2,
                                              conf_thresh=0.05,
                                              num_pts=args.kpts_per_batch,
                                              mask=mask.to(device),
                                              mask2=mask_hat.to(device),
                                              soft_masking=args.soft_masking,
                                              test=False)

                    if outputs is None: # Too few keypoints found
                        continue

                    if args.dummy is True:
                        assert(torch.equal(outputs['kpt_sampling_grid'][0], outputs['kpt_sampling_grid'][1]))


                    gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                               kpts2=outputs['kpt_sampling_grid_2'],
                                                                                               deformation=batch_deformation_grid,
                                                                                               pixel_thresh=pixel_thresh)

                    loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits_1'],
                                            landmark_logits2=outputs['kpt_logits_2'],
                                            desc_pairs_score=outputs['desc_score'],
                                            desc_pairs_norm=outputs['desc_norm'],
                                            gt1=gt1,
                                            gt2=gt2,
                                            match_target=gt_matches,
                                            k=args.kpts_per_batch,
                                            mask_idxs_1=outputs['mask_idxs_1'],
                                            mask_idxs_2=outputs['mask_idxs_2'],
                                            device=device,
                                            desc_loss_comp_wt=desc_loss_comp_wt.to(device))

                    for batch_id in range(images.shape[0]):
                        im1 = images[batch_id, ...].squeeze(dim=0)
                        im2 = images_hat[batch_id, ...].squeeze(dim=0)
                        patient_id = val_data['patient_id'][batch_id]
                        if args.dataset == 'umc':
                            scan_id = val_data['scan_id'][batch_id]
                        elif args.dataset == 'dirlab':
                            scan_id = val_data['type'][batch_id]

                        out_dir = os.path.join(viz_dir,
                                               patient_id,
                                               scan_id,
                                               'epoch_{}'.format(epoch),
                                               'sample_{}'.format(sample_val_idx))

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
                    writer.add_scalar('val/kpts_inside_lung_1', loss_dict['kpts_inside_lung_1'], n_iter_val)
                    writer.add_scalar('val/kpts_inside_lung_2', loss_dict['kpts_inside_lung_2'], n_iter_val)
                    writer.add_scalar('val/kpts_outside_lung_1', loss_dict['kpts_outside_lung_1'], n_iter_val)
                    writer.add_scalar('val/kpts_outside_lung_2', loss_dict['kpts_outside_lung_2'], n_iter_val)
                    val_loss.append(loss_dict['loss'].item())
                    pbar_val.set_postfix({'Validation loss': loss_dict['loss'].item()})
                    n_iter_val += 1

            mean_val_loss = np.mean(np.array(val_loss))

            if args.earlystop is True:
                assert(args.dataset != 'dirlab')
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
            else: # Save the model every epoch
                save_model(model=model,
                           optimizer=optimizer,
                           scheduler=None,
                           scaler=scaler,
                           n_iter=n_iter,
                           epoch=epoch,
                           n_iter_val=n_iter_val,
                           checkpoint_dir=args.checkpoint_dir)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--displacement_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--loss_type', type=str, default='aux', help='Choices: hinge, ce, aux')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--kpts_per_batch', type=int, default=512)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--earlystop', action='store_true')
    parser.add_argument('--renew', action='store_true')
    parser.add_argument('--dry_sponge', action='store_true')
    parser.add_argument('--soft_masking', action='store_true')

    args = parser.parse_args()

    train(args)
