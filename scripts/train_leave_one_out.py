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

COPD_DIR = '/home/ishaan/COPDGene/mha'
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

    # Create list of patient directories
    train_patients = [os.path.join(COPD_DIR, 'copd{}'.format(idx)) for idx in range(1, 11)] # (copd1, copd2, ..., copd10)

    # Delete the entry corresponding to the holdout index
    assert(args.holdout_idx > 0 and args.holdout_idx <= len(train_patients))
    del train_patients[args.holdout_idx-1]

    train_dicts = create_data_dicts_dir_lab(train_patients,
                                            dataset='copd')

    train_loader = create_dataloader_dir_lab(data_dicts=train_dicts,
                                             batch_size=args.batch_size,
                                             num_workers=4,
                                             test=False,
                                             patch_size=TRAINING_PATCH_SIZE_DIRLAB,
                                             seed=args.seed)

    model = LesionMatchingModel(K=args.kpts_per_batch,
                                W=args.window_size,
                                descriptor_length=args.desc_length)


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


    disp_pdf = joblib.load(os.path.join(args.displacement_dir,
                                        'bspline_motion_pdf_copd.pkl'))

    affine_df = pd.read_pickle(os.path.join(args.displacement_dir,
                                            'affine_transform_parameters_copd.pkl'))
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
                images, mask = (batch_data['image'], batch_data['lung_mask'])

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
                                               padding_mode="zeros")
                else:
                    continue

                mask_hat = F.grid_sample(input=mask,
                                         grid=batch_deformation_grid,
                                         align_corners=True,
                                         mode="nearest",
                                         padding_mode="zeros")

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
    parser.add_argument('--loss_type', type=str, default='aux', help='Choices: hinge, ce, aux')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--desc_length', type=int, default=-1)
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
    parser.add_argument('--holdout_idx', type=int, default=1)

    args = parser.parse_args()

    train(args)
