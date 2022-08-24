"""

Script to train landmark correspondence model

See:
    Paper: http://arxiv.org/abs/2001.07434
    Code: https://github.com/monikagrewal/End2EndLandmarks

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
from torch.utils.tensorboard import SummaryWriter
from torchearlystopping.pytorchtools import EarlyStopping
import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'util_scripts'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'lesion_matching', 'src', 'arch'))
from model import LesionMatchingModel
from loss import *
from deformations import *
from datapipeline import *
from tqdm import tqdm

# Required for CacheDataset
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


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

    # Set up data pipeline
    train_patients = joblib.load('train_patients.pkl')
    val_patients = joblib.load('val_patients.pkl')
    print('Number of patients in training set: {}'.format(len(train_patients)))
    print('Number of patients in validation set: {}'.format(len(val_patients)))


    train_dicts = create_data_dicts_lesion_matching(train_patients)
    val_dicts = create_data_dicts_lesion_matching(val_patients)

    train_loader, _ = create_dataloader_lesion_matching(data_dicts=train_dicts,
                                                        train=True,
                                                        batch_size=args.batch_size)

    # Patch-based validation
    val_loader, _ = create_dataloader_lesion_matching(data_dicts=val_dicts,
                                                      train=True,
                                                      batch_size=1,
                                                      num_workers=4)


    model = LesionMatchingModel(K=512,
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
                                                                   device=images.device)

            if batch_deformation_grid is None:
                continue

            images_hat = F.grid_sample(input=images,
                                       grid=batch_deformation_grid,
                                       align_corners=False,
                                       mode="bilinear")

            # Image intensity augmentation
            images_hat = shift_intensity(images_hat)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):


                outputs = model(images.to(device),
                                images_hat.to(device),
                                training=True)

                gt1, gt2, matches, num_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid'][0],
                                                                                     kpts2=outputs['kpt_sampling_grid'][1],
                                                                                     deformation=batch_deformation_grid)

                loss = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                   landmark_logits2=outputs['kpt_logits'][1],
                                   desc_pairs_score=outputs['desc_score'],
                                   desc_pairs_norm=outputs['desc_norm'],
                                   gt1=gt1,
                                   gt2=gt2,
                                   match_target=matches,
                                   k=512,
                                   device=device)
            # Backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'Training loss': loss.item()})
            writer.add_scalar('loss/train', loss.item(), n_iter)
            writer.add_scalar('matches/train', num_matches, n_iter)
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
                                                                       device=images.device)
                # Folding may have occured
                if batch_deformation_grid is None:
                    continue

                images_hat = F.grid_sample(input=images,
                                           grid=batch_deformation_grid,
                                           align_corners=False,
                                           mode="bilinear")

                # Intensity shifts for images_hat

                outputs = model(images.to(device),
                                images_hat.to(device),
                                training=True)

                gt1, gt2, matches, num_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid'][0],
                                                                                     kpts2=outputs['kpt_sampling_grid'][1],
                                                                                     deformation=batch_deformation_grid,
                                                                                     pixel_thresh=2)

                loss = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                   landmark_logits2=outputs['kpt_logits'][1],
                                   desc_pairs_score=outputs['desc_score'],
                                   desc_pairs_norm=outputs['desc_norm'],
                                   gt1=gt1,
                                   gt2=gt2,
                                   match_target=matches,
                                   k=512,
                                   device=device)

                writer.add_scalar('loss/val', loss.item(), n_iter_val)
                writer.add_scalar('matches/val', num_matches, n_iter_val)
                val_loss.append(loss.item())
                pbar_val.set_postfix({'Validation loss': loss.item()})
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
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()

    train(args)
