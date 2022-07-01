"""

Script to train 3-D U-Net using MONAI API
See: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

from monai.utils import first, set_determinism
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from datapipeline import *
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from helper_functions import *
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
from torchearlystopping.pytorchtools import EarlyStopping


def train(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    if os.path.exists(args.checkpoint_dir) is False:
        os.makedirs(args.checkpoint_dir)

    log_dir = os.path.join(args.checkpoint_dir, 'logs')

    if os.path.exists(log_dir) is True:
        shutil.rmtree(log_dir)

    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # Set up data pipeline
    train_patients = joblib.load('train_patients.pkl')
    val_patients = joblib.load('val_patients.pkl')

    train_dicts = create_data_dicts(train_patients)
    val_dicts = create_data_dicts(val_patients)

    train_loader = create_dataloader(data_dicts=train_dicts,
                                     train=True,
                                     batch_size=args.batch_size)

    val_loader, _ = create_dataloader(data_dicts=val_dicts,
                                      train=False,
                                      batch_size=1)

    model = UNet(spatial_dims=3,
                 in_channels=6,
                 out_channels=1,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 1e-4)

    loss_function = nn.BCEWithLogitsLoss()

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    n_iter = 0
    n_iter_val = 0

    early_stopper = EarlyStopping(patience=100,
                                  checkpoint_dir=args.checkpoint_dir,
                                  delta=1e-5)

    print('Start training')
    for epoch in range(10000):

        model.train()

        for batch_data in train_loader:

            inputs, labels = (batch_data['image'].to(device), batch_data['label'].to(device))

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), n_iter)

            n_iter += 1
        print('EPOCH {} done'.format(epoch))

        # Validation
        with torch.no_grad():
            metric_list = []
            for val_data in val_loader:
                val_inputs, val_labels = (val_data['image'].to(device),
                                          val_data['label'].to(device))

                roi_size = (128, 128, 48)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size,
                                                       sw_batch_size,
                                                       model,
                                                       overlap=0.5)

                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # Dice metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            writer.add_scalar('val/dice', metric, n_iter_val)
            n_iter_val += 1
            dice_metric.reset()

        early_stop_condition, best_epoch = early_stopper(val_loss=-1*metric,
                                                         curr_epoch=epoch,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         scheduler=None,
                                                         scaler=None,
                                                         n_iter=n_iter,
                                                         n_iter_val=n_iter_val)

        if early_stop_condition is True:
            print('Early stop condition reached at epoch {}'.format(best_epoch))
            return



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()

    train(args)

