"""
Miscelleneous utility functions

"""
import torch
import imageio
import pickle
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label
from skimage.transform import resize
from math import sqrt


def get_sub_dirs(base_dir):
    """
    Returns list of sub-directories within a given base directory

    :param base_dir: (str) Path to the base directory
    :return list_of_sub_dirs: (python list) List of sub-directories
    """
    return [os.path.join(base_dir, p_dir) for p_dir in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, p_dir))]


def save_as_image_grid(result_dir=None,
                       image_batch=tuple([]),
                       label_batch=None,
                       preds_batch=None,
                       fmt='png',
                       prefix=None,
                       n_channels=1,
                       gpu_id=-1,
                       ):
    """
    Take a batch of tensors (images, labels and predictions) and save the batch
    as a collection of image grids, each image grid being one image-label-prediction
    triplet from a single member of the batch.

    Row 0 : Original Image
    Row 1 : True labels for all classes (Background-Liver-Right Kidney-Left Kidney-Spleen)
    Row 2 : Predicted masks for each of the organs
    Row 3 : Segmented (per-class) output after applying the mask to the image

    Parameters:
        result_dir (str or Path object) : Directory to store the images
        image_batch (list) : List of image batches to be saved (batch_size x channels x height x width)
        label_batch (torch.Tensor) : Ground truth maps batch to be saved (batch_size x n_classes x height x width)
        preds_batch (torch.Tensor) : Model predictions batch to be saved (batch_size x n_classes x height x width)
        fmtimport pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
 (str) : Extension used to save the image
        prefix (str) : Used a prefix in the filename under which the image is saved
        n_channels (int) : Number of channels in an image (usually 1 or 3)
        gpu_id (int) : ID of the GPU Pytorch is using. If gpu_id < 0 => CPU is used
        train (bool) :  Flag to indicate whether saved images are part of training or inference

    Returns:
        None

    """

    if len(image_batch) == 0:
        raise RuntimeError('List of images is empty.')

    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)

    image_batch = list(image_batch)

    # Convert torch tensors to numpy ndarray
    if gpu_id >= 0:
        for idx in range(len(image_batch)):
            image_batch[idx] = image_batch[idx].cpu().numpy()

        label_batch = label_batch.cpu().numpy()
        preds_batch = preds_batch.cpu().numpy()
    else:
        for idx in range(len(image_batch)):
            image_batch[idx] = image_batch[idx].detach().numpy()
        label_batch = label_batch.detach().numpy()
        preds_batch = preds_batch.detach().numpy()

    preds_batch = np.where(preds_batch >= 0.5, 1, 0)

    # Adjust dynamic range while converting to np.uint8
    # to avoid loss compression
    for idx in range(len(image_batch)):
        image_batch[idx] = convert_to_grayscale(image_batch[idx], dtype=np.uint8)

    label_batch = np.array(np.where(label_batch >= 0.5, 255, 0), dtype=np.uint8)
    preds_batch = np.array(np.where(preds_batch >= 0.5, 255, 0), dtype=np.uint8)

    # [C,H,W] -> [H,W,C]
    for idx in range(len(image_batch)):
        if len(image_batch[idx].shape) == 4:
            image_batch[idx] = image_batch[idx].transpose((0, 3, 2, 1))

    label_batch = label_batch.transpose((0, 3, 2, 1))
    preds_batch = preds_batch.transpose((0, 3, 2, 1))

    h, w = image_batch[0].shape[1], image_batch[0].shape[2]

    n_classes = label_batch.shape[3]

    for batch_idx in range(image_batch[0].shape[0]):
        disp_images = []
        for image_tensors in image_batch:
            disp_images.append(image_tensors[batch_idx, :, :, :])
        labels = label_batch[batch_idx, :, :, :]
        preds = preds_batch[batch_idx, :, :, :]

        num_image_cols = len(disp_images)
        # Init empty grid as np array
        #  FIXME: Hard-code
        image_grid = np.zeros((4*h, n_classes*w, n_channels), dtype=np.uint8)

        # Add image(s) at the top
        for idx in range(len(disp_images)):
            if disp_images[idx].ndim == 3:
                image_grid[0:h, idx*w:(idx+1)*w, :] = disp_images[idx]
            else:
                image_grid[0:h, idx*w:(idx+1)*w, 0] = disp_images[idx]

        # Add ground truth and predicted maps
        for class_id in range(n_classes):
            image_grid[h:2*h, class_id*w:(class_id+1)*w, 0] = labels[:, :, class_id]
            image_grid[2*h:3*h, class_id*w:(class_id+1)*w, 0] = preds[:, :, class_id]
            image_grid[3*h:4*h, class_id*w:(class_id+1)*w, 0] = np.multiply(disp_images[0][:, :, 0],
                                                                            np.divide(preds[:, :, class_id],
                                                                                      255)
                                                                            )
            fname = os.path.join(result_dir, '{}_{}.{}'.format(prefix, batch_idx, fmt))

        imageio.imwrite(fname, image_grid)


def read_pickle_file(fpath):
    """
    Read a serialized pickle file

    :param fpath:
    :return:
    """
    with open(fpath, 'rb') as f:
        py_obj = pickle.load(f)
    return py_obj


def convert_to_grayscale(image, dtype=np.float32):
    """
    Maps image intensities to grayscale range (uint8)

    Parameters:
        image (numpy ndarray) : Numpy array
        dtype: (numpy dtype) Desired data type of output

    Returns:
        np.uint8 matrix that can be saved as a grayscale image

    """

    eps = 1e-5

    image = image.astype(np.float32)

    max_intensity_value = np.amax(image)
    min_intensity_value = np.amin(image)

    #  From an arbitrary range, map intensities (linearly) to a [0, 1] range
    image = (image - min_intensity_value)/(max_intensity_value - min_intensity_value + eps)

    #  Map intensities to the range of allowed grayscale values
    if dtype == np.uint8 or dtype == np.uint16:
        image = np.multiply(image, np.iinfo(dtype).max)
    else:
        image = np.multiply(image, np.iinfo(np.uint8).max)

    return np.array(image, dtype=dtype)


def save_model(model=None, optimizer=None, scheduler=None, scaler=None, n_iter=None, n_iter_val=None, epoch=None, checkpoint_dir=None, suffix=None):
    """
    Function save the PyTorch model along with optimizer state

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be saved
        optimizer (torch.optim object) : Optimizer used to train the model
        epoch (int) : Epoch the model was saved in
        path (str or Path object) : Path to directory where model parameters will be saved
        suffix (str): Suffix string to prevent new save from overwriting old ones

    Returns:
        None

    """

    if model is None:
        print('Save operation failed because model object is undefined')
        return

    if optimizer is None:
        print('Save operation failed because optimizer is undefined')
        return

    if scaler is None:
        scaler_state_dict = None
    else:
        scaler_state_dict = scaler.state_dict()

    if scheduler is None:
        save_dict = {'n_iter': n_iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': None,
                     'scaler': scaler_state_dict,
                     'n_iter_val': n_iter_val,
                     'epoch': epoch
                    }
    else:
        save_dict = {'n_iter': n_iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(),
                     'scaler': scaler_state_dict,
                     'n_iter_val': n_iter_val,
                     'epoch': epoch
                     }

    #  Overwrite existing checkpoint file to avoid running out of memory
    if suffix is None:
        fname = 'checkpoint.pt'
    else:
        fname = 'checkpoint_{}.pt'.format(suffix)


    save_path = os.path.join(checkpoint_dir, fname)

    torch.save(save_dict, save_path)


def is_perfect_square(n):
    """
    Function to check if a number is a perfect square

    :param n: (int)
    :return: is_perfect_square: (bool)

    """
    return sqrt(n).is_integer()


def load_model(model=None, optimizer=None, scheduler=None, scaler=None, checkpoint_dir=None, training=False, to_load=-1, suffix=-1, device='cpu'):
    """
    Function to load the PyTorch model

    Parameters:
        model (torch.nn.Module object) : Pytorch model whose parameters are to be loaded
        checkpoint_dir: (str) Path to checkpoint directory
        optimizer: (torch.optim) Pytorch optimizer (used to resume training)
        scheduler: (torch.lr_scheduler) Pytorch learning scheduler
        training (bool) : Set to true if training is to be resumed. Set to False if inference is to be performed
        to_load: (int) Epoch to load


    Returns:
        model (torch.nn.Module): Model object with after loading the trainable parameters
        epoch (int) : Epoch where the model was frozen. None if training is false
        checkpoint(Python dictionary) : Dictionary that saves the state

    """
    if suffix < 0:
        checkpoint_file_list = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        checkpoint_file_path, _ = select_checkpoint(checkpoint_file_list)
        if checkpoint_file_path is None:
            # If only single checkpoint saved
            checkpoint_file_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

    else:
        checkpoint_file_path = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_{}.pt'.format(suffix)))[0]


    # Since the directory contains only the last checkpoint, there will only be a single .pt file
    #checkpoint_file_path, last_iter = select_checkpoint(checkpoint_file_paths, to_load=to_load)

    checkpoint = torch.load(checkpoint_file_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if training is True:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler = None
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            scaler = None
        n_iter = checkpoint['n_iter']
        n_iter_val = checkpoint['n_iter_val']
        epoch = checkpoint['epoch']
        model.train()
    else:
        n_iter = checkpoint['n_iter']
        n_iter_val = None
        epoch = checkpoint['epoch']
        model.eval()

    print('Load checkpoint for Epoch : {}, Iteration : {}'.format(epoch, n_iter))

    load_dict = {'model': model,
                 'optimizer': optimizer,
                 'scheduler': scheduler,
                 'n_iter': n_iter,
                 'n_iter_val': n_iter_val,
                 'epoch': epoch}

    return load_dict


def find_sub_dirs(dir_path=None):
    """
    Find sub-directories (depth=1) in a given directory

    Args:
        dir_path: (path) Path to directory where sub-directories need to be found
    Returns:
        sub_dirs: (list) List of sub-dirs names (not full paths)

    """
    sub_dirs = [sub_dir for sub_dir in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, sub_dir))]
    return sub_dirs


def select_checkpoint(file_list, to_load=-1):
    """
    Given a list of checkpoint files, selects the
    most recent checkpoint. The checkpoints are saved
    in the following format : PATH/checkpoint_epoch_<epoch_id>

    Parameters:
        file_list (Pyton list) : List of file paths to different checkpoint files
        to_load: (int) Epoch to load

    Returns:
        last_checkpoint_file (str or Path object) : Path to the last saved checkpoint

    """
    try:
        iters = [int(fname.split('/')[-1].split('.')[0].split('_')[-1]) for fname in file_list]
        last_iter = max(iters)
        if to_load < 0:
            index_checkpoint = iters.index(last_iter)
        else:
            if to_load > last_epoch:
                raise ValueError('Iteration to load ({}) is greater than the last saved iteration ({}). Load failed'.
                                 format(to_load,
                                        last_iter))

            index_checkpoint = iters.index(to_load)

        return file_list[index_checkpoint], last_iter
    except ValueError:
        return None, None


def clean_checkpoint_directory(checkpoint_dir=None):
    """
    Cleaning up the checkpoint directory after model is trained. Removes all but the most recently saved
    checkpoint file

    :param checkpoint_dir: (str)
    :return: None
    """

    checkpoint_file_paths = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    most_recent_checkpoint_file_path = select_last_checkpoint(checkpoint_file_paths)

    #  Remove most recent checkpoint from deletion list
    del checkpoint_file_paths[checkpoint_file_paths.index(most_recent_checkpoint_file_path)]

    for ckpt_file_path in checkpoint_file_paths:
        os.remove(ckpt_file_path)


def calculate_total_gradient_norm(parameters):
    """
    Debug function, used to calculate the gradient norm
    during training to check effectiveness

    Parameters:
        parameters(List of torch.nn.parameter.Parameter objects) : Parameters of the model being analyzed

    Returns:
        grad_norm(float) : L2 norm of the gradients

    """
    with torch.no_grad():
        grads = []
        for p in parameters:
            if p.requires_grad is True:
                grads.append(p.grad.flatten().tolist())

        flattened_list = [grad_value for layer in grads for grad_value in layer]
        grad_norm = np.linalg.norm(np.array(flattened_list,dtype=np.float32))
        return grad_norm


def threshold_predictions(preds):
    """
    Creates thresholded binary images from network
    predictions

    Parameters:
        preds (Torch tensor) : Tensor of softmax outputs (N x C x H x W)

    Returns:
        thresh_preds (Torch tensor) : Tensor of thresholded outputs (N x C x H x W)

    """

    thresh_preds = np.where(preds >= 0.5, 1.0, 0.0)
    return thresh_preds


def save_prediction_heatmaps(result_dir=None,
                             preds= None,
                             prefix=None,
                             gpu_id=-1,
                             fmt='png',
                             train=True,
                             patient_ids=None,
                             scan_ids=None,
                             slice_idxs=None):
    """
    Visualize probabilty heatmaps of model prediction instead of the thresholded outputs

    Parameters:
        result_dir (str or Path object) : Directory to store the images
        preds (Torch tensor) :  Batch of class probability masks
        prefix (str) : File name prefix
        gpu_id (int) : gpu_id < 0 => CPU used
        fmt (str) : Format to store the image (png by default)

    """
    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)

    if gpu_id >= 0:
        preds = preds.cpu().numpy()
    else:
        preds = preds.numpy()

    labels = ['Background', 'Liver', 'Right Kidney', 'Left Kidney', 'Spleen']

    batch_size = preds.shape[0]
    num_classes = preds.shape[1]

    for idx in range(batch_size):
        for c in range(num_classes):
            prob_map = preds[idx, c, :, :]
            heat_map = sns.heatmap(data=prob_map, vmax=1, vmin=0)
            if train is True:
                fname = '{}_{}_{}.{}'.format(prefix, idx, labels[c], fmt)
            else:
                fname = 'hmap_{}_{}_{}_{}.{}'.format(patient_ids[idx], scan_ids[idx], slice_idxs[idx], labels[c],
                                                     fmt)

            save_path = os.path.join(result_dir, fname)
            fig = heat_map.get_figure()
            fig.savefig(save_path)
            plt.close(fig)


def choose_display_channel(images=None, display_channel=5):
    """
    Utility function to pick out a single time-point from the
    DCE MR series to display

    :param images (numpy ndarray): batch_size x 6 x 256 x 256
    :param display_channel: (int) channels to display
    :return: images (numpy ndarray): batch_size x 256 x 256
    """

    images = torch.transpose(input=images, dim0=0, dim1=1)
    images = images[display_channel, :, :, :]
    bt_sz, h, w = images.shape
    images = torch.reshape(images, (bt_sz, 1, h, w))
    return images


def create_roi_image(image=None, roi_coordinates=None, display=False, order=1):
    """
    Function to resample image containing only the ROI

    :param image: (numpy ndarray) 3-D H x W x num_slices
    :param roi_coordinates: (tuple) Contains the bounding box co-ordinates for the ROI
    :param display: (bool) Flag to indicate if the ROI needs to be displayed
    :param order: (int) Order of interpolation during resizing (Recommended: 1 for images, 0 for labels)
    :return roi_image: 3-D H x W x num_slices
    """

    if image.ndim != 3:
        raise RuntimeError('Expected a 3D image, actual shape : {}'.format(image.shape))

    h, w, num_slices = image.shape

    roi_image = image[roi_coordinates[0]:min(roi_coordinates[1], h), roi_coordinates[2]:min(roi_coordinates[3], w), :]
    roi_image = resize(roi_image, output_shape=(h, w), order=order)

    if display is True:
        roi_image = convert_to_grayscale(roi_image)

    return roi_image


def find_roi_coordinates(seg_mask=None):
    """
    Find the extreme points for the ROI (liver) using the segmentation mask

    :param seg_mask: (numpy ndarray) H x W x num_slices
    :return:
    """
    h, w, num_slices = seg_mask.shape
    roi_volume = np.nonzero(seg_mask)

    if roi_volume[1].shape[0] != 0:
        max_x_coordinate = np.amax(roi_volume[1])
        min_x_coordinate = np.amin(roi_volume[1])
    else:
        max_x_coordinate = w
        min_x_coordinate = 0

    if roi_volume[0].shape[0] != 0:
        max_y_coordinate = np.amax(roi_volume[0])
        min_y_coordinate = np.amin(roi_volume[0])
    else:
        max_y_coordinate = h
        min_y_coordinate = 0

    return min_y_coordinate, max_y_coordinate, min_x_coordinate, max_x_coordinate

