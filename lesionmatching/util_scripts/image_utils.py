from scipy.ndimage.interpolation import zoom
from scipy.ndimage import label, find_objects, generate_binary_structure, center_of_mass, binary_opening, binary_closing
import numpy as np
import sys
import nibabel as nib
import torch
import SimpleITK as sitk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float32
import os
import shutil
import glob
import warnings

def calculate_mse(img1, img2):
    """
    Calculate mean square error between 2 images

    :param img1: (numpy ndarray) 3-D image
    :param img2: (numpy ndarray) 3-D image
    :return mse: (float) Mean square error
    """
    assert (isinstance(img1, np.ndarray))
    assert (isinstance(img2, np.ndarray))

    assert (img1.ndim == 3)
    assert (img2.ndim == 3)

    assert (img1.shape == img2.shape)

    num_pixels = img1.shape[0]*img1.shape[1]*img2.shape[2]

    sq_err = np.sum((img1.astype(np.float32) - img2.astype(np.float32))**2)

    mse = sq_err/float(num_pixels)

    return mse


def calculate_ssim(img1, img2):
    """
    Caclulate SSIM between 2 images

    :param img1: (numpy ndarray) 3-D image
    :param img2: (numpy ndarray) 3-D image
    :return ssim: (float) Mean structural similarity index (over the image)

    """
    assert (isinstance(img1, np.ndarray))
    assert (isinstance(img2, np.ndarray))
    assert (img1.ndim == 3)
    assert (img2.ndim == 3)
    assert (img1.shape == img2.shape)

    return ssim(img1.astype(np.float32), img2.astype(np.float32))


def resample_itk_image_to_new_spacing(image=None, new_spacing=None, new_size=None, interp_order=3):
    """
    Function to resample image to have desired voxel spacing. Target use to obtain make anistropic spacings to isotropic
    Refer: https://github.com/jonasteuwen/SimpleITK-examples/blob/master/examples/resample_isotropically.py

    :param image: (sitk Image)
    :param new_spacing: (tuple)
    :param new_size: (tuple)
    :param interp_order: (int) Order of interpolation
    :return: resampled_image: (sitk Image)
    """
    assert (isinstance(image, sitk.Image))

    if (new_spacing is not None) and (new_size is not None):
        raise ValueError('The new spacing and size cannot be specified together because one is computed from the other!!!')

    fill_value = int(np.amin(sitk.GetArrayFromImage(image)).astype(np.float32))

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_direction = image.GetDirection()

    if new_size is None:
        new_size = [int(round(original_size[0]*(original_spacing[0]/new_spacing[0]))),
                    int(round(original_size[1]*(original_spacing[1]/new_spacing[1]))),
                    int(round(original_size[2]*(original_spacing[2]/new_spacing[2])))]
    else:
        new_spacing = [float(original_spacing[0]*(original_size[0]/new_size[0])),
                       float(original_spacing[1]*(original_size[1]/new_size[1])),
                       float(original_spacing[2]*(original_size[2]/new_size[2]))]

    if interp_order == 0:
        sitk_interpolator = sitk.sitkNearestNeighbor
    elif interp_order == 1:
        sitk_interpolator = sitk.sitkLinear
    elif interp_order == 3:
        sitk_interpolator = sitk.sitkBSpline
    else:
        raise RuntimeError('Interpolator order {} not supported'.format(interp_order))

    new_spacing = [float(s) for s in new_spacing]

    resampled_image = sitk.Resample(image,
                                    new_size,
                                    sitk.Transform(),
                                    sitk_interpolator,
                                    image.GetOrigin(),
                                    new_spacing,
                                    image.GetDirection(),
                                    fill_value,
                                    image.GetPixelID())

    return resampled_image



def read_itk_image(image_path):
    """
    Returns image + array (with axis re-ordered)
    Args:
        image_path: (str) Path to image

    Returns:
        itk_img: (sitk Image)
        np_img: (numpy ndarray)

    """
    itk_img = sitk.ReadImage(image_path)
    np_img = sitk.GetArrayFromImage(itk_img)
    if np_img.ndim == 4:
        np_img = np_img.transpose((2, 3, 1, 0)) # H x W x D x C
    elif np_img.ndim == 3:
        np_img = np_img.transpose((1, 2, 0)) # H x W x D
    else:
        raise RuntimeError('Image with {} dimensions not supported'.format(np_img.ndim))
    return itk_img, np_img


def create_itk_image(np_img):
    assert (isinstance(np_img, np.ndarray))
    if np_img.ndim == 3:
        np_img = np_img.transpose((2, 0, 1))
    elif np_img.ndim == 4:
        np_img = np_img.transpose((3, 2, 0, 1))

    itk_img = sitk.GetImageFromArray(np_img)
    return itk_img


def copy_image_metadata(img, spacing, origin, direction):
    """
    Function to copy image metadata to the new ITK image created from
    the processed pixel array when dimensionality is changed (4D -> 3D)
    In case both the src and target images are of the same dimension, it is better
    to use the sitk.CopyInformation() to copy meta-data

    :param img: (ITK image)
    :param spacing: Spacing information of the image
    :param origin:
    :param direction:
    :return img: (ITK image)

    """
    assert (isinstance(img, sitk.Image))
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    return img


def calculate_new_size(image_shape=tuple([]), old_spacing = tuple([]), new_spacing=tuple([]), verbose=False):
    """
    Calculate new Z, X, Y dimensions after resampling

    :param image_shape: (numpy ndarray)
    :param old_spacing: (tuple)
    :param new_spacing: (Python tuple)
    :param verbose:
    :return: new_size: (numpy ndarray)
    :return: eff_spacing: (numpy ndarray)
    """
    if isinstance(new_spacing, tuple) is False:
        raise ValueError("Please provide the new_spacing argument as a tuple.")

    assert isinstance(old_spacing, tuple)

    assert (isinstance(image_shape, tuple))

    assert (len(old_spacing) == len(new_spacing) == 3)

    resize_factor = np.divide(np.array(old_spacing), np.array(new_spacing))
    new_shape = np.array(np.round(image_shape * resize_factor), dtype=np.int32)
    eff_resize_factor = new_shape / image_shape

    #  Effective spacing, error originates from rounding of size
    eff_spacing = np.array(old_spacing) / eff_resize_factor

    if verbose is True:
        print('The effective spacing is {}'.format(eff_spacing))

    return tuple(new_shape), tuple(eff_spacing)


def perform_n4_bias_correction(image):
    """
    Perform N4 bias correction on a 3D MR Image
    Reference: https://github.com/bigbigbean/N4BiasFieldCorrection/blob/master/N4BiasFieldCorrection.py

    :param image: (sitk.Image)
    :return: bias_corrected_image: (sitk.Image)
    """
    #  Create mask image
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    inputImage = sitk.Cast(image, sitk.sitkFloat32)

    n4_corrector = sitk.N4BiasFieldCorrectionImageFilter()

    bias_corrected_image = n4_corrector.Execute(inputImage, maskImage)
    return bias_corrected_image


def write_slices_to_video(image_path=None, codec='MJPG', output_path='example_video.avi', fps=20):
    """
    Take a list of slices to make a video

    :param image_path: (str) Path to volume to be read
    :param codec: (str) Specify the codec to to create the video
    :param output_path: (str)
    :param fps: (int) Frames per second
    :return:
    """

    if sys.version_info[1] == 7:
        raise RuntimeError('OpenCV 3 is not compatible with Python 3.7.x')
    else:
        import cv2

    image, _ = read_niftee_image(filename=image_path, rearrange_axes=False)
    writer = None

    if image.ndim < 3:
        raise RuntimeError('Insufficient number of dimensions to make a video {}'.format(image.ndim))

    if image.ndim == 4:
        tnsp_image = np.transpose(image, (2, 0, 1, 3))  # Swap z-dir and color channels
    else:
        tnsp_image = np.transpose(2, 0, 1)

    n_slices = tnsp_image.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    h, w = tnsp_image.shape[1:3]

    for slice in range(n_slices):
        if writer is None:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (h, w), True)
        writer.write(tnsp_image[slice])

    cv2.destroyAllWindows()
    writer.release()


def find_number_of_objects(mask):
    """
    Function to find the number of lesions in a given mask, but can also be used to find connected components
    in a binary nd-image

    :param mask: (numpy ndarray) Binary 3D mask
    :return: num_objects: (int) Number of objects found in the mask
    :return: list_of_objects: (list) List of coordinate tuples containing each object

    """

    assert (mask.ndim == 3)

    struct_element = generate_binary_structure(rank=mask.ndim, connectivity=4)

    labelled_mask, num_objects = label(mask, struct_element)


    list_of_centers = center_of_mass(input=mask,
                                     labels=labelled_mask,
                                     index=np.arange(1, num_objects+1))

    assert (len(list_of_centers) == num_objects)

    return num_objects, list_of_centers


def return_lesion_coordinates(mask):
    """
    Function to return volumes where lesion (objects) in mask are present
    These coordinates define the minimum parallelopied that contains the object

    """
    assert(isinstance(mask, np.ndarray))
    assert(mask.ndim == 3)
    struct_element = generate_binary_structure(rank=mask.ndim, connectivity=4)
    labelled_mask, num_objects = label(input=mask, structure=struct_element)

    lesion_slices = find_objects(input=labelled_mask)
    return lesion_slices, num_objects


# Post-Processing Functions
def post_process_predicted_mask(pred_mask=None):
    """
    Post-processing for the predicted (binary) mask
    Hole-filling followed by binary opening to remove holes within
    the segmented area and retain only the largest connected component in the
    foreground

    :param pred_mask: (numpy ndarray)Predicted Mask
    :return proc_mask: (numpy ndarray) Processed Mask

    """

    assert(pred_mask.dtype == np.uint8)

    # Full connectivity in 3x3x3 neighborhood
    struct_elem_closing = generate_binary_structure(rank=pred_mask.ndim, connectivity=4)
    struct_elem_opening = generate_binary_structure(rank=pred_mask.ndim, connectivity=1)
    #struct_elem_closing = np.ones((3, 3, 3), dtype=np.uint8)
    #struct_elem_opening = np.expand_dims(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8), axis=2)

    # Fill gaps
    pred_mask = binary_closing(input=pred_mask, structure=struct_elem_closing, iterations=1).astype(np.uint8)

    # Remove noise
    post_processed_pred_mask = binary_opening(input=pred_mask, structure=struct_elem_opening, iterations=1).astype(np.uint8)

    return post_processed_pred_mask


def denoise_mr_volume(image=None, verbose=False, fast_mode=False, patch_size=7, patch_distance=21):
    """
    Function to denoise the MR volume using the non-local means algorithm
    More about the skimage implementation: https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf
    Notes about usage: https://scikit-image.org/docs/dev/api/skimage.restoration.html?highlight=skimage%20restoration#denoise-nl-means

    :param image: (numpy ndarray) 3D (noisy) image H x W x #slices
    :param verbose: (bool) Set to True to enable prints for debug/analysis
    :param fast_mode: (bool) If true, the fast_mode flag for the skimage implementation is set to True. See 'Notes about
                             usage' for details.
    :param patch_size: (int) Patch size for neighbourhood
    :param patch_distance: (int) Range within which patches should be considered
    :return: denoised_image: (numpy ndarray) 3D de-noised image H x W x #slices
    """
    assert (isinstance(image, np.ndarray))
    assert (image.ndim == 3)

    if image.dtype == np.uint8 or image.dtype == np.uint16:
        image = img_as_float32(image=image)

    # Estimate noise in image
    noise_var = estimate_sigma(image=image)
    if fast_mode is True:
        h = 0.8 * noise_var
    else:
        h = 0.6 * noise_var

    if verbose is True:
        print('Estimated (Gaussian) noise variance in image = {}'.format(noise_var))
        print('Fast Mode : {} => h = {}'.format(fast_mode, h))

    denoised_image = denoise_nl_means(image=image,
                                      patch_size=patch_size,
                                      patch_distance=patch_distance,
                                      fast_mode=fast_mode,
                                      h=h,
                                      multichannel=False,
                                      sigma=noise_var)
    return denoised_image


def save_ras_as_itk(img=None,
                    metadata=None,
                    fname=None):

    if isinstance(img, torch.Tensor):
        img = img.numpy()

    if img.ndim == 4:
        img = np.squeeze(img, axis=0)

    # Get from RAS -> Std numpy axes ordering (ZYX)
    img = np.transpose(img, (2, 1, 0))
    img_itk = sitk.GetImageFromArray(img)
    img_itk.SetOrigin(metadata['origin'])
    img_itk.SetSpacing(metadata['spacing'])
    img_itk.SetDirection(metadata['direction'])

    sitk.WriteImage(img_itk,
                    fname)

def convert_itk_to_ras_numpy(image):

    assert(isinstance(image, sitk.Image))

    im_np = sitk.GetArrayFromImage(image)

    # Convert to RAS axis ordering : [z, y, x] -> [x, y, z]
    im_np = np.transpose(im_np, (2, 1, 0))

    return im_np.astype(np.float32)


def dry_sponge_augmentation(image, jac_det):
    """

    This is a model that models lung progression between inhale and exhale
    assuming mass preservation (Staring et al., 2014). This is used in Sokooti et al. (2019) as
    an intensity augmentation technique.

        I(x) = I(x)[J(x)^(-1)]

    See: https://github.com/hsokooti/RegNet/blob/master/functions/artificial_generation/intensity_augmentation.py

    """

    # Bound the changes in volume
    jac_det[jac_det < 0.7] = 0.7
    jac_det[jac_det > 1.3] = 1.3

    random_perturb = np.random.uniform(0.9, 1.1)
    image_sponge = torch.div(image, jac_det*random_perturb)

    return image_sponge


def get_min_max_from_image(image:sitk.Image,
                           mask:sitk.Image):

    image_np = convert_itk_to_ras_numpy(image)
    mask_np = convert_itk_to_ras_numpy(mask)


    lung_max = np.amax(image_np[np.where(mask_np==1)])
    lung_min = np.amin(image_np[np.where(mask_np==1)])

    return lung_max, lung_min


def create_separate_lesion_masks(fname):
    """
    For ease of creating ground truth matches, we split the lesion annotation into separate
    masks, each mask contains a single lesion. This is done for both fixed and moving lesion
    masks.

    Return number of lesions so that each (moving) lesion can be resampled

    """
    # Is it fixed or moving?
    mask_type = fname.split(os.sep)[-1].split('_')[0]
    lesion_mask_itk = sitk.ReadImage(fname)
    lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk)

    reg_dir = os.sep.join(fname.split(os.sep)[0:-1])

    # Weird!
    if lesion_mask_np.ndim != 3:
        return -1

    # Handle non-binary masks!!!
    if np.amax(lesion_mask_np) > 1:
        lesion_mask_np = np.where(lesion_mask_np >=1, 1, lesion_mask_np)
        lesion_mask_itk_binary = sitk.GetImageFromArray(lesion_mask_np)
        lesion_mask_itk_binary.SetOrigin(lesion_mask_itk.GetOrigin())
        lesion_mask_itk_binary.SetDirection(lesion_mask_itk.GetDirection())
        lesion_mask_itk_binary.SetSpacing(lesion_mask_itk.GetSpacing())
        # Overwrite old file
        sitk.WriteImage(lesion_mask_itk_binary, fname)
        # Convert the fixed binary ITK image to a numpy ndarray
        lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk_binary)

    lesion_slices, n_lesions = return_lesion_coordinates(lesion_mask_np)

    for idx, lesion_slice in enumerate(lesion_slices):

        # Create a new mask for a single lesion
        single_lesion_mask = np.zeros_like(lesion_mask_np)
        single_lesion_mask[lesion_slice] += lesion_mask_np[lesion_slice]

        # DEBUG -- ENSURE EACH MASK CONTAINS A SINGLE LESION
        lesion_slices_debug, n_lesions_debug = return_lesion_coordinates(single_lesion_mask)
        if n_lesions_debug != 1:
            raise RuntimeError('Lesion mask {} contains {} lesions. Ideally only 1 lesion should be present'.format(fname,
                                                                                                                    n_lesions_debug))

        single_lesion_mask_itk = sitk.GetImageFromArray(single_lesion_mask)

        # Add metadata
        single_lesion_mask_itk.SetOrigin(lesion_mask_itk.GetOrigin())
        single_lesion_mask_itk.SetDirection(lesion_mask_itk.GetDirection())
        single_lesion_mask_itk.SetSpacing(lesion_mask_itk.GetSpacing())

        lesion_dir = os.path.join(reg_dir, '{}_lesion_{}'.format(mask_type, idx))
        if os.path.exists(lesion_dir) is True:
            shutil.rmtree(lesion_dir)
        os.makedirs(lesion_dir)

        lesion_fpath = os.path.join(lesion_dir, 'lesion.nii.gz')
        sitk.WriteImage(single_lesion_mask_itk,
                        lesion_fpath)

    return n_lesions

def handle_lesion_separation_error(pat_dir):
    """
    Handle errors arising from saving each lesion as a separate mask by
    removing all folders

    """
    m_lesion_dirs = [d for d in glob.glob(os.path.join(pat_dir, 'moving_lesion_*')) if os.path.isdir(d)]
    f_lesion_dirs = [d for d in glob.glob(os.path.join(pat_dir, 'fixed_lesion_*')) if os.path.isdir(d)]

    if len(m_lesion_dirs) > 0:
        for ldir in m_lesion_dirs:
            shutil.rmtree(ldir)

    if len(f_lesion_dirs) > 0:
        for ldir in f_lesion_dirs:
            shutil.rmtree(ldir)

def merge_lesions_masks(dir_list=None):

    for idx in range(len(dir_list)):
        lesion_mask_itk = sitk.ReadImage(os.path.join(dir_list[idx], 'result.nii'))
        if idx == 0:
            lesion_mask_np = sitk.GetArrayFromImage(lesion_mask_itk)
        else:
            lesion_mask_np += sitk.GetArrayFromImage(lesion_mask_itk)

    lesion_mask_np = np.where(lesion_mask_np > 1, 1, lesion_mask_np)
    _, n_lesions = return_lesion_coordinates(lesion_mask_np)

    return lesion_mask_np


def get_lesion_slices(dir_list=None, fixed=True):

    if fixed is True:
        fname = 'lesion.nii.gz'
    else:
        fname = 'result.nii'

    slices = []

    for idx, lesion_dir in enumerate(dir_list):
        single_lesion_mask_itk = sitk.ReadImage(os.path.join(lesion_dir, fname))
        single_lesion_mask_np = sitk.GetArrayFromImage(single_lesion_mask_itk)
        lesion_slices, n_lesions = return_lesion_coordinates(single_lesion_mask_np)
        if n_lesions > 1:
            warning_str =  "Weird, there should be only one lesion present, but there are {}."\
                            " ID = {}, fixed = {}. Skip this patient".format(n_lesions, idx, fixed)
            warnings.warn(warning_str, RuntimeWarning)
            return None

        slices.append(lesion_slices[0])

    return slices
