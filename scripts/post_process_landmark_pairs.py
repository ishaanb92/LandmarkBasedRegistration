"""

Script to filter outliers from landmark pairs so that the deformation defined the point pairs is smooth and plausible.
We do this by using a smoothing term > 0 for the thin-plate spline used to estimate the deformation defined by the
landmark correspondences

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""


import os
import numpy as np
from lesionmatching.util_scripts.utils import *
from lesionmatching.util_scripts.image_utils import *
from argparse import ArgumentParser
import shutil
import SimpleITK as sitk
from lesionmatching.data.deformations import transform_grid
from scipy.interpolate import RBFInterpolator
import joblib

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--landmarks_dir', type=str, required=True)
    parser.add_argument('--smoothing_terms', type=float, help='Smoothing terms', nargs='+')
    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.landmarks_dir) if f.is_dir()]


    for pdir in pdirs:

        pid = pdir.split(os.sep)[-1]

        # Pre-process the landmark correspondences
        fixed_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'fixed_image.mha'))

        moving_image_itk = sitk.ReadImage(os.path.join(pdir,
                                                      'moving_image.mha'))

        moving_image_np = convert_itk_to_ras_numpy(moving_image_itk)

        fixed_image_shape = fixed_image_itk.GetSize()
        moving_image_shape = moving_image_itk.GetSize()

        fixed_points_arr = parse_points_file(os.path.join(pdir,
                                                          'fixed_landmarks_elx.txt'))

        moving_points_arr = parse_points_file(os.path.join(pdir,
                                                           'moving_landmarks_elx.txt'))


        # Convert world coordinates to voxel coordinates
        fixed_points_arr = map_world_coord_to_voxel_index(world_coords=fixed_points_arr,
                                                          spacing=fixed_image_itk.GetSpacing(),
                                                          origin=fixed_image_itk.GetOrigin())

        moving_points_arr = map_world_coord_to_voxel_index(world_coords=moving_points_arr,
                                                           spacing=moving_image_itk.GetSpacing(),
                                                           origin=moving_image_itk.GetOrigin())

        # Scale the coordinates between [0, 1]
        fixed_points_scaled = np.divide(fixed_points_arr,
                                        np.expand_dims(np.array(fixed_image_shape),
                                                       axis=0))

        moving_points_scaled = np.divide(moving_points_arr,
                                         np.expand_dims(np.array(moving_image_shape),
                                                        axis=0))

        for sterm in args.smoothing_terms:
            print('Post-processing landmarks :: Patient {} Smoothing {}'.format(pid, sterm))

            # 1. Estimate TPS-based deformation using landmarks corr.
            tps_interpolator = RBFInterpolator(y=fixed_points_scaled,
                                               d=moving_points_scaled,
                                               smoothing=sterm,
                                               kernel='thin_plate_spline',
                                               degree=1)

            # 2. Compute updated landmark correspondences using tps
            updated_moving_points = tps_interpolator(fixed_points_scaled)

            # Sanity: If lambda = 0 => perfect* interpolation
            if sterm == 0:
                assert(np.allclose(updated_moving_points,
                                   moving_points_scaled))
                continue # Don't save

            # 3. Rescale moving landmarks to to image size
            updated_moving_points_rescaled = np.multiply(updated_moving_points,
                                                         np.expand_dims(np.array(moving_image_shape),
                                                                        axis=0))

            # 4. Convert the landmarks back to physical/world coordinates
            updated_moving_points_rescaled_world = map_voxel_index_to_world_coord(voxels=updated_moving_points_rescaled,
                                                                                  spacing=moving_image_itk.GetSpacing(),
                                                                                  origin=moving_image_itk.GetOrigin())

            # 5. Save the updated landmark correspondences
            create_landmarks_file(landmarks=updated_moving_points_rescaled_world,
                                  world=True,
                                  fname=os.path.join(pdir, 'moving_landmarks_elx_{}.txt'.format(sterm)))

            # 6. Save the TPS interpolator object
            joblib.dump(tps_interpolator,
                        os.path.join(pdir, 'tps_{}.pkl'.format(sterm)))




