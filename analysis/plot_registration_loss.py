"""

Plot registration loss to gain an insight into the role of design choices (presence of landmarks, cost function weights) play in the optimization process

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

iterations = [0, 1, 2] # 0: Affine 1: B-Spline-1 2: B-Spline-2
resolutions = [0, 1, 2, 3, 4] # Each registration stage operates over 4 resolutins

ANNOTATION_Y = -2
ANNOTATION_Y_LM = 20

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--reg_dir', type=str, required=True)
    args = parser.parse_args()

    pdirs = [f.path for f in os.scandir(args.reg_dir) if f.is_dir()]

    for pdir in pdirs:
        metric = []
        transform_bending_penalty = []
        landmarks_cost_function = []
        end_points = {} # Dictionary to store end points (used for plotting)
        end_points['Affine'] = {}
        end_points['BSpline-1'] = {}
        end_points['BSpline-2'] = {}
        start_points = {}
        start_points['Affine'] = {}
        start_points['BSpline-1'] = {}
        start_points['BSpline-2'] = {}

        for stage in iterations:
            for res in resolutions:

                if stage == 0:
                    start_points['Affine']['R{}'.format(res)] = len(metric)
                elif stage == 1:
                    start_points['BSpline-1']['R{}'.format(res)] = len(metric)
                elif stage == 2:
                    start_points['BSpline-2']['R{}'.format(res)] = len(metric)

                elastix_log = pd.read_csv(os.path.join(pdir, 'IterationInfo.{}.R{}.txt'.format(stage,
                                                                                               res)),
                                          sep='\t')

                total_loss = elastix_log['2:Metric'].to_numpy().astype(np.float32)
                metric.extend(list(total_loss))

                # Store the "end-points" for each stage/resolution
                if stage == 0:
                    end_points['Affine']['R{}'.format(res)] = len(metric)
                elif stage == 1:
                    end_points['BSpline-1']['R{}'.format(res)] = len(metric)
                elif stage == 2:
                    end_points['BSpline-2']['R{}'.format(res)] = len(metric)

                if stage == 0: # Affine (=> No transform bending loss)
                    try:
                        landmark_loss = elastix_log['2:Metric1'].to_numpy().astype(np.float32)
                        landmarks_cost_function.extend(landmark_loss)
                    except KeyError:
                        print('Landmarks not used for Affine regsitration')
                        continue
                else: # B-Spline (Landmarks loss is Metric2)
                    try:
                        landmark_loss = elastix_log['2:Metric2'].to_numpy().astype(np.float32)
                        landmarks_cost_function.extend(landmark_loss)
                    except KeyError:
                        print('Landmarks not used for B-Spline registration')
                        continue

        fig, ax = plt.subplots(1, 2,
                               figsize=(10, 8))

        # Plotting the main metric
        ax[0].plot(np.arange(len(metric)),np.array(metric))

        ax[0].set_ylabel('Total Metric')
        ax[0].set_xlabel('Iterations')

        ax[0].set_ylim((-3, 3))

        # Draw the "boundaries" between different stages and resolutions
        # Start lines
        offset = 0
        for reg_stage in start_points.keys():
            spoint = start_points[reg_stage]['R0']
            ax[0].axvline(x=spoint,
                       linestyle='--')

            ax[0].annotate('{} Start'.format(reg_stage),
                        xy=(spoint, ANNOTATION_Y+offset))
            offset += 1


        if len(landmarks_cost_function) != 0:
            # Plotting the landmark metric
            ax[1].plot(np.arange(len(landmarks_cost_function)), np.array(landmarks_cost_function))

            ax[1].set_ylabel('CorrespondingPointsEuclideanDistanceMetric')
            ax[1].set_xlabel('Iterations')

            ax[1].set_ylim((0, 50))

            # Draw the "boundaries" between different stages and resolutions
            # Start lines
            offset = 0
            for reg_stage in start_points.keys():
                spoint = start_points[reg_stage]['R0']
                if spoint <= len(landmarks_cost_function):
                    ax[1].axvline(x=spoint,
                                  linestyle='--')

                    ax[1].annotate('{} Start'.format(reg_stage),
                                   xy=(spoint, ANNOTATION_Y_LM+offset))
                    offset += 4

        # Save the plot
        fig.savefig(os.path.join(pdir, 'reg_loss.png'),
                    bbox_inches='tight')

        plt.close()




