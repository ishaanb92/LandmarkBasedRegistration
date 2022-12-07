""""
Script to compute performance metrics

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import torch
import numpy as np
from utils.utils import maybe_convert_tensor_to_numpy

def get_match_statistics(gt, pred):

    gt = maybe_convert_tensor_to_numpy(gt)
    pred = maybe_convert_tensor_to_numpy(pred)

    assert(gt.ndim == 2)
    assert(pred.ndim == 2)

    # True positives
    tps = np.nonzero(gt*pred)[0].shape[0]

    # False positives
    fps = np.nonzero((1-gt)*pred)[0].shape[0]

    # False negatives
    fns = np.nonzero(gt*(1-pred))[0].shape[0]

    stats = {}
    stats['True Positives'] = tps
    stats['False Positives'] = fps
    stats['False Negatives'] = fns

    return stats



