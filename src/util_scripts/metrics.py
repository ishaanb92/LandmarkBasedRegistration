""""
Script to compute performance metrics

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import torch
import numpy as np

def get_match_statistics(gt, pred):

    assert(isinstance(gt, torch.Tensor))
    assert(isinstance(pred, torch.Tensor))

    assert(gt.ndim == 2)
    assert(pred.ndim == 2)

    # True positives
    tps = torch.nonzero(gt*pred).shape[0]

    # False positives
    fps = torch.nonzero((1-gt)*pred).shape[0]

    # False negatives
    fns = torch.nonzero(gt*(1-pred)).shape[0]

    stats = {}
    stats['True Positives'] = tps
    stats['False Positives'] = fps
    stats['False Negatives'] = fns

    return stats



