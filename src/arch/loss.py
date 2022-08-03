"""

Script to define loss function used to find correspondences

Reference:
    Paper: http://arxiv.org/abs/2001.07434
    Code: https://github.com/monikagrewal/End2EndLandmarks/blob/main/loss.py

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch
import torch.nn.functional as F



def create_ground_truth_correspondences(kpts1, kpts2, deformation):
    """

    Using the (known) deformation grid create ground truth for keypoints
    A candidate is chosen as a keypoint if it maps to a point in the transformed image
    that is also a keypoint candidate (repeatability)

    """

    device = kpts1.device

    b, z, y, x, _ = deformation.shape

    thresh = torch.Tensor([max(2/z, 2/y, 2/x)]).to(kpts1.device).type(kpts1.dtype)

    # Reshape deformations to : [B, X, Y, Z] to ensure torch conventions
    deformation = deformation.permute(0, 4, 3, 2, 1)
    kpts1_projected = F.grid_sample(deformation.to(device).type(kpts2.dtype),
                                    kpts2,
                                    align_corners=False)


    kpts1_projected = kpts1_projected.permute(0, 4, 2, 3, 1)

    kpts1_projected = kpts1_projected.squeeze(dim=-2).squeeze(dim=-2)
    kpts1 = kpts1.squeeze(dim=1).squeeze(dim=1)

    # Add fake channel axes to different dims so that we get a 'cross' matrix
    kpts1_projected = kpts1_projected.unsqueeze(dim=1)
    kpts1 = kpts1.unsqueeze(dim=2)

    cell_distances = torch.norm(kpts1 - kpts1_projected, dim=3)

    # two-way (bruteforce) matching
    min_cell_distances_row = torch.min(cell_distances, dim=1)[0].view(b, 1, -1)
    min_cell_distances_col = torch.min(cell_distances, dim=2)[0].view(b, -1, 1)

    # Forward mask
    s1 = torch.eq(cell_distances, min_cell_distances_row)
    # Backward mask
    s2 = torch.eq(cell_distances, min_cell_distances_col)

    # Match = 1 => Bidirectionally minimal distances + threshold satisfaction
    matches = s1 * s2 * torch.ge(thresh, cell_distances)  #b, k, k
    # Bool -> numeric dtype
    matches = matches.type(kpts1.dtype)

    indices = torch.nonzero(matches)

    gt1 = torch.zeros(b, matches.shape[1], dtype=torch.float).to(device)
    gt2 = torch.zeros(b, matches.shape[2], dtype=torch.float).to(device)

    gt1[indices[:, 0], indices[:, 1]] = 1.
    gt2[indices[:, 0], indices[:, 2]] = 1.

    return gt1, gt2, matches



def custom_loss(landmark_logits1, landmark_logits2, desc_pairs_score, desc_pairs_norm, gt1, gt2, match_target, k, device="cuda:0"):

    # LandmarkProbabilityLoss Image 1
    landmark_logits1_lossa = torch.mean(torch.tensor(1.).to(device) - torch.sum(landmark_logits1, dim=(1)) / torch.tensor(float(k)).to(device))
    landmark_logits1_lossb = F.binary_cross_entropy_with_logits(landmark_logits1, gt1)
    landmark_logits1_loss = landmark_logits1_lossa + landmark_logits1_lossb

    # LandmarkProbabilityLoss Image 2
    landmark_logits2_lossa = torch.mean(torch.tensor(1.).to(device) - torch.sum(landmark_logits2, dim=(1)) / torch.tensor(float(k)).to(device))
    landmark_logits2_lossb =	F.binary_cross_entropy_with_logits(landmark_logits2, gt2)
    landmark_logits2_loss = landmark_logits2_lossa + landmark_logits2_lossa

    # Descriptor loss: Weighted CE
    b, k1, k2 = match_target.shape
    wt = float(k) / float(k)** 2
    desc_loss1 = F.cross_entropy(desc_pairs_score,
                                 match_target.long().view(-1),
                                 weight=torch.tensor([wt, 1 - wt]).to(device))

    # Descriptor loss: Contrastive loss
    Npos = match_target.sum()
    Nneg = b*k1*k2 - Npos
    pos_loss = torch.sum(match_target * torch.max(torch.zeros_like(desc_pairs_norm).to(device), desc_pairs_norm - 0.1)) / (2*Npos + 1e-6)
    neg_loss = torch.sum((1.0 - match_target) * torch.max(torch.zeros_like(desc_pairs_norm).to(device), 1.0 - desc_pairs_norm)) / (2*Nneg + 1e-6)
    desc_loss2 = pos_loss + neg_loss
    desc_loss = desc_loss1 + desc_loss2

    # total loss
    loss = landmark_logits1_loss + landmark_logits2_loss + desc_loss

    return loss



