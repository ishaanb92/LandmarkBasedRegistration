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


def convert_grid_to_image_coords(pts, shape=(64, 128, 128)):
    """
    Convert grid points ([-1, 1] range) to image coords

    """

    # Scale to [0, 1] range
    pts = (pts + 1.)/2.

    # Scale to image dimensions (DHW ordering)
    pts = pts * torch.Tensor([(shape[0]-1, shape[1]-1, shape[2]-1)]).view(1, 1, 3).int().to(pts.device)

    return pts

def create_ground_truth_correspondences(kpts1, kpts2, deformation, pixel_thresh=(2, 4, 4), train=True):
    """

    Using the (known) deformation grid create ground truth for keypoints
    A candidate is chosen as a keypoint if it maps to a point in the transformed image
    that is also a keypoint candidate (repeatability)

    pixel_thresh :  (x_thresh, y_thresh, z_thresh)

    """

    device = kpts1.device

    assert(kpts1, kpts2)


    b, i, j, k, _ = deformation.shape

    # The grid values are in the [-1, 1] => a single grid spacing = (2/k, 2/j, 2/i)
    # So we compute the grid thresh that corresponds to the user-supplies pixel thresh
    grid_thresh = torch.Tensor([pixel_thresh[0]*(2/k), pixel_thresh[1]*(2/j), pixel_thresh[2]*(2/i)])


    deformation = deformation.permute(0, 4, 1, 2, 3) # [B, 3, H, W, D]

    # Deformation grid shape: [B, 3, H, W, D]
    # kpts2 (grid) : [B, 1, 1, K, 3]
    # kpts2[:, :, :, :, 0] : x (D)
    # kpts2[:, :, :, :, 1] : y (W)
    # kpts2[:, :, :, :, 2] : z (H)
    kpts1_projected = F.grid_sample(deformation.to(device).type(kpts2.dtype),
                                    kpts2,
                                    align_corners=True,
                                    mode='bilinear')

    kpts1_projected = kpts1_projected.permute(0, 4, 2, 3, 1)


    kpts1_projected = kpts1_projected.squeeze(dim=-2).squeeze(dim=-2)
    kpts1 = kpts1.squeeze(dim=1).squeeze(dim=1)

    # Add fake channel axes to different dims so that we get a 'cross' matrix
    kpts1_projected = kpts1_projected.unsqueeze(dim=1)
    kpts1 = kpts1.unsqueeze(dim=2)

    # Kpts shape : [B, K, 1, 3]
    # Projected Kpts shape : [B, 1, K, 3]
    # To decide whether a keypoint pair matches, we check whether the projected keypoints
    # lie inside an ellipsoid (axis lengths depend on the pixel threshold)
    ellipsoid_values = torch.pow((kpts1_projected[:, :, :, 0] - kpts1[:, :, :, 0]), 2)/torch.pow(grid_thresh[0], 2) + \
                       torch.pow((kpts1_projected[:, :, :, 1] - kpts1[:, :, :, 1]), 2)/torch.pow(grid_thresh[1], 2) + \
                       torch.pow((kpts1_projected[:, :, :, 2] - kpts1[:, :, :, 2]), 2)/torch.pow(grid_thresh[2], 2)


    # two-way (bruteforce) matching
    min_ellipsoid_row = torch.min(ellipsoid_values, dim=1)[0].view(b, 1, -1)
    min_ellipsoid_col = torch.min(ellipsoid_values, dim=2)[0].view(b, -1, 1)

    # Forward mask
    s1 = torch.eq(ellipsoid_values, min_ellipsoid_row)
    # Backward mask
    s2 = torch.eq(ellipsoid_values, min_ellipsoid_col)

    # If ellipsoid equation for a given pair is <=1 => The pair is a match!
    matches = s1 * s2 * torch.ge(torch.ones_like(ellipsoid_values),
                                 ellipsoid_values)  #b, k1, k2
    # Bool -> numeric dtype
    matches = matches.type(kpts1.dtype)

    indices = torch.nonzero(matches)
    num_matches = indices.shape[0]

    gt1 = torch.zeros(b, matches.shape[1], dtype=torch.float).to(device)
    gt2 = torch.zeros(b, matches.shape[2], dtype=torch.float).to(device)

    gt1[indices[:, 0], indices[:, 1]] = 1.
    gt2[indices[:, 0], indices[:, 2]] = 1.

    if train is True:
        return gt1, gt2, matches, num_matches
    else:
        # Shape: [B, K, 3]
        kpts1_projected = torch.squeeze(kpts1_projected, dim=1)
        landmarks_projected = convert_grid_to_image_coords(kpts1_projected,
                                                           shape=(k, j, i))

        return gt1, gt2, matches, num_matches, landmarks_projected




def custom_loss(landmark_logits1,
                landmark_logits2,
                desc_pairs_score,
                desc_pairs_norm,
                gt1,
                gt2,
                match_target,
                k,
                device="cuda:0",
                mask_idxs_1=None,
                mask_idxs_2=None,
                desc_loss_comp_wt=torch.Tensor([1.0, 1.0])):

    # LandmarkProbabilityLoss Image 1
    # Mean over batch (first torch.mean from left) and over #landmarks (=K) (first torch.mean from right)

    # Boost landmark prob. for points inside lung
    landmark_logits1_lossa_inside = torch.mean(1 - torch.mean(torch.mul(mask_idxs_1,
                                                                 torch.sigmoid(landmark_logits1)),
                                               dim=1))

    # Suppress landmark prob. for points outside lung
    landmark_logits1_lossa_outside = torch.mean(torch.mean(torch.mul(1-mask_idxs_1,
                                                                     torch.sigmoid(landmark_logits1)),
                                               dim=1))

    landmark_logits1_lossa = landmark_logits1_lossa_inside + landmark_logits1_lossa_outside

    landmark_logits1_lossb = F.binary_cross_entropy_with_logits(landmark_logits1, gt1, reduction='none')
    landmark_logits1_lossb = torch.mean(landmark_logits1_lossb, dim=1) # Mean over #landmarks
    landmark_logits1_lossb = torch.mean(landmark_logits1_lossb, dim=0) # Mean over batch

    # Sum the losses!
    landmark_logits1_loss = landmark_logits1_lossa + landmark_logits1_lossb

    # LandmarkProbabilityLoss Image 2
    landmark_logits2_lossa_inside = torch.mean(1 - torch.mean(torch.mul(mask_idxs_2,
                                                                 torch.sigmoid(landmark_logits2)),
                                               dim=1))

    # Suppress landmark prob. for points outside lung
    landmark_logits2_lossa_outside = torch.mean(torch.mean(torch.mul(1-mask_idxs_2,
                                                                     torch.sigmoid(landmark_logits2)),
                                               dim=1))

    landmark_logits2_lossa = landmark_logits2_lossa_inside + landmark_logits2_lossa_outside

    landmark_logits2_lossb = F.binary_cross_entropy_with_logits(landmark_logits2, gt2, reduction='none')
    landmark_logits2_lossb = torch.mean(landmark_logits2_lossb, dim=1) # Mean over #landmarks
    landmark_logits2_lossb = torch.mean(landmark_logits2_lossb, dim=0) # Mean over batch

    landmark_logits2_loss = landmark_logits2_lossa + landmark_logits2_lossb

    # Descriptor loss: Weighted CE
    b, k1, k2 = match_target.shape
    Npos = match_target.sum()
    Nneg = b*k1*k2 - Npos


    # +1 for numerical stability since Npos could be 0 at the start!
    pos_weight = (Nneg+1)/(Npos+Nneg)
    neg_weight = (Npos+1)/(Npos+Nneg)

    desc_loss1 = F.cross_entropy(desc_pairs_score,
                                 match_target.long().view(-1),
                                 weight=torch.tensor([neg_weight, pos_weight]).to(device))

    # Descriptor loss: Contrastive loss (with margin)
    # TODO: Margin (pos and neg should be hyper-parameters)
    if desc_loss_comp_wt[0] == 1: # loss_mode = 'aux'
        mpos = 0.1
    elif desc_loss_comp_wt[0] == 0: # loss_mode = 'hinge'
        mpos = 0
    mneg = 1
    pos_loss = torch.sum(match_target * torch.max(torch.zeros_like(desc_pairs_norm).to(device), desc_pairs_norm - mpos)) / (2*Npos + 1e-6)
    neg_loss = torch.sum((1.0 - match_target) * torch.max(torch.zeros_like(desc_pairs_norm).to(device), mneg - desc_pairs_norm)) / (2*Nneg + 1e-6)
    desc_loss2 = pos_loss + neg_loss

    # DescLoss = alpha*CE-Loss + beta*HingeLoss
    desc_loss = desc_loss_comp_wt[0]*desc_loss1 + desc_loss_comp_wt[1]*desc_loss2

    # total loss
    loss = landmark_logits1_loss + landmark_logits2_loss + desc_loss

    loss_dict = {}
    loss_dict['loss'] = loss
    loss_dict['landmark_1_loss'] = landmark_logits1_loss
    loss_dict['landmark_2_loss'] = landmark_logits2_loss
    loss_dict['landmark_1_loss_wce'] = landmark_logits1_lossb
    loss_dict['landmark_1_loss_max_p'] = landmark_logits1_lossa
    loss_dict['landmark_2_loss_wce'] = landmark_logits2_lossb
    loss_dict['landmark_2_loss_max_p'] = landmark_logits2_lossa
    loss_dict['desc_loss_ce'] = desc_loss1
    loss_dict['desc_loss_hinge'] = desc_loss2
    loss_dict['kpts_inside_lung_1'] = torch.nonzero(mask_idxs_1).shape[0]
    loss_dict['kpts_inside_lung_2'] = torch.nonzero(mask_idxs_2).shape[0]
    loss_dict['kpts_outside_lung_1'] = torch.nonzero(1-mask_idxs_1).shape[0]
    loss_dict['kpts_outside_lung_2'] = torch.nonzero(1-mask_idxs_2).shape[0]

    return loss_dict



