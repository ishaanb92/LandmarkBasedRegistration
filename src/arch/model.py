"""

A model to find correspondences (landmarks) between a pair of images.
Contrary to traditional approaches, this model performs detection and description
in parallel (using different layers of the same network)

Reference:
    Paper: http://arxiv.org/abs/2109.02722
    Code: https://github.com/monikagrewal/End2EndLandmarks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from unet3d import UNet

class LesionMatchingModel(nn.Module):

    def __init__(self,
                 K=512,
                 W=4,
                 n_channels=1
                 ):

        super(LesionMatchingModel, self).__init__()

        # Main learning block that jointly performs
        # description (intermediate feature maps) and
        # detection (output map)
        self.cnn = UNet(n_channels=n_channels,
                        n_classes=1,
                        trilinear=False)

        self.descriptor_matching = DescriptorMatcher(in_channels=self.cnn.descriptor_length,
                                                     out_channels=2)

        self.W = W
        self.K = K

    def forward(self, x1, x2, training=True):

        kpts_1, features_1 = self.cnn(x1)
        kpts_2, features_2 = self.cnn(x2)

        # Sample keypoints and corr. descriptors
        kpt_sampling_grid_1, kpt_logits_1, descriptors_1 = self.sampling_block(kpt_map=kpts_1,
                                                                             features=features_1,
                                                                             W=self.W,
                                                                             num_pts=self.K,
                                                                             training=training)

        kpt_sampling_grid_2, kpt_logits_2, descriptors_2 = self.sampling_block(kpt_map=kpts_2,
                                                                             features=features_2,
                                                                             W=self.W,
                                                                             num_pts=self.K,
                                                                             training=training)

        # Match descriptors
        desc_pairs_score, desc_pair_norm = self.descriptor_matching(descriptors_1,
                                                                    descriptors_2)

        output_dict = {}
        output_dict['kpt_sampling_grid'] = (kpt_sampling_grid_1, kpt_sampling_grid_2)
        output_dict['kpt_logits'] = (kpt_logits_1, kpt_logits_2)
        output_dict['desc_score'] = desc_pairs_score
        output_dict['desc_norm'] = desc_pair_norm

        return output_dict


    def sampling_block(self,
                       kpt_map=None,
                       features=None,
                       conf_thresh=0.0001,
                       W=4,
                       training=True,
                       num_pts = 512):

        """

        Choose top-k points as candidates from the keypoint probability map
        Sample corresponding keypoint probabilities and descriptors

        """

        b, _, i, j, k = kpt_map.shape
        kpt_probmap = torch.sigmoid(kpt_map)


        # Only retain the maximum activation in a given neighbourhood of size of W, W, W
        # All non-maximal values set to zero
        kpt_probmap_downsampled, indices = F.max_pool3d(kpt_probmap,
                                                        kernel_size=(W, W, W),
                                                        stride=(W, W, W),
                                                        return_indices=True)

        kpt_probmax_suppressed = F.max_unpool3d(kpt_probmap_downsampled,
                                                indices=indices,
                                                kernel_size=(W, W, W),
                                                stride=(W, W, W))

        kpt_probmax_suppressed = torch.squeeze(kpt_probmax_suppressed,
                                               dim=1)

        # Create keypoint tensor : Each key-point is a 4-D vector: (x, y, z, prob.)
        kpts = torch.zeros(size=(b, num_pts, 4),
                           dtype=kpt_map.dtype).to(kpt_map.device)

        for batch_idx in range(b):
            # Create binary mask of shape [D, H, W]
            kpt_mask = torch.where(kpt_probmax_suppressed[batch_idx, ...]>=conf_thresh,
                                   1.0,
                                   0.0).type(kpt_probmap.dtype)

            zs, ys, xs= torch.nonzero(kpt_mask,
                                      as_tuple=True)

            # FIXME: Handle cases where N < num_pts
            N = len(zs)

            if N < num_pts:
                raise RuntimeError('Number of point above threshold ({}) are less thant K ({})'.format(N, num_pts))

            item_kpts = torch.zeros(size=(N, 4),
                                    dtype=kpt_map.dtype).to(kpt_map.device)

            item_kpts[:, 0] = xs
            item_kpts[:, 1] = ys
            item_kpts[:, 2] = zs
            item_kpts[:, 3] = kpt_probmax_suppressed[batch_idx, zs, ys, xs]

            idxs_desc = torch.argsort(-1*item_kpts[:, 3])

            item_kpts_sorted = item_kpts[idxs_desc, :]
            top_k_kpts = item_kpts_sorted[:num_pts, :]

            kpts[batch_idx, ...] = top_k_kpts


        kpts_idxs = kpts[:, :, :3]

        kpts_sampling_grid = torch.zeros_like(kpts_idxs)
        kpts_sampling_grid[:, :, 0] = (kpts_idxs[:, :, 0]*2/(k-1)) - 1
        kpts_sampling_grid[:, :, 1] = (kpts_idxs[:, :, 1]*2/(j-1)) - 1
        kpts_sampling_grid[:, :, 2] = (kpts_idxs[:, :, 2]*2/(i-1)) - 1

        # Expected shape: [B, K, 1, 1, 3]
        kpts_sampling_grid = torch.unsqueeze(kpts_sampling_grid, dim=1)
        kpts_sampling_grid = torch.unsqueeze(kpts_sampling_grid, dim=1)

        # Sample keypoint logits
        # BCEWithLogits safe to use with autocast
        # See: https://discuss.pytorch.org/t/runtimeerror-binary-cross-entropy-and-bceloss-are-unsafe-to-autocast/118538
        kpts_scores = F.grid_sample(kpt_map,
                                    kpts_sampling_grid,
                                    align_corners=False)

        # Get rid of fake channels axes
        kpts_scores = kpts_scores.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)

        # Sample descriptors using chosen keypoints
        descriptors = []
        for fmap in features:
            fmap_resampled = F.grid_sample(fmap,
                                           kpts_sampling_grid,
                                           align_corners=False)
            descriptors.append(fmap_resampled)

        # Expected shape: [B, C, 1, ,1, K]
        descriptors = torch.cat(descriptors, dim=1)


        return kpts_sampling_grid, kpts_scores, descriptors

class DescriptorMatcher(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super(DescriptorMatcher, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, out1, out2):

        b, c, d1, h1, w1 = out1.size()
        _, _, d2, h2, w2 = out2.size()

        out1 = out1.view(b, c, d1*h1*w1).permute(0, 2, 1).view(b, d1*h1*w1, 1, c)
        out2 = out2.view(b, c, d2*h2*w2).permute(0, 2, 1).view(b, 1, d2*h2*w2, c)

        # Outer product to get all possible pairs
        # Shape: [b, k, k, c]
        out = out1*out2

        out = out.contiguous().view(-1, c)

        # Unnormalized logits used for CE loss
        # Alternatively can be single channel with BCE loss
        out = self.fc(out)

        # Compute norms
        desc_l2_norm_1 = torch.norm(out1, p=2, dim=3)
        out1_norm = out1.div(1e-6 + torch.unsqueeze(desc_l2_norm_1, dim=3))
        desc_l2_norm_2 = torch.norm(out2, p=2, dim=3)
        out2_norm = out1.div(1e-6 + torch.unsqueeze(desc_l2_norm_2, dim=3))

        out_norm = torch.norm(out1_norm-out2_norm, p=2, dim=3)

        return out, out_norm



