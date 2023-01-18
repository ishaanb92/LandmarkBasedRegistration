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
from lesionmatching.arch.unet3d import UNet
import warnings

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

    def forward(self, x1, x2, mask=None, mask2=None, training=True):

        kpts_1, features_1 = self.cnn(x1)
        kpts_2, features_2 = self.cnn(x2)


        # Sample keypoints and corr. descriptors
        kpt_sampling_grid_1, kpt_logits_1, descriptors_1 = self.sampling_block(kpt_map=kpts_1,
                                                                               features=features_1,
                                                                               W=self.W,
                                                                               num_pts=self.K,
                                                                               training=training,
                                                                               mask=mask)

        if kpt_sampling_grid_1 is None:
            return None

        kpt_sampling_grid_2, kpt_logits_2, descriptors_2 = self.sampling_block(kpt_map=kpts_2,
                                                                               features=features_2,
                                                                               W=self.W,
                                                                               num_pts=self.K,
                                                                               training=training,
                                                                               mask=mask2)
        if kpt_sampling_grid_2 is None:
            return None

        # Match descriptors
        desc_pairs_score, desc_pair_norm = self.descriptor_matching(descriptors_1,
                                                                    descriptors_2)

        output_dict = {}
        output_dict['kpt_sampling_grid'] = (kpt_sampling_grid_1, kpt_sampling_grid_2)
        output_dict['kpt_logits'] = (kpt_logits_1, kpt_logits_2)
        output_dict['desc_score'] = desc_pairs_score
        output_dict['desc_norm'] = desc_pair_norm

        return output_dict

    # Break the 'inference' function into CNN output and sampling+matching
    # so that sliding_window_inference can be performed
    # We need to run the model twice because sliding_window_inference
    # expects all outputs to be of the same size
    # THESE FUNCTIONS ARE USED ONLY DURING INFERENCE!!!!
    def get_patch_keypoint_scores(self, x):

        x1 = x[:, 0, ...].unsqueeze(dim=1)
        x2 = x[:, 1, ...].unsqueeze(dim=1)

        # 1. Landmark (candidate) detections (logits)
        kpts_1, _ = self.cnn(x1)
        kpts_2, _ = self.cnn(x2)

        return kpts_1, kpts_2

    def get_patch_feature_descriptors(self, x):

        x1 = x[:, 0, ...].unsqueeze(dim=1)
        x2 = x[:, 1, ...].unsqueeze(dim=1)

        # 1. Landmark (candidate) detections
        _, features_1 = self.cnn(x1)
        _, features_2 = self.cnn(x2)

        # "features" is a tuple of feature maps from different U-Net resolutions
        # We return separate tensors so 'sliding_window_inference' works
        return features_1[0], features_1[1], features_2[0], features_2[1]


    def inference(self, kpts_1, kpts_2, features_1, features_2, conf_thresh=0.5, num_pts=1000, mask=None, mask2=None, test=True):

        b, c, i, j, k = kpts_1.shape

        # 2.)Sampling grid + descriptors
        kpt_sampling_grid_1, kpt_logits_1, descriptors_1 = self.sampling_block(kpt_map=kpts_1,
                                                                               features=features_1,
                                                                               W=self.W,
                                                                               num_pts=num_pts,
                                                                               training=not(test),
                                                                               conf_thresh=conf_thresh,
                                                                               mask=mask)

        kpt_sampling_grid_2, kpt_logits_2, descriptors_2 = self.sampling_block(kpt_map=kpts_2,
                                                                               features=features_2,
                                                                               W=self.W,
                                                                               num_pts=num_pts,
                                                                               training=not(test),
                                                                               conf_thresh=conf_thresh,
                                                                               mask=mask2)

        if kpt_sampling_grid_1 is None or kpt_sampling_grid_2 is None:
            return None

        landmarks_1 = self.convert_grid_to_image_coords(kpt_sampling_grid_1,
                                                        shape=(k, j, i))

        landmarks_2 = self.convert_grid_to_image_coords(kpt_sampling_grid_2,
                                                        shape=(k, j, i))

        # 3. Compute descriptor matching scores
        desc_pairs_score, desc_pair_norm = self.descriptor_matching(descriptors_1,
                                                                    descriptors_2)

        _, k1, k2 = desc_pair_norm.shape

        # Match probability
        desc_pairs_prob = F.softmax(desc_pairs_score, dim=1)[:, 1].view(b, k1, k2)

        # Two-way matching
        matches = []
        matches_norm = []
        matches_prob = []

        for batch_idx in range(b):

            pairs_prob = desc_pairs_prob[batch_idx]
            pairs_norm = desc_pair_norm[batch_idx]

            # 2-way matching w.r.t pair probabilities
            match_cols = torch.zeros((k1, k2))
            match_cols[torch.argmax(pairs_prob, dim=0), torch.arange(k2)] = 1
            match_rows = torch.zeros((k1, k2))
            match_rows[torch.arange(k1), torch.argmax(pairs_prob, dim=1)] = 1
            match_prob = match_rows*match_cols

            # 2-way matching w.r.t probabilities & min norm
            match_cols = torch.zeros((k1, k2))
            match_cols[torch.argmin(pairs_norm, dim=0), torch.arange(k2)] = 1
            match_rows = torch.zeros((k1, k2))
            match_rows[torch.arange(k1), torch.argmin(pairs_norm, dim=1)] = 1
            match_norm = match_rows*match_cols

            match = match_prob*match_norm

            matches.append(match)
            matches_norm.append(match_norm)
            matches_prob.append(match_prob)

        matches = torch.stack(matches)
        matches_norm = torch.stack(matches_norm)
        matches_prob = torch.stack(matches_prob)

        outputs = {}
        outputs['landmarks_1'] = landmarks_1
        outputs['landmarks_2'] = landmarks_2
        outputs['desc_score'] = desc_pairs_score
        outputs['desc_norm'] = desc_pair_norm
        outputs['matches'] = matches
        outputs['matches_norm'] = matches_norm
        outputs['matches_prob'] = matches_prob
        outputs['kpt_sampling_grid_1'] = kpt_sampling_grid_1
        outputs['kpt_sampling_grid_2'] = kpt_sampling_grid_2
        outputs['kpt_logits_1'] = kpt_logits_1
        outputs['kpt_logits_2'] = kpt_logits_2

        return outputs

    @staticmethod
    def convert_grid_to_image_coords(pts, shape=(64, 128, 128)):
        """
        Convert grid points ([-1, 1] range) to image coords

        """

        pts = pts.squeeze(dim=1).squeeze(dim=1) # Shape: [B, K, 3]

        # Scale to [0, 1] range
        pts = (pts + 1.)/2.

        # Scale to image dimensions (DHW ordering)
        pts = pts * torch.Tensor([(shape[0]-1, shape[1]-1, shape[2]-1)]).view(1, 1, 3).int().to(pts.device)

        return pts


    def sampling_block(self,
                       kpt_map=None,
                       features=None,
                       conf_thresh=0.1,
                       W=4,
                       training=True,
                       mask=None,
                       num_pts = 512):

        """

        Choose top-k points as candidates from the keypoint probability map
        Sample corresponding keypoint probabilities and descriptors

        """


        b, _, i, j, k = kpt_map.shape

        kpt_probmap = torch.sigmoid(kpt_map)

        # Multiply the probability map by the mask (so sampling probability in non-mask regions = 0)
        if mask is not None:
            kpt_probmap = kpt_probmap*mask

        # Only retain the maximum activation in a given neighbourhood of size of W, W, W
        # All non-maximal values set to zero
        kpt_probmap_downsampled, indices = F.max_pool3d(kpt_probmap,
                                                        kernel_size=(W, W, W//2),
                                                        stride=(W, W, W//2),
                                                        return_indices=True)

        kpt_probmap_suppressed = F.max_unpool3d(kpt_probmap_downsampled,
                                                indices=indices,
                                                kernel_size=(W, W, W//2),
                                                stride=(W, W, W//2))

        kpt_probmap_suppressed = torch.squeeze(kpt_probmap_suppressed,
                                               dim=1)

        kpts = torch.zeros(size=(b, num_pts, 4),
                           dtype=kpt_map.dtype).to(kpt_map.device)

        for batch_idx in range(b):
            # Create binary mask of shape [H, W, D]
            kpt_mask = torch.where(kpt_probmap_suppressed[batch_idx, ...]>=conf_thresh,
                                   torch.ones_like(kpt_probmap_suppressed[batch_idx, ...]),
                                   torch.zeros_like(kpt_probmap_suppressed[batch_idx, ...])).type(kpt_probmap.dtype)

            ii, jj, kk = torch.nonzero(kpt_mask,
                                       as_tuple=True)

            # Re-map coordinates from matrix/tensor convention to Cartesian convention
            zs = ii
            ys = jj
            xs = kk

            N = len(zs)

            if training is False:
                print('Found {} keypoint candidates above confidence threshold'.format(N))

            if N < num_pts:
                if training is True:
                    kpt_mask = torch.where(kpt_probmap_suppressed[batch_idx, ...]>0,
                                           torch.ones_like(kpt_probmap_suppressed[batch_idx, ...]),
                                           torch.zeros_like(kpt_probmap_suppressed[batch_idx, ...])).type(kpt_probmap.dtype)

                    ii, jj, kk = torch.nonzero(kpt_mask,
                                               as_tuple=True)

                    # Re-map coordinates from matrix/tensor convention to Cartesian convention
                    zs = ii
                    ys = jj
                    xs = kk

                    N = len(zs)

                    print('Number of keypoints found after reducing the threshold = {}'.format(N))

                    # Even after reducing the threshold to 0, we still do not have enough kpts => most of the heatmap is "cold"
                    if num_pts > N:
                        print('Skip this batch. Too few keypoint candidates')
                        return None, None, None
                else:
                    if N > 0:
                        warnings.warn('The number of key-points requested ({}) is \
                                       less than the number of keypoints above threshold ({})'.format(num_pts,
                                                                                             N))
                        kpts = torch.zeros(size=(b, N, 4),
                                           dtype=kpt_map.dtype).to(kpt_map.device)
                        num_pts = N
                    else:
                        raise RuntimeError('No landmark candidates found in image')


            item_kpts = torch.zeros(size=(N, 4),
                                    dtype=kpt_probmap.dtype).to(kpt_map.device)

            item_kpts[:, 0] = xs
            item_kpts[:, 1] = ys
            item_kpts[:, 2] = zs
            item_kpts[:, 3] = kpt_probmap_suppressed[batch_idx, zs , ys, xs]

            idxs_desc = torch.argsort(-1*item_kpts[:, 3])

            item_kpts_sorted = item_kpts[idxs_desc, :]


            top_k_kpts = item_kpts_sorted[:num_pts, :]


            kpts[batch_idx, ...] = top_k_kpts


        kpts_idxs = kpts[:, :, :3]

        # Create sampling grid from kpt coords by recaling them to [-1, 1] range
        kpts_sampling_grid = torch.zeros_like(kpts_idxs)
        kpts_sampling_grid[:, :, 0] = (kpts_idxs[:, :, 0]*2/(k-1)) - 1
        kpts_sampling_grid[:, :, 1] = (kpts_idxs[:, :, 1]*2/(j-1)) - 1
        kpts_sampling_grid[:, :, 2] = (kpts_idxs[:, :, 2]*2/(i-1)) - 1

        # Expected shape: [B, 1, 1, K, 3]
        kpts_sampling_grid = torch.unsqueeze(kpts_sampling_grid, dim=1)
        kpts_sampling_grid = torch.unsqueeze(kpts_sampling_grid, dim=1)

        # Sample keypoint logits instead of heatmap probabilities
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

        # Expected shape: [B, C, 1, 1, K]
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
        # Shape: [b, k1, k2, c]
        out = out1*out2

        out = out.contiguous().view(-1, c)

        # Unnormalized logits used for CE loss
        # Alternatively can be single channel with BCE loss
        out = self.fc(out)

        # Compute norms
        desc_l2_norm_1 = torch.norm(out1, p=2, dim=3)
        out1_norm = out1.div(1e-6 + torch.unsqueeze(desc_l2_norm_1, dim=3))

        desc_l2_norm_2 = torch.norm(out2, p=2, dim=3)
        out2_norm = out2.div(1e-6 + torch.unsqueeze(desc_l2_norm_2, dim=3))

        out_norm = torch.norm(out1_norm-out2_norm, p=2, dim=3)

        return out, out_norm



