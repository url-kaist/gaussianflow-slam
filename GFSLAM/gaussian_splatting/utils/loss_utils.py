#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean(dim=0)#.mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average).mean(dim=0)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map#.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# window shape: [C, 1, K, K]
def ssim_masked(img1, img2, mask, window_size=11, size_average=True, eps=1e-6, K=(0.01, 0.03)):
    """
    img1, img2: [C,H,W] or [N,C,H,W]; values are expected in [0,1]
    mask: [H,W], [1,H,W] or [N,1,H,W] (1 = valid, 0 = invalid)

    size_average=True  -> mean SSIM over the valid region (scalar)
    size_average=False -> per-channel SSIM map [N,C,H,W]
    """
    added_batch = False
    if img1.dim() == 3:  # [C,H,W] -> [1,C,H,W]
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        added_batch = True
    assert img1.shape == img2.shape and img1.dim() == 4, "img1/img2 must be [N,C,H,W] or [C,H,W]"

    N, C, H, W = img1.shape

    if mask.dim() == 2:          # [H,W]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(N, 1, H, W)
    elif mask.dim() == 3:
        if mask.size(0) == 1:
            mask = mask.unsqueeze(0).expand(N, 1, H, W)  # [N,1,H,W]
        else:
            raise ValueError("mask shape [C,H,W] is ambiguous. Provide [H,W], [1,H,W], or [N,1,H,W].")
    elif mask.dim() == 4:        # [N,1,H,W]
        if mask.size(0) != N or mask.size(1) != 1 or mask.size(2) != H or mask.size(3) != W:
            raise ValueError("mask must be [N,1,H,W] matching img spatial dims.")
    else:
        raise ValueError("Unsupported mask shape")

    mask = mask.to(dtype=img1.dtype).clamp(0, 1)

    window = create_window(window_size, C).type_as(img1)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    pad = window_size // 2

    base_window = window[:1, :1, :, :]  # [1,1,K,K]

    denom = F.conv2d(mask, base_window, padding=pad)  # [N,1,H,W]
    denom = denom.clamp_min(eps)

    mask_c = mask.expand(-1, C, -1, -1)

    mu1 = F.conv2d(img1 * mask_c, window, padding=pad, groups=C) / denom
    mu2 = F.conv2d(img2 * mask_c, window, padding=pad, groups=C) / denom

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    m11 = F.conv2d((img1 * img1) * mask_c, window, padding=pad, groups=C) / denom
    m22 = F.conv2d((img2 * img2) * mask_c, window, padding=pad, groups=C) / denom
    m12 = F.conv2d((img1 * img2) * mask_c, window, padding=pad, groups=C) / denom

    sigma1_sq = (m11 - mu1_sq).clamp_min(0.0)
    sigma2_sq = (m22 - mu2_sq).clamp_min(0.0)
    sigma12   = (m12 - mu1_mu2)

    C1 = (K[0] ** 2)
    C2 = (K[1] ** 2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        ssim_map_cmean = ssim_map.mean(dim=1, keepdim=True)     # [N,1,H,W]
        valid = (denom > eps).to(img1.dtype)                    # [N,1,H,W]
        val = (ssim_map_cmean * valid).sum() / (valid.sum() + eps)
        if added_batch:
            return val
        return val
    else:
        if added_batch:
            return ssim_map.squeeze(0).mean(dim=0)  # [C,H,W]
        return ssim_map.mean(dim=0)
