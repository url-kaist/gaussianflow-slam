import numpy as np
import torch
from gfslam_net import cvx_upsample
import cv2
import torch.nn.functional as F

def upsample_flow(target, coords0, upmask, ii):
        flows   = target.squeeze(0) - coords0         # (E,h,w,2)
        _, inv  = torch.unique(ii, sorted=True, return_inverse=True)
        mask_e  = upmask.squeeze(0)[inv]              # (E,9*8*8,h,w)
        return cvx_upsample(flows, mask_e) * 8        # (E,8h,8w,2)
    
def upsample_weight(weight, upmask, ii):
    weights   = weight.squeeze(0)         # (E,h,w,2)
    _, inv  = torch.unique(ii, sorted=True, return_inverse=True)
    mask_e  = upmask.squeeze(0)[inv]              # (E,9*8*8,h,w)
    return cvx_upsample(weights, mask_e)            # (E,8h,8w,2)

# def downsample_flow(flow_up, scale_factor=1/8):
#     flow_down = F.interpolate(flow_up, scale_factor=scale_factor, mode='bilinear', align_corners=False)
#     return flow_down * scale_factor # Don't touch this!!!!!!!!!!!!

def downsample_flow(
    flow_up: torch.Tensor,
    scale: int = 8,
    mode: str = "antialiased",   # "antialiased" | "blur_pool"
):
    scale_factor = 1.0 / scale          # ex) 1/8
    if mode == "antialiased":
        flow_down = F.interpolate(
            flow_up,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    elif mode == "blur_pool":
        sigma = 0.5 * scale                # ≈ 4 px @ scale=8
        ksize = int(4 * sigma + 1) | 1
        x = torch.arange(ksize, device=flow_up.device, dtype=flow_up.dtype)
        x -= ksize // 2
        kernel1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel1d /= kernel1d.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.expand(2, 1, ksize, ksize)  # (C=2,1,k,k)

        flow_blur = F.conv2d(
            flow_up, kernel2d, padding=ksize // 2, groups=2
        )                                            # (B,2,H,W)
        flow_down = F.avg_pool2d(flow_blur, kernel_size=scale, stride=scale)
    else:
        raise ValueError(f"mode must be 'antialiased' or 'blur_pool', got {mode}")

     # Don't touch this!!!!!!!!!!!!
    return flow_down * scale_factor

def downsample_disp(disp_up, scale_factor=1/8):
    disp_down = F.interpolate(disp_up, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return disp_down # Don't touch this!!!!!!!!!!!!

def vis_flow(flow, scale=0):
   fx, fy = cv2.split(flow)
   mag,ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
   if scale== 0:
      cv2.normalize(mag, mag, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
   else:
      mag /= scale
   ret = cv2.merge([ang,mag,np.ones_like(mag)])
   ret = cv2.cvtColor(ret, cv2.COLOR_HSV2BGR)
   return ret

@torch.no_grad()
def normalize_weights(weight_ups, use_for_pose=False):
   """ Normalize the weights of the flow weights """
   w = weight_ups  # [N, 2]
   # (x^2 + y^2) -> sqrt -> * 1/sqrt(2)
   sq = (w * w).sum(dim=-1)         # [N]
   sq.sqrt_().mul_(0.7071067811865475)
   return sq

#    w_xy = weight_ups.detach()
#    w_norm = torch.linalg.vector_norm(w_xy, dim=-1) * 0.7071067811865475
#    w_xy = weight_ups.detach().clone().cuda()
#    w_norm = torch.norm(w_xy, dim=-1) / torch.sqrt(torch.tensor(2.0, device=w_xy.device)) ### BEST
#    w_min = w_norm.flatten(1).min(dim=1).values.view(-1, 1, 1)   # (E, 1, 1)
#    w_max = w_norm.flatten(1).max(dim=1).values.view(-1, 1, 1)   # (E, 1, 1)
#    w_norm = w_norm / torch.sqrt(torch.tensor(2.0, device=w_xy.device))
#    w_norm_mm = (w_norm - w_min) / (w_max - w_min + 1e-8)
#    w_norm = torch.sqrt(w_norm)
#    w_norm = torch.mean(w_xy, dim=-1)
#    w_norm = torch.sqrt(w_norm)
   
#    w_norm = torch.sqrt(torch.clamp(w_xy[...,0], 0, 1) * torch.clamp(w_xy[...,1], 0, 1))
   return w_norm


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
