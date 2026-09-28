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

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


# Sentinel default — letting callers omit `translate` lets the fast path
# below be taken via an `is` check (cheap) instead of a tensor compare.
_DEFAULT_TRANSLATE = torch.zeros(3)


# Get Tcw
def getWorld2View2(R, t, translate=_DEFAULT_TRANSLATE, scale=1.0):
    # Compose [R | t ; 0 0 0 1] directly. The original implementation called
    # torch.linalg.inv twice to apply (translate, scale) to the camera centre;
    # for the default args (translate == 0, scale == 1) those two inverses are
    # mutually cancelling no-ops, and even in the general case the inverse of
    # an SE(3) matrix is analytic ([R^T | -R^T t]) so we never need linalg.inv.
    # Force float32 — the diff-gaussian-rasterizer (and many downstream
    # consumers like the GUI viewer) expect Float, and the original
    # implementation always produced Float because it didn't specify dtype.
    Rt = torch.zeros((4, 4), device=R.device, dtype=torch.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    if translate is _DEFAULT_TRANSLATE and scale == 1.0:
        return Rt

    # General case: cam_center_new = (-R^T t + translate) * scale, then
    # rebuild Rt from [R | -R @ cam_center_new].
    translate = translate.to(R.device)
    cam_center_new = (-(R.transpose(0, 1) @ t) + translate) * scale
    Rt_new = torch.zeros((4, 4), device=R.device, dtype=torch.float32)
    Rt_new[:3, :3] = R
    Rt_new[:3, 3] = -(R @ cam_center_new)
    Rt_new[3, 3] = 1.0
    return Rt_new


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P

# znear, zfar: depth range of the scene that is visible
# 
def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left) 
    P[1, 1] = 2.0 * znear / (top - bottom) 
    P[0, 2] = (right + left) / (right - left) 
    P[1, 2] = (top + bottom) / (top - bottom) 
    P[3, 2] = z_sign # 1.0 -> right-handed system, -1.0 -> left-handed system
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
