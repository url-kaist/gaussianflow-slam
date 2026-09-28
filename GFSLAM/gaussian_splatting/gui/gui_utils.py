import queue

import cv2
import numpy as np
import open3d as o3d
import torch
import queue as pyqueue

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def put_latest(q, packet):
    try:
        q.put_nowait(packet)
    except pyqueue.Full:
        # Drop the packet
        pass

class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


class GaussianPacket:
    def __init__(
        self,
        gaussians=None,
        keyframe=None,
        current_frame=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        gtflow=None,
        gsflow=None,
        keyframes=None,
        finish=False,
        kf_window=None,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            self.get_xyz = gaussians.get_xyz.detach().clone()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone()
            self.get_scaling = gaussians.get_scaling.detach().clone()
            self.get_rotation = gaussians.get_rotation.detach().clone()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone()

            self._rotation = gaussians._rotation.detach().clone()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            # self.n_obs = gaussians.n_obs.clone()

        self.keyframe = keyframe
        self.current_frame = current_frame
        self.gtcolor = self.resize_img(gtcolor, 240)
        self.gtdepth = self.resize_img(gtdepth, 240)
        self.gtnormal = self.resize_img(gtnormal, 240)
        self.gtflow = self.resize_img(gtflow, 240)
        self.gsflow = self.resize_img(gsflow, 240)
        self.keyframes = keyframes
        self.finish = finish
        self.kf_window = kf_window

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation) # L = R @ S => Nx3x3
        actual_covariance = L @ L.transpose(1, 2) # R @ S @ S^T @ R^T => Nx3x3
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


class ParamsGUI:
    def __init__(
        self,
        pipe=None,
        background=None,
        gaussians=None,
        q_main2vis=None,
        q_vis2main=None,
    ):
        self.pipe = pipe
        self.background = background
        self.gaussians = gaussians
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

def optical_flow_to_rgb(flow_tensor):
    """
    Convert an optical flow field to an RGB image.
    - input : `torch.Tensor` (H, W, 2) optical flow (may live on the GPU)
    - output: `torch.Tensor` (H, W, 3) RGB image in [0, 1] (stays on the GPU)
    """
    if isinstance(flow_tensor, torch.Tensor):
        flow_numpy = flow_tensor.detach().cpu().numpy()

    # fx, fy = cv2.split(flow_numpy)
    # mag,ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    # cv2.normalize(mag, mag, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # ret = cv2.merge([ang,mag,np.ones_like(mag)])
    # ret = cv2.cvtColor(ret, cv2.COLOR_HSV2RGB)
    # ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # return ret

    # flow_x, flow_y = flow_numpy[..., 0], flow_numpy[..., 1]
    flow_x, flow_y = flow_numpy[0, ...], flow_numpy[1, ...]

    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)

    h = angle
    s = magnitude
    v = np.ones_like(magnitude)

    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)  # (H, W, 3) uint8
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)  # (H, W, 3) RGB uint8
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    rgb = rgb.astype(np.uint8)

    return rgb

    # rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=device) #/ 255.0  # (H, W, 3), [0, 1]
    # return rgb_tensor


def tnormalize(v: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Stable L2 normalize for vectors (supports batch)."""
    return v / torch.clamp(v.norm(dim=dim, keepdim=True), min=eps)

def look_at_torch(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Build an OpenGL-style camera-to-world matrix (assumes z = +forward).
    eye, center, up: (..., 3)
    returns: (..., 4, 4) C2W
    """
    # forward/right/up
    f = tnormalize(center - eye, dim=-1)               # (..., 3)
    s = tnormalize(torch.cross(up, f, dim=-1), dim=-1) # (..., 3)
    u = torch.cross(f, s, dim=-1)                      # (..., 3)

    dtype, device = eye.dtype, eye.device
    batch_shape = eye.shape[:-1]
    M = torch.eye(4, dtype=dtype, device=device).expand(*batch_shape, 4, 4).clone()

    M[..., 0, :3] = s
    M[..., 1, :3] = u
    M[..., 2, :3] = f
    M[..., :3, 3] = eye
    return M

def get_behind_view_Tcw(
    Tcw_target: torch.Tensor,            # (...,4,4)  W2C of target view
    size: float = 0.02,
    offset_local = (0.0, -2.5, -30.0),
) -> torch.Tensor:
    dtype, device = Tcw_target.dtype, Tcw_target.device
    R_cw = Tcw_target[..., :3, :3]
    t_cw = Tcw_target[..., :3,  3]

    # p_local = behind offset in camera coords
    p_local = torch.tensor(offset_local, dtype=dtype, device=device) * size
    p_local = p_local.expand_as(t_cw)

    t_new = t_cw - p_local

    T_new = Tcw_target.clone()
    T_new[..., :3, :3] = R_cw
    T_new[..., :3,  3] = t_new
    return T_new

# def get_behind_view_Tcw(
#     size: float = 0.02,
# ) -> torch.Tensor:
#     """
#     """
#     dtype, device = Tcw_target.dtype, Tcw_target.device
#     batch = Tcw_target.shape[:-2]

#     # W2C = [R_cw | t_cw];  C2W = [R_wc | t_wc] where R_wc = R_cw^T, t_wc = -R_cw^T t_cw
#     R_cw = Tcw_target[..., :3, :3]
#     t_cw = Tcw_target[..., :3,  3]
#     R_wc = R_cw.transpose(-1, -2)
#     t_wc = (-R_wc @ t_cw.unsqueeze(-1)).squeeze(-1)

#     p_center_local = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device) * size
#     p_behind_local = torch.tensor(offset_local,      dtype=dtype, device=device) * size
#     up_cam         = torch.tensor([0.0, 1.0, 0.0],   dtype=dtype, device=device)

#     p_center_local = p_center_local.expand(*batch, 3)
#     p_behind_local = p_behind_local.expand(*batch, 3)
#     up_cam         = up_cam.expand(*batch, 3)

#     center_world = (R_wc @ p_center_local.unsqueeze(-1)).squeeze(-1) + t_wc
#     eye_behind   = (R_wc @ p_behind_local.unsqueeze(-1)).squeeze(-1) + t_wc
#     up_world     = tnormalize((R_wc @ up_cam.unsqueeze(-1)).squeeze(-1), dim=-1)

#     C2W_view = look_at_torch(eye_behind, center_world, up_world)   # (...,4,4)
#     Tcw_view = torch.linalg.inv(C2W_view)                           # (...,4,4)
#     Tcw_view[:3,:3] = Tcw_target[:3,:3]
#     return Tcw_view

def frustum_points_local(size=0.002):
    P = np.array([
        [0.0,  0.0, 0.0],
        [1.0, -0.5, 2.0],
        [-1.0,-0.5, 2.0],
        [1.0,  0.5, 2.0],
        [-1.0, 0.5, 2.0],
    ], dtype=np.float64) * size
    L = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,4],[3,4]], dtype=np.int32)
    return P, L

def transform_points(R, t, P):
    # to torch, dtype/device align with R
    if isinstance(P, np.ndarray):
        P = torch.from_numpy(P)
    P = P.to(dtype=R.dtype, device=R.device)

    # normalize shapes
    single_batch = (R.dim() == 2)  # (3,3) vs (B,3,3)
    if single_batch:
        Rb = R.unsqueeze(0)            # (1,3,3)
        tb = t.unsqueeze(0)            # (1,3)
    else:
        Rb, tb = R, t                  # (B,3,3), (B,3)

    if P.dim() == 2:                   # (N,3) -> (1,N,3)
        Pb = P.unsqueeze(0)
    elif P.dim() == 3:                 # (B?,N,3)
        Pb = P
    else:
        raise ValueError(f"Invalid P shape {tuple(P.shape)}; expected (N,3) or (B,N,3)")

    # broadcast N to batch if needed
    if Pb.shape[0] == 1 and Rb.shape[0] > 1:
        Pb = Pb.expand(Rb.shape[0], -1, -1)    # (B,N,3)

    # matmul: (B,3,3) @ (B,N,3)^T -> (B,3,N) -> + t -> (B,3,N) -> (B,N,3)
    Pc = (Rb @ Pb.transpose(1, 2)) + tb.unsqueeze(-1)   # (B,3,N)
    Pc = Pc.transpose(1, 2).contiguous()               # (B,N,3)

    # squeeze batch if input was single
    if single_batch and (P.dim() == 2):
        Pc = Pc.squeeze(0)                              # (N,3)
    return Pc

def project_points(K, Pc):
    z = Pc[:,2:3]
    uv = (Pc[:, :2] / np.clip(z, 1e-9, None))
    uv_h = (K @ np.c_[uv, np.ones((uv.shape[0],1))].T).T  # (N,3)
    u, v = uv_h[:,0], uv_h[:,1]
    return np.stack([u, v, z[:,0]], axis=1)  # (N,3) with depth

# def draw_wireframe(img_bgr, pts2d, lines, color=(0,255,0), thickness=2):
#     H, W = img_bgr.shape[:2]
#     for i, j in lines:
#         p1 = pts2d[i]; p2 = pts2d[j]
#         if p1[2] <= 0 or p2[2] <= 0: 
#             continue
#         x1,y1 = int(round(p1[0])), int(round(p1[1]))
#         x2,y2 = int(round(p2[0])), int(round(p2[1]))
#         if (x1<0 and x2<0) or (x1>=W and x2>=W) or (y1<0 and y2<0) or (y1>=H and y2>=H):
#             pass
#         cv2.line(img_bgr, (x1,y1), (x2,y2), color, thickness, lineType=cv2.LINE_AA)

def draw_wireframe(
    img_bgr,
    pts2d,
    edges,
    color=(0,255,0),
    thickness=1,
    mode: str = "uvz",
    do_clip_outside: bool = False
):
    """
    mode:
      - "uvz": pts2d[:,0:2] = pixel (u,v), pts2d[:,2] = depth z (only z>0 is drawn)
      - "homogeneous": pts2d[:,0:2]=(x,y), pts2d[:,2]=w → (u,v) = (x/w, y/w)
    """
    # to numpy
    if isinstance(pts2d, torch.Tensor):
        pts2d = pts2d.detach().cpu().numpy()
    pts2d = np.asarray(pts2d)

    # shape normalize
    if pts2d.ndim == 1:
        if pts2d.size % 3 == 0:
            pts2d = pts2d.reshape(-1, 3)
        elif pts2d.size % 2 == 0:
            pts2d = pts2d.reshape(-1, 2)
        else:
            raise ValueError(f"draw_wireframe: unexpected pts2d size {pts2d.size}")

    if pts2d.ndim != 2 or pts2d.shape[1] not in (2,3):
        raise ValueError(f"draw_wireframe: expected (N,2) or (N,3), got {pts2d.shape}")

    # compute uv according to mode
    if pts2d.shape[1] == 2:
        uv = pts2d.astype(np.float64)
        z_ok = np.ones((uv.shape[0],), dtype=bool)
    else:
        if mode == "homogeneous":
            w = pts2d[:, 2:3]
            w_safe = np.where(np.abs(w) > 1e-12, w, 1.0)
            uv = pts2d[:, :2] / w_safe
            z_ok = (w.reshape(-1) > 0)
        elif mode == "uvz":
            uv = pts2d[:, :2].astype(np.float64)
            z = pts2d[:, 2].astype(np.float64)
            z_ok = (z > 0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    finite = np.isfinite(uv).all(axis=1)
    valid = finite & z_ok
    uv_i = np.rint(uv).astype(np.int32)

    H, W = img_bgr.shape[:2]
    for (i, j) in edges:
        if not (0 <= i < len(uv_i) and 0 <= j < len(uv_i)):
            continue
        if not (valid[i] and valid[j]):
            continue

        x1, y1 = int(uv_i[i, 0]), int(uv_i[i, 1])
        x2, y2 = int(uv_i[j, 0]), int(uv_i[j, 1])

        if do_clip_outside:
            if (x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or \
               (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
                continue

        cv2.line(img_bgr, (x1, y1), (x2, y2), color, int(thickness), lineType=cv2.LINE_AA)


