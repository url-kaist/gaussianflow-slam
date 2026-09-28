import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask

class CameraVis:
    def __init__(self, uid, R, T, R_gt, T_gt):
        self.uid = int(uid)
        self.R = R.detach().cpu()
        self.T = T.detach().cpu()
        self.R_gt = R_gt
        self.T_gt = T_gt

def to_camera_vis(viewpoint):
    """Convert a full Camera into the lightweight CameraVis used by the GUI."""
    return CameraVis(
        uid=int(viewpoint.uid),
        R=viewpoint.R,
        T=viewpoint.T,
        R_gt=viewpoint.R_gt,
        T_gt=viewpoint.T_gt,
    )

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None
        self.flow_vis = None
        self.flow_image = torch.zeros(2, image_height, image_width).cuda()
        self.flow_conf = torch.ones(image_height, image_width).cuda()
        self.flow_comp_idx = -1

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
    
    # donguk
    def __eq__(self, other):
        if (isinstance(other, Camera)):
            return self.uid == other.uid
        return False

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        # Direct construction of (W2V).T (column-major) — getWorld2View2 with
        # default args is now a 4x4 fill, so this avoids two linalg.inv that
        # used to dominate the renderer's per-frame cost.
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        # Analytic SE(3) inverse: cam centre in world frame is -R^T @ T.
        # Force float32 — downstream consumers (rasterizer, GUI) expect Float.
        # Replaces world_view_transform.inverse()[3, :3] which invoked
        # torch.linalg.inv on every render call (~108k/200 frames previously).
        return (-(self.R.transpose(0, 1) @ self.T)).to(torch.float32)

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
    
    def update_GTRT(self, R, t):
        self.R_gt = R.to(device=self.device)
        self.T_gt = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img) # Compute image gradient using Scharr Filter
        # Mask that indicates th presence of a significant gradient or an edge
        mask_v, mask_h = image_gradient_mask(gray_img) 
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        # if config["Dataset"]["type"] == "replica":
        #     row, col = 32, 32
        #     multiplier = edge_threshold
        #     _, h, w = self.original_image.shape
        #     for r in range(row):
        #         for c in range(col):
        #             block = img_grad_intensity[
        #                 :,
        #                 r * int(h / row) : (r + 1) * int(h / row),
        #                 c * int(w / col) : (c + 1) * int(w / col),
        #             ]
        #             th_median = block.median()
        #             block[block > (th_median * multiplier)] = 1
        #             block[block <= (th_median * multiplier)] = 0
        # else:
        
        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
