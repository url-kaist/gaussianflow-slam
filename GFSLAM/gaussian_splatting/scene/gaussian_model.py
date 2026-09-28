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

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH

from utils.slam_utils import vectorize_and_stack_values, vectorize_and_stack_grads, \
    normalize_stacked_tensor, reduce_max, reduce_median, reduce_min



class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0 # current level of detail 
        self.max_sh_degree = sh_degree # maximum level of detail available in the SH

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda") # Scale for covariance
        self._rotation = torch.empty(0, device="cuda") # Rotation for covariance
        self._opacity = torch.empty(0, device="cuda") # Opacity

        self.unique_kfIDs = torch.empty(0, dtype=torch.int, device="cuda") # tracking unique keyframe IDs
        self.n_obs = torch.empty(0, dtype=torch.int, device="cuda")
        self.n_found = torch.empty(0, dtype=torch.int, device="cuda")
        self.found_count = torch.empty(0, dtype=torch.int, device="cuda")

        # ---- Frozen-background tracking (fast_mode opt-in) --------------
        # Stores the mapping iteration_count at which each gaussian was
        # last "touched" (n_touched > 0 from any edge's render). Empty
        # until the feature is enabled via _freeze_tracking_enabled.
        self._last_seen_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        # Toggled by gs_mapper before each chunk based on YAML
        # ``fast_mode.freezing.enabled``. When False, all freeze-related
        # code below is a no-op (zero per-iter overhead).
        self._freeze_tracking_enabled = False
        # Caller (gs_mapper) sets this to its current ``iteration_count``
        # before any densify/render call so freshly-created gaussians
        # get the right "creation iter" stamp.
        self._freeze_current_iter = 0

        self.optimizer = None

        # for positive scaling, bounded opacity (0-1)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = self.build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None # Pointcloud input for gaussian

        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, found_filter=None):
        cam = cam_info
        image_ab = (cam.original_image - cam.exposure_b) / (torch.exp(cam.exposure_a)) 
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                # depth_raw = np.empty((cam.image_height, cam.image_width))
                # if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, found_filter, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, found_filter=None, init=False):

        if init:
            downsample_factor = self.config["Training"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Training"]["pcd_downsample"]

        point_size = self.config["Training"]["point_size"]
        if "adaptive_pointsize" in self.config["Training"]:
            if self.config["Training"]["adaptive_pointsize"]:
                # point_size = min(0.05, point_size * np.median(depth))
                point_size = max(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        # self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

        if fused_point_cloud.shape[0] > 0:
            fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
            features = (
                torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
                .float()
                .cuda()
            )
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            pcd_from_insert_region = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
            if found_filter is not None:
                pcd_in_frustum = self._xyz[found_filter]
                pcd_for_scale = torch.cat([pcd_in_frustum, pcd_from_insert_region], dim=0)
            else:
                pcd_for_scale = pcd_from_insert_region
            dist2 = (
                torch.clamp_max(
                torch.clamp_min(
                    distCUDA2(pcd_for_scale),
                    0.0000001,
                ), 0.1
                )
                # * point_size
            )
            scales = torch.log(torch.sqrt(dist2))[..., None]
            if found_filter is not None:
                scales = scales.narrow(0, pcd_in_frustum.shape[0], pcd_from_insert_region.shape[0])
                # scales = scales[pcd_in_frustum.shape[0]: pcd_in_frustum.shape[0] + pcd_for_scale.shape[0]]
            if not self.isotropic:
                scales = scales.repeat(1, 3)

            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(
                0.8
                * torch.ones(
                    (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
                )
            )
            return fused_point_cloud, features, scales, rots, opacities
        else:
            dummy = torch.tensor([]).float().cuda()
            return fused_point_cloud, dummy, dummy, dummy, dummy

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
    
    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))
        
        new_unique_kfIDs = torch.ones((new_xyz.shape[0]), dtype=torch.int, device=new_xyz.device) * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0]), dtype=torch.int, device=new_xyz.device)
        new_n_found = torch.zeros((new_xyz.shape[0]), dtype=torch.int, device=new_xyz.device)
        new_found_count = torch.zeros((new_xyz.shape[0]), dtype=torch.int, device=new_xyz.device)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_n_found=new_n_found,
            new_found_count=new_found_count,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, found_filter=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, found_filter=found_filter)
        )

        if fused_point_cloud.shape[0] < 5:
            return
        
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def cutoff_gradients(self, filter):
        if filter is None or filter.numel() == 0:
            return

        filter.to(self._xyz.device)

        def safe_mul(tensor, shape):
            g = tensor.grad
            if g is not None:
                m = filter.view(shape).to(g.device).to(g.dtype)
                g.mul_(m)

        safe_mul(self._xyz, (-1, 1))              # [N, 3]
        safe_mul(self._features_dc, (-1, 1, 1))   # [N, 1, 3]
        safe_mul(self._features_rest, (-1, 1, 1)) # [N, 3, K]
        safe_mul(self._opacity, (-1, 1))          # [N, 1]
        safe_mul(self._scaling, (-1, 1))          # [N, 3]
        safe_mul(self._rotation, (-1, 1))         # [N, 4]


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense # control the density of points or Gaussians being actively used or optimized

        l = [
            {   # params for learning xyz 
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {   # params for learning SH coeff 
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {   # params for learning SH coeff 
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {   # params for learning opacity 
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                # params for learning scaling 
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                # params for learning rotation 
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # not used
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale # initial learning rate
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale # final learning rate
        self.lr_delay_mult = training_args.position_lr_delay_mult # Learning Rate Delay Multiplier
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration, is_refinement=False):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init, # lr_init=self.lr_init * 10.0,
                    lr_final=self.lr_final, # * 0.1 if is_refinement else self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps, #+ 1000
                )

                param_group["lr"] = lr
                # return lr
                
            if param_group["name"] == "scaling" and is_refinement:
                curr_lr = param_group["lr"]
                param_group["lr"] = 0.005
        return lr

    def masked_optimizer_step(self, mask):
        """Apply Adam to only the rows where ``mask`` is True (visible Gaussians).

        Each registered Gaussian param has shape ``(N, ...)`` where N is the
        Gaussian count; ``mask`` is a 1-D bool tensor of length N. Rows where
        ``mask`` is False are left completely untouched (param data, exp_avg,
        exp_avg_sq, step counter are all frozen for those rows).

        Implementation note: step counter is kept as a Python int (not a
        device tensor) so we never need ``.item()`` mid-step (which would
        force a CPU/GPU sync). Per-tensor Adam math is grouped into one
        ``_foreach_*`` call per arithmetic op so we hit ~5 kernel launches
        for the whole step instead of 6 × per-param.
        """
        import math
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        if mask.numel() == 0:
            return
        # Cheap any() — visibility_filter rarely all-False; the kernel for
        # `mask.any()` is small enough to ignore vs the cost of the step.
        if not mask.any():
            return
        idx = torch.nonzero(mask, as_tuple=True)[0]

        # Collect per-group sliced tensors so we can apply foreach Adam math
        # to a list of params with one kernel per op.
        sliced_params, sliced_grads, sliced_m, sliced_v = [], [], [], []
        owners, lr_ratios, lrs, denom_eps = [], [], [], []
        for group in self.optimizer.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.shape[0] != mask.shape[0]:
                    # Non-Gaussian-shaped tensor (e.g. camera delta) — not
                    # our responsibility, fall through to the regular path.
                    continue
                state = self.optimizer.state[p]
                if len(state) == 0:
                    # store step as a Python int so we never .item() on a
                    # CUDA tensor and stall the stream.
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                if not isinstance(state["step"], int):
                    # Coerce existing tensor-step (e.g. left over from a
                    # previous standard optimizer.step) to int.
                    state["step"] = int(state["step"].item()) if torch.is_tensor(state["step"]) else int(state["step"])
                state["step"] += 1
                step_t = state["step"]
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t

                sliced_params.append(p.data.index_select(0, idx))
                sliced_grads.append(p.grad.index_select(0, idx))
                sliced_m.append(state["exp_avg"].index_select(0, idx))
                sliced_v.append(state["exp_avg_sq"].index_select(0, idx))
                owners.append((p, state, beta1, beta2, eps))
                lrs.append(lr / bc1)
                denom_eps.append(eps)
                lr_ratios.append(math.sqrt(bc2))

        if not sliced_params:
            return

        # m = beta1 * m + (1-beta1) * g    (per-tensor since betas differ)
        for m, g, (_, _, beta1, _, _) in zip(sliced_m, sliced_grads, owners):
            m.mul_(beta1).add_(g, alpha=1 - beta1)
        # v = beta2 * v + (1-beta2) * g*g
        for v, g, (_, _, _, beta2, _) in zip(sliced_v, sliced_grads, owners):
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        # denom = sqrt(v) / sqrt(bc2) + eps   (per tensor)
        denoms = []
        for v, scale, eps in zip(sliced_v, lr_ratios, denom_eps):
            denoms.append((v.sqrt() / scale).add_(eps))

        # p -= lr/bc1 * m / denom    (foreach on list-of-scalars)
        torch._foreach_addcdiv_(sliced_params, sliced_m, denoms,
                                [-lr_ for lr_ in lrs])

        # Scatter back into the full tensors.
        for (p, state, *_), p_s, m_s, v_s in zip(owners, sliced_params, sliced_m, sliced_v):
            p.data.index_copy_(0, idx, p_s)
            state["exp_avg"].index_copy_(0, idx, m_s)
            state["exp_avg_sq"].index_copy_(0, idx, v_s)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "gaussians.ply")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.n_found = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.found_count = torch.zeros((self._xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        n_init_points = self.get_xyz.shape[0]
        padded_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_mask[: mask.shape[0]] = mask.squeeze()
        valid_points_mask = ~padded_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask]
        self.n_obs = self.n_obs[valid_points_mask]
        self.n_found = self.n_found[valid_points_mask]
        self.found_count = self.found_count[valid_points_mask]
        if self._freeze_tracking_enabled and self._last_seen_iter.numel() > 0:
            # Keep _last_seen_iter aligned with _xyz row count.
            self._last_seen_iter = self._last_seen_iter[valid_points_mask]

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def stamp_uids(self, min_uid, filter):
        self.unique_kfIDs[filter] = min_uid

    def reset_opacity_by_filter(
        self, total_filter
    ):  ##Reset opacity for non-found gaussians\
        opacity_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.71)
        opacity_new[total_filter] = inverse_sigmoid(self.get_opacity[total_filter])

        # non_filter = ~total_filter
        # opacity_non_found = self.get_opacity[non_filter]
        # crucial_mask = opacity_non_found >= 0.5
        # invalid_mask = opacity_non_found < 0.5
        # opacity_non_found[crucial_mask] = inverse_sigmoid(torch.ones_like(opacity_non_found[crucial_mask]) * 0.7)
        # # opacity_non_found[invalid_mask] = inverse_sigmoid(torch.ones_like(opacity_non_found[invalid_mask]) * 0.1)
        # opacity_non_found[invalid_mask] = inverse_sigmoid(opacity_non_found[invalid_mask] * 0.5)
        # opacity_new = inverse_sigmoid(self.get_opacity)
        # opacity_new[non_filter] = opacity_non_found

        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]] 
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_n_found=None,
        new_found_count=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids))#.int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs))#.int()
        if new_n_found is not None:
            self.n_found = torch.cat((self.n_found, new_n_found))
        if new_found_count is not None:
            self.found_count = torch.cat((self.found_count, new_found_count))
        if self._freeze_tracking_enabled:
            K = new_xyz.shape[0]
            fresh = torch.full(
                (K,), int(self._freeze_current_iter),
                dtype=torch.int32, device=self._xyz.device,
            )
            if self._last_seen_iter.numel() == 0:
                # First time enabling — stamp every existing row too
                # (otherwise the pre-existing gaussians would have a
                # value of "0" and be flagged frozen on iter 1).
                self._last_seen_iter = torch.full(
                    (self._xyz.shape[0],), int(self._freeze_current_iter),
                    dtype=torch.int32, device=self._xyz.device,
                )
            else:
                self._last_seen_iter = torch.cat([self._last_seen_iter, fresh])

    # ---- Frozen-background helpers (fast_mode opt-in) ----------------------
    def update_last_seen(self, touched_mask):
        """Stamp _last_seen_iter[touched] = current_iter.

        ``touched_mask`` is a bool/int tensor where True/>0 means the
        gaussian was rendered (n_touched > 0) by at least one edge in
        the current iter. No-op when freeze tracking is disabled.
        """
        if not self._freeze_tracking_enabled:
            return
        N = self._xyz.shape[0]
        if self._last_seen_iter.numel() != N:
            self._last_seen_iter = torch.full(
                (N,), int(self._freeze_current_iter),
                dtype=torch.int32, device=self._xyz.device,
            )
        m = touched_mask
        if m.dtype != torch.bool:
            m = m > 0
        if m.shape[0] != N:
            # Render kernel can return n_touched of length M ≤ N if
            # gaussians were appended mid-pass. Pad False to length N.
            pad = torch.zeros(N - m.shape[0], dtype=torch.bool, device=m.device)
            m = torch.cat([m, pad])
        self._last_seen_iter[m] = int(self._freeze_current_iter)

    def get_frozen_mask(self, max_age):
        """Return bool mask of gaussians not seen in last ``max_age`` iters."""
        if not self._freeze_tracking_enabled:
            return None
        if self._last_seen_iter.numel() != self._xyz.shape[0]:
            return None
        age = int(self._freeze_current_iter) - self._last_seen_iter
        return age > int(max_age)

    def zero_grads_for_frozen(self, frozen_mask):
        """Zero ``.grad`` of all gaussian-param rows flagged frozen.

        Adam.step() still iterates but takes ~0 update step since grad=0
        (and Adam's m/v decay geometrically). Cheap way to "freeze"
        without restructuring the parameter list.
        """
        for p in (self._xyz, self._features_dc, self._features_rest,
                  self._opacity, self._scaling, self._rotation):
            if p.grad is not None:
                p.grad[frozen_mask] = 0

    def split_for_mask(self, selected_pts_mask, kf_uid=-1, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_mask = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        padded_mask[: selected_pts_mask.shape[0]] = selected_pts_mask.squeeze()
        selected_pts_mask = padded_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (N) # (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        orig_ids = self.unique_kfIDs[selected_pts_mask]
        P = orig_ids.shape[0]
        if kf_uid < 0:
            new_kf_id = orig_ids.repeat(N)
        else:
            repeated_uid = torch.full((P * (N - 1),), kf_uid, dtype=orig_ids.dtype, device=orig_ids.device)
            new_kf_id = torch.cat([orig_ids, repeated_uid], dim=0)

        # if kf_uid < 0:
        #     new_kf_id = self.unique_kfIDs[selected_pts_mask].repeat(N)
        # else:
        #     new_kf_id = torch.ones_like(self.unique_kfIDs[selected_pts_mask]).repeat(N) * kf_uid
        # new_kf_id = self.unique_kfIDs[selected_pts_mask].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask].repeat(N)
        new_n_found = self.n_found[selected_pts_mask].repeat(N)
        new_found_count = self.found_count[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_n_found=new_n_found,
            new_found_count=new_found_count,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_prune_by_error(self, radii_acm, error_per_gs_acm, weight_per_gs_acm, error_per_gs_3_acm, \
                                   viewspace_point_tensor_acm, found_count, total_found_filter, do_prune=True, 
                                   kf_uid = -1, prune_kf_id_thres=-1):
        stacked_epg = vectorize_and_stack_grads(error_per_gs_acm)  # error per gs
        stacked_epg3 = vectorize_and_stack_grads(error_per_gs_3_acm) # weight per gs
        stacked_radiis = vectorize_and_stack_values(radii_acm) # (max-)radii per gs
        stacked_2Dpg = vectorize_and_stack_grads(viewspace_point_tensor_acm) # 2Dpos-grad per GS

        stacked_wpg = vectorize_and_stack_grads(weight_per_gs_acm) # weight per gs
        stacked_nepg = normalize_stacked_tensor(stacked_epg, stacked_wpg, total_found_filter) # normalized error per gs
        stacked_nepg3 = normalize_stacked_tensor(stacked_epg3, stacked_wpg, total_found_filter) # normalized error per gs
        # stacked_nepg_by_radii = squared_normalize_stacked_tensor(stacked_epg, stacked_radiis, total_found_filter)
        # stacked_nepg3_by_radii = squared_normalize_stacked_tensor(stacked_epg3, stacked_radiis, total_found_filter)

        max_radiis = reduce_max(stacked_radiis)
        # median_radiis = reduce_median(stacked_radiis)
        # mean_radiis = reduce_mean(stacked_radiis, found_count, total_found_filter)
        min_radiis = reduce_min(stacked_radiis)
        ##################################     

        # max_epg = reduce_max(stacked_epg)
        median_epg = reduce_median(stacked_epg)
        ##################################

        # max_epg3 = reduce_max(stacked_epg3)
        ##################################

        # max_nepg = reduce_max(stacked_nepg)
        median_nepg = reduce_median(stacked_nepg)
        # max_epg_by_radii = reduce_median(stacked_nepg_by_radii)
        # normalized_epg = max_epg / (mean_radiis**2)
        # normalized_epg[normalized_epg.isnan()] = 0.0
        ##################################

        # max_nepg3 = reduce_max(stacked_nepg3)
        median_nepg3 = reduce_median(stacked_nepg3)
        # max_epg3_by_radii = reduce_median(stacked_nepg3_by_radii)

        # normalized_epg3 = max_epg3 / (mean_radiis**2)
        # normalized_epg3[normalized_epg3.isnan()] = 0.0
        ##################################

        max_2Dpg = reduce_max(stacked_2Dpg)
        # median_2Dpg = reduce_median(stacked_2Dpg)
        # mean_2Dpg = reduce_mean(stacked_2Dpg, found_count, total_found_filter)
        # min_2Dpg = reduce_min(stacked_2Dpg)
        ##################################        

        del stacked_epg, stacked_nepg
        del stacked_epg3, stacked_nepg3
        del stacked_radiis
        del stacked_2Dpg
        del stacked_wpg

        #######################
        # Usage: median_epg, median_radiis, median_2Dpg, median_nepg
        #######################
        error_mask = median_epg > self.config["Training"]["error_th"]
        radii_split_mask = max_radiis > self.config["Training"]["radii_split"]
        posgrad_less_mask = max_2Dpg < self.config["Training"]["posgrad_th"]
        norm_error_split_mask = median_nepg > self.config["Training"]["norm_error_split"]
        found_consensus_mask = found_count >= self.config["Training"]["found_consensus_th"]

        if not do_prune:
            split1_mask = torch.logical_and(error_mask, radii_split_mask)
            split1_mask = torch.logical_and(split1_mask, posgrad_less_mask)
            split2_mask = torch.logical_and(norm_error_split_mask, radii_split_mask)
            split3_mask = max_radiis > self.config["Training"]["radii_cut_always"]
            final_split_mask = (split1_mask | split2_mask | split3_mask) & total_found_filter

            self.split_for_mask(final_split_mask, kf_uid=kf_uid)

        elif do_prune:
            #######################
            # Usage: median_radiis, median_nepg, median_nepg3, found_count
            #######################

            radii_tiny_mask = min_radiis < self.config["Training"]["radii_tiny"]
            radii_prune_mask = min_radiis < self.config["Training"]["radii_prune"]
            norm_error_prune_mask_img = median_nepg > self.config["Training"]["norm_error_prune_img"]
            norm_error_prune_mask_flow = median_nepg3 > self.config["Training"]["norm_error_prune_flow"]
            norm_error_prune_mask_img_too_large = median_nepg > self.config["Training"]["norm_error_prune"]
            found_consensus_mask = found_count >= self.config["Training"]["found_consensus_th"]

            prune_mask_opa = (self.get_opacity < self.config["Training"]["opacity_th"]).squeeze()
            prune_mask_opa = torch.logical_and(prune_mask_opa, total_found_filter)
            prune_mask_tiny = torch.logical_and(
                radii_tiny_mask,
                torch.logical_or(norm_error_prune_mask_img, norm_error_prune_mask_flow),
            )
            prune_mask_large_error = torch.logical_and(radii_prune_mask, norm_error_prune_mask_img_too_large)

            final_prune_mask = torch.logical_or(prune_mask_tiny, prune_mask_large_error)
            final_prune_mask = torch.logical_and(final_prune_mask, total_found_filter)
            final_prune_mask = torch.logical_or(final_prune_mask, prune_mask_opa)

            self.prune_points(final_prune_mask)

            # self.reset_opacity_by_filter(~final_prune_mask)
            # self.prune_points(prune_mask_opa)
