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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from dataclasses import dataclass
import math

import cv2
import numpy as np
import os
import re

# def make_index_path_generator(base_path: str):
#     """
#     base_path: .../N.ext (예: /a/b/1.png)
#     반환: next_path() 함수
#       - 호출할 때마다 (candidate_path, idx) 반환
#       - 파일 존재 여부와 무관하게 idx가 1씩 증가
#     """
#     dirn = os.path.dirname(base_path)
#     base = os.path.basename(base_path)
#     name, ext = os.path.splitext(base)

#     m = re.fullmatch(r"(\d+)", name)
#     if not m:
#         raise ValueError("파일명이 숫자여야 합니다. 예: /path/1.png")

#     os.makedirs(dirn, exist_ok=True)

#     i = int(m.group(1))  # 시작 인덱스

#     def next_path():
#         nonlocal i
#         candidate = os.path.join(dirn, f"{i}{ext}")
#         idx = i
#         i += 1
#         return candidate, idx

#     return next_path

# next_path = make_index_path_generator("/workspace/CUDAProjects/DROID-SLAM/grad_debug/1.jpg")

# def _to_hw_heat(x: torch.Tensor) -> np.ndarray:
#     """(H,W) float32 numpy로."""
#     x = x.detach()
#     if x.is_cuda:
#         torch.cuda.synchronize()
#     x = x.squeeze()
#     if x.dim() != 2:
#         raise ValueError(f"heat tensor should be 2D after squeeze, got {tuple(x.shape)}")
#     x = x.float().cpu().numpy()
#     x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#     return x

# def _to_hwc_color_u8(color01: torch.Tensor, assume_rgb=True) -> np.ndarray:
#     """
#     color01: [0,1] 범위의 torch.Tensor
#       지원 shape: (H,W,3), (3,H,W), (1,3,H,W), (1,H,W,3)
#     return: uint8 BGR (H,W,3)  (cv2용)
#     """
#     c = color01.detach()
#     if c.is_cuda:
#         torch.cuda.synchronize()

#     # 배치 차원 제거
#     if c.dim() == 4 and c.shape[0] == 1:
#         c = c[0]

#     c = c.squeeze()

#     if c.dim() != 3:
#         raise ValueError(f"color tensor should be 3D after squeeze, got {tuple(c.shape)}")

#     # CHW -> HWC
#     if c.shape[0] == 3 and c.shape[-1] != 3:
#         c = c.permute(1, 2, 0)

#     if c.shape[-1] != 3:
#         raise ValueError(f"color last dim should be 3 (HWC), got {tuple(c.shape)}")

#     c = c.float().clamp(0.0, 1.0).cpu().numpy()
#     rgb_u8 = (c * 255.0 + 0.5).astype(np.uint8)

#     if assume_rgb:
#         bgr_u8 = rgb_u8[..., ::-1]  # RGB -> BGR
#     else:
#         bgr_u8 = rgb_u8

#     return bgr_u8

# def save_jet_image_cv2(
#     out_grad_blending: torch.Tensor,
#     save_path: str,
#     color01: torch.Tensor = None,   # (optional) 원본 컬러 이미지 [0,1]
#     alpha: float = 0.5,             # heatmap 가중치 (0~1)
#     clip_percentile=(1, 99),
#     eps: float = 1e-12,
#     assume_color_rgb: bool = True,
# ):
#     """
#     - color01=None: heatmap(JET)만 저장
#     - color01!=None: color(복원)과 heatmap을 overlay로 블렌딩해서 저장
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # 1) heat 값 -> [0,1] 정규화 -> gray8 -> jet
#     x = _to_hw_heat(out_grad_blending)

#     if clip_percentile is not None:
#         lo, hi = np.percentile(x, clip_percentile)
#         x = np.clip(x, lo, hi)

#     mn, mx = x.min(), x.max()
#     x01 = (x - mn) / (mx - mn + eps)
#     gray8 = (x01 * 255.0 + 0.5).astype(np.uint8)
#     jet_bgr = cv2.applyColorMap(gray8, cv2.COLORMAP_JET)

#     # 2) color 복원 + overlay
#     if color01 is not None:
#         color_bgr = _to_hwc_color_u8(color01, assume_rgb=assume_color_rgb)

#         if color_bgr.shape[:2] != jet_bgr.shape[:2]:
#             raise ValueError(f"Size mismatch: color {color_bgr.shape[:2]} vs heat {jet_bgr.shape[:2]}")

#         alpha = float(alpha)
#         alpha = 0.0 if alpha < 0 else 1.0 if alpha > 1 else alpha

#         blended = cv2.addWeighted(jet_bgr, alpha, color_bgr, 1.0 - alpha, 0.0)
#         cv2.imwrite(save_path, blended)
#     else:
#         cv2.imwrite(save_path, jet_bgr)

#     return save_path


# def safe_sum_view6(tensor, name="grad_tau"):
#     if tensor is None or tensor.numel() == 0:
#         print(f"[Warning] {name} is empty or None.")
#         return torch.zeros(6, device=tensor.device if tensor is not None else 'cuda')

#     # Ensure contiguous
#     if not tensor.is_contiguous():
#         tensor = tensor.contiguous()

#     # Check shape validity
#     if tensor.numel() % 6 != 0:
#         print(f"[Warning] {name} shape {tensor.shape} is not divisible by 6.")
#         return torch.zeros(6, device=tensor.device)

#     # Check NaN or Inf
#     if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#         print(f"[Warning] {name} contains NaN or Inf. Replacing with zeros.")
#         return torch.zeros(6, device=tensor.device)

#     return torch.sum(tensor.view(-1, 6), dim=0)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    # stable_count,
    # stable_status,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    error_per_gs,
    error_per_gs_2,
    error_per_gs_3,
    theta,
    rho,
    theta_next,
    rho_next,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,        # backward와 format을 맞춰주기 위한 input 
        # stable_count,
        # stable_status,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        error_per_gs,
        error_per_gs_2,
        error_per_gs_3,
        theta,
        rho,
        theta_next,     
        rho_next,       
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, # ctx: torch.autograd.Function에서 사용하는 context개체.
        means3D,
        means2D,        # backward와 format을 맞춰주기 위한 input
        # stable_count,
        # stable_status,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        error_per_gs,   
        error_per_gs_2,
        error_per_gs_3,
        theta,          
        rho,            
        theta_next,     
        rho_next,       
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.viewmatrix_next,    # donguk
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.flow_image, # donguk
            raster_settings.flow_conf,  # donguk
            raster_settings.prefiltered,
            raster_settings.use_color,
            raster_settings.use_flow,   # donguk
            raster_settings.train_pose,
            raster_settings.update_flow_conf,   # donguk
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, silh, gsflow, flowcost, radii, geomBuffer, binningBuffer, imgBuffer, n_touched, n_found, weights_sum = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, silh, gsflow, flowcost, radii, geomBuffer, binningBuffer, imgBuffer, n_touched, n_found, weights_sum = _C.rasterize_gaussians(*args)

        aux = torch.full((1, raster_settings.image_height, raster_settings.image_width), 0.0, device=means3D.device, dtype=torch.float32)
        aux2 = torch.full((1, raster_settings.image_height, raster_settings.image_width), 0.0, device=means3D.device, dtype=torch.float32)
        aux3 = torch.full((1, raster_settings.image_height, raster_settings.image_width), 0.0, device=means3D.device, dtype=torch.float32)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, gsflow, sh, geomBuffer, binningBuffer, imgBuffer) # donguk : gsflow # stable_count, stable_status
        # ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, gsflow, sh, geomBuffer, binningBuffer, imgBuffer, color) # donguk : gsflow # stable_count, stable_status

        # print("Number of weights_sum > 0: ", (weights_sum > 0).sum().item(), " with total num: ", weights_sum.numel())
        # if raster_settings.keep_kernel_inputs:
        #     kernel_inputs = tuple(
        #         t.detach() if isinstance(t, torch.Tensor) else t
        #         for t in (
        #             colors_precomp, means3D, scales, rotations,
        #             cov3Ds_precomp, radii, gsflow, sh,
        #             geomBuffer, binningBuffer, imgBuffer, num_rendered,
        #             weights_sum
        #         )
        #     )
        # else:
        #     kernel_inputs = None

        # return color, radii, depth, silh, gsflow, flowcost, aux, aux2, aux3, n_touched, n_found, weights_sum, kernel_inputs # donguk
        return color, radii, depth, silh, gsflow, flowcost, aux, aux2, aux3, n_touched, n_found, weights_sum

    # backwar의 인풋과 forward의 아웃풋이 이어져야함
    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_silh, grad_out_gsflow, grad_out_flowcost, 
                 grad_out_aux, grad_out_aux2, grad_out_aux3, 
                #  grad_out_n_touched, grad_out_n_found, grad_weights_sum, grad_kernel_inputs): # donguk
                grad_out_n_touched, grad_out_n_found, grad_weights_sum): # donguk

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        # colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, gsflow, sh, geomBuffer, binningBuffer, imgBuffer, color = ctx.saved_tensors # stable_count, stable_status
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, gsflow, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors # stable_count, stable_status

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                # stable_count,
                # stable_status,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.viewmatrix_next,
                raster_settings.projmatrix_raw,
                raster_settings.flow_image,         # donguk
                raster_settings.flow_conf,          # donguk
                gsflow,                             # donguk      
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_silh,
                grad_out_gsflow,
                grad_out_flowcost,
                grad_out_aux,
                grad_out_aux2,
                grad_out_aux3,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.use_color,
                raster_settings.use_flow,   
                raster_settings.train_pose,
                raster_settings.use_flowraw_grad)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                # grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, \
                grad_sh, grad_scales, grad_rotations, grad_tau, grad_tau_next, \
                grad_error_per_gs, grad_error_per_gs_2, grad_error_per_gs_3 = _C.rasterize_gaussians_backward(*args)
                # grad_error_per_gs, grad_error_per_gs_2, grad_error_per_gs_3 = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, \
            grad_sh, grad_scales, grad_rotations, grad_tau, grad_tau_next, \
            grad_error_per_gs, grad_error_per_gs_2, grad_error_per_gs_3 = _C.rasterize_gaussians_backward(*args)
            # grad_error_per_gs, grad_error_per_gs_2, grad_error_per_gs_3, out_grad_blending = _C.rasterize_gaussians_backward(*args)
            #  grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)

        # save_path, idx = next_path()
        # if idx % 100 == 0:
        #     save_jet_image_cv2(out_grad_blending, save_path, color01=color, alpha=0.7)
        #     flow_path = os.path.splitext(save_path)[0] + "_flow.jpg"
        #     flow_diff = torch.linalg.vector_norm( (gsflow - raster_settings.flow_image) * raster_settings.flow_conf, dim=0)
        #     save_jet_image_cv2(flow_diff, flow_path)

        ###################################
        # 여기서 color, depth, flow등을 저장하고
        # dL_dmeans3D_blending 이미지도 저장하거나 겹쳐서 저장한다음 봐보기
        ###################################

        # if raster_settings.train_pose:
        #     raster_settings.grad_tau_raw = grad_tau.clone()
        #     raster_settings.grad_tau_next_raw = grad_tau_next.clone()
        # print("grad_tau: ", grad_tau)
        # print("grad_tau_next: ", grad_tau_next)

        # grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        # print("grad_tau: ", grad_tau)
        # print("grad_tau_next: ", grad_tau_next)

        # grad_tau = safe_sum_view6(grad_tau, "grad_tau")
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        # grad_tau_next = safe_sum_view6(grad_tau_next, "grad_tau_next")
        grad_tau_next = torch.sum(grad_tau_next.view(-1, 6), dim=0)
        grad_rho_next = grad_tau_next[:3].view(1, -1)
        grad_theta_next = grad_tau_next[3:].view(1, -1)

        # donguk TODO: grad_rho_next, grad_theta_next, error_per_gs에 대한 gradient 추가 예정..
        # stable_count, stable_status에 관한건 None으로 하면됨.
        grads = (
            grad_means3D,
            grad_means2D,
            # None, # stable_count, # donguk
            # None, # stable_status, # donguk
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_error_per_gs,
            grad_error_per_gs_2,
            grad_error_per_gs_3,
            grad_theta,
            grad_rho,
            grad_theta_next,
            grad_rho_next,
            None,
        )

        return grads

# @dataclass
# class GaussianRasterizationSettings:  
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    viewmatrix_next : torch.Tensor # donguk
    # projmatrix : torch.Tensor
    # projmatrix_next : torch.Tensor  # donguk
    projmatrix_raw : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    flow_image : torch.Tensor   # donguk
    flow_conf : torch.Tensor    # donguk
    prefiltered : bool
    use_flow : bool             # donguk
    update_flow_conf : bool     # donguk
    use_color : bool
    train_pose : bool
    use_flowraw_grad : bool
    debug : bool
    # keep_kernel_inputs : bool
    # grad_tau_raw: torch.Tensor = None
    # grad_tau_next_raw: torch.Tensor = None

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        # self._colors_precomp_cached = None
        # self._mean3Ds_cached = None
        # self._scales_cached = None
        # self._rotations_cached = None
        # self._cov3Ds_precomp_cached = None
        # self._radii_cached = None
        # self._gsflow_cached = None
        # self._sh_cached = None
        # self._geomBuffer_cached = None
        # self._binningBuffer_cached = None
        # self._imgBuffer_cached = None
        # self._num_rendered_cached = None
        self.weights_sum = None

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, #stable_count, stable_status, 
                shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, 
                error_per_gs=None, error_per_gs_2=None, error_per_gs_3=None, 
                theta=None, rho=None, theta_next=None, rho_next=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        

        # Invoke C++/CUDA rasterization routine
        rasterize_output = rasterize_gaussians(
                            means3D,
                            means2D,
                            # stable_count,
                            # stable_status,
                            shs,
                            colors_precomp,
                            opacities,
                            scales, 
                            rotations,
                            cov3D_precomp,
                            error_per_gs,
                            error_per_gs_2,
                            error_per_gs_3,
                            theta,
                            rho,
                            theta_next,
                            rho_next,
                            raster_settings, 
                        )
        *main_outputs, weights_sum = rasterize_output
        # *main_outputs, weights_sum, kernel_inputs = rasterize_output
        # if raster_settings.keep_kernel_inputs:
        #     self._colors_precomp_cached, self._mean3Ds_cached, self._scales_cached, self._rotations_cached, \
        #     self._cov3Ds_precomp_cached, self._radii_cached, self._gsflow_cached, self._sh_cached, \
        #     self._geomBuffer_cached, self._binningBuffer_cached, self._imgBuffer_cached, \
        #     self._num_rendered_cached, self.weights_sum = kernel_inputs
        
        # return tuple(main_outputs)
        return tuple(main_outputs) + (weights_sum,)
    
        # return rasterize_gaussians(
        #     means3D,
        #     means2D,
        #     # stable_count,
        #     # stable_status,
        #     shs,
        #     colors_precomp,
        #     opacities,
        #     scales, 
        #     rotations,
        #     cov3D_precomp,
        #     error_per_gs,
        #     error_per_gs_2,
        #     error_per_gs_3,
        #     theta,
        #     rho,
        #     theta_next,
        #     rho_next,
        #     raster_settings, 
        # )
    
    # @torch.no_grad()
    # def compute_jacobian_hessian(self):
    #     if self.raster_settings.keep_kernel_inputs is False:
    #         raise RuntimeError("To call Jacobian/Hessian kernel, please set keep_kernel_inputs=True in rasterization settings.")
    #     if self._geomBuffer_cached is None:
    #         raise RuntimeError("You must call forward() at least once before computing Jacobian/Hessian.")    

