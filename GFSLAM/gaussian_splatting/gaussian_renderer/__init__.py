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

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2


def render(
    viewpoint_camera,
    viewpoint_camera_next,  # donguk
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor, 
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    flow_conf=None,
    use_flow=False,         # donguk
    update_flow_conf=False, # donguk
    use_color=True,  
    train_pose=False,
    downsampled_gsflow=False,
    use_flowraw_grad=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    # stable_count = torch.zeros_like(pc.get_opacity, dtype=torch.int32, device="cuda")
    # stable_status = torch.zeros_like(pc.get_opacity, dtype=torch.bool, device="cuda")
    error_per_gs = (torch.zeros_like(pc.get_opacity, dtype=torch.float, 
                                       requires_grad=True, device="cuda")+ 0)
    
    error_per_gs_2 = (torch.zeros_like(pc.get_opacity, dtype=torch.float, 
                                       requires_grad=True, device="cuda")+ 0)

    error_per_gs_3 = (torch.zeros_like(pc.get_opacity, dtype=torch.float, 
                                       requires_grad=True, device="cuda")+ 0)

    try:
        error_per_gs.retain_grad()
        error_per_gs_2.retain_grad()
        error_per_gs_3.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # donguk
    if viewpoint_camera == viewpoint_camera_next:
        use_flow = False
    
    if flow_conf is None:
        flow_conf = torch.ones(int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)).cuda()

    if downsampled_gsflow:
        W  = float(viewpoint_camera.image_width)
        H  = float(viewpoint_camera.image_height)
        Wd = int(W) // 8
        Hd = int(H) // 8
        sx = Wd / W
        sy = Hd / H
        fx_d = viewpoint_camera.fx * sx
        fy_d = viewpoint_camera.fy * sy
        cx_d = viewpoint_camera.cx * sx
        cy_d = viewpoint_camera.cy * sy
        FoVx_d = 2.0 * math.atan(Wd / (2.0 * fx_d))
        FoVy_d = 2.0 * math.atan(Hd / (2.0 * fy_d))
        tanfovx = math.tan(FoVx_d * 0.5)
        tanfovy = math.tan(FoVy_d * 0.5)
        downsampled_proj = getProjectionMatrix2(znear=0.01, zfar=100.0, 
                                               W=Wd, H=Hd, 
                                               fx=fx_d, fy=fy_d, 
                                               cx=cx_d, cy=cy_d).transpose(0,1).to(viewpoint_camera.device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if downsampled_gsflow==False else int(viewpoint_camera.image_height) // 8,
        image_width=int(viewpoint_camera.image_width) if downsampled_gsflow==False else int(viewpoint_camera.image_width) // 8,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier, # 1.0
        viewmatrix=viewpoint_camera.world_view_transform,
        viewmatrix_next=viewpoint_camera_next.world_view_transform, # donguk
        # projmatrix=viewpoint_camera.full_proj_transform,
        # projmatrix_next= viewpoint_camera_next.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix if downsampled_gsflow==False else downsampled_proj,
        sh_degree=pc.active_sh_degree,                                                                                                            
        campos=viewpoint_camera.camera_center,
        flow_image=viewpoint_camera.flow_image if downsampled_gsflow==False else torch.ones((2, Hd, Wd), device=viewpoint_camera.flow_image.device),
        # flow_conf=viewpoint_camera.flow_conf,   # donguk
        flow_conf=flow_conf if downsampled_gsflow==False else torch.ones((Hd, Wd), device=flow_conf.device),
        prefiltered=False,
        use_flow=use_flow,                  # donguk
        update_flow_conf=update_flow_conf,  # donguk
        use_color=use_color,  # always True
        train_pose=train_pose,
        use_flowraw_grad=use_flowraw_grad,
        debug=False,
        # keep_kernel_inputs=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity 

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2 
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            ) 
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rasterizer() -> forward() -> rasterize_gaussians() -> _RasterizeGaussians.apply()-> forward() 
        
    dummy_delta_rot = torch.zeros_like(viewpoint_camera.cam_rot_delta, device="cuda")
    dummy_delta_trans = torch.zeros_like(viewpoint_camera.cam_trans_delta, device="cuda")
    # dummy_delta_rot2 = torch.zeros_like(viewpoint_camera.cam_rot_delta, device="cuda")
    # dummy_delta_trans2 = torch.zeros_like(viewpoint_camera.cam_trans_delta, device="cuda")
    if mask is not None:
        rendered_image, radii, depth, silh, gsflow, flowcost, aux, aux2, aux3, n_touched, n_found, weights_sum = rasterizer(   # donguk
            means3D=means3D[mask], 
            means2D=means2D[mask], 
            opacities=opacity[mask],
            # stable_count = stable_count[mask],
            # stable_status = stable_status[mask],
            shs=shs[mask], 
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            error_per_gs=error_per_gs[mask],
            error_per_gs_2=error_per_gs_2[mask],
            error_per_gs_3=error_per_gs_3[mask],
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
            theta_next=viewpoint_camera_next.cam_rot_delta if use_flow else dummy_delta_rot, # donguk
            rho_next=viewpoint_camera_next.cam_trans_delta if use_flow else dummy_delta_trans, # donguk
        )
    else:
        rendered_image, radii, depth, silh, gsflow, flowcost, aux, aux2, aux3, n_touched, n_found, weights_sum = rasterizer(    # donguk
            means3D=means3D, 
            means2D=means2D, 
            opacities=opacity,
            # stable_count = stable_count,
            # stable_status = stable_status,
            shs=shs, 
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            error_per_gs=error_per_gs,
            error_per_gs_2=error_per_gs_2,
            error_per_gs_3=error_per_gs_3,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
            theta_next=viewpoint_camera_next.cam_rot_delta if use_flow else dummy_delta_rot, # donguk
            rho_next=viewpoint_camera_next.cam_trans_delta if use_flow else dummy_delta_trans, # donguk
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image, 
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "silh": silh,
        "gsflow": gsflow, # donguk
        "flowcost": flowcost, # donguk
        "error_per_gs": error_per_gs,
        "error_per_gs_2": error_per_gs_2,
        "error_per_gs_3": error_per_gs_3,
        "aux_image": aux,
        "aux_image2": aux2,
        "aux_image3": aux3,
        "n_touched": n_touched, 
        "n_found": n_found, 
        "weights_sum": weights_sum,
        # "raster_settings": raster_settings,
    }
