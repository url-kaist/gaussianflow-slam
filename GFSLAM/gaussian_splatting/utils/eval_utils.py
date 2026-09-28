import json
import os

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.logging_utils import Log
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import ssim, psnr
from gaussian_splatting.utils.graphics_utils import focal2fov
from utils.camera_utils import Camera

cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")


def _save_imgs_enabled():
    return os.environ.get("GFSLAM_SAVE_KF_IMAGES", "1") != "0"


def eval_rendering(
    image_stream,
    traj_full,
    kf_indices,
    gaussians,
    pipeline_params,
    background,
    projection_matrix,
    save_dir,
    iteration="final",
):
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []

    save_imgs = _save_imgs_enabled()
    image_save_dir = f'{save_dir}/renders/image_{iteration}'
    depth_save_dir = f'{save_dir}/renders/depth_{iteration}'
    if save_imgs:
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)

    # vis_save_dir = f'{save_dir}/renders/vis_{iteration}'
    # os.makedirs(vis_save_dir, exist_ok=True)

    render_idx = 0
    dump_tstamp = -1
    fake_gt_pose = np.eye(4)
    device = "cuda"
    # for i, (idx, image) in enumerate(gtimages.items()):
    for e, (tstamp, image, intrinsic, is_last) in enumerate(image_stream):

        # if iteration != "after_opt":
        if True:
            if tstamp in kf_indices:
            # if tstamp not in kf_indices:
                continue

            render_idx += 1
            if render_idx % 5 != 0:
                continue

        fx, fy, cx, cy = intrinsic.cpu().numpy()
        ht, wd = image.shape[-2], image.shape[-1]
        fovx = focal2fov(fx, wd)
        fovy = focal2fov(fy, ht)
        frame = Camera(tstamp, image.squeeze()/255.0, None,
                       fake_gt_pose, projection_matrix,
                       fx, fy, cx, cy,
                       fovx, fovy, ht, wd, device)
        frame.update_RT(traj_full[tstamp][:3,:3], traj_full[tstamp][:3, 3])
        gtimage = frame.original_image.cuda()
        rendering = render(frame, frame,
                           gaussians, pipeline_params, background,
                           use_flow=False)
        image = torch.clamp(rendering["render"], 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()

        if save_imgs:
            pred = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{image_save_dir}/pred_{tstamp:06d}.jpg', pred)

            depth_disp = np.nan_to_num(depth.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            depth_disp = np.clip(depth_disp, 0, np.percentile(depth_disp, 99))
            depth_norm = cv2.normalize(depth_disp, None, 0, 255, cv2.NORM_MINMAX)
            if depth_norm.ndim == 3:
                depth_norm = depth_norm.squeeze()
            depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(f"{depth_save_dir}/jet_{tstamp:06d}.png", depth_color)

            gt = (gtimage.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{image_save_dir}/gt_{tstamp:06d}.jpg', gt)

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0)).mean()
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    # output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def eval_rendering_kf(total_kf_num,
                      gs_viewpoints,
                      tstamps,
                      gaussians,
                      pipeline_params,
                      background,
                      save_dir,
                      iteration_count):
    
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []

    save_imgs = _save_imgs_enabled()
    image_save_dir = f'{save_dir}/debug/{iteration_count}'
    depth_save_dir = f'{save_dir}/debug/{iteration_count}'
    if save_imgs:
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)

    for idx in range(total_kf_num):
        kf_viewpoint = gs_viewpoints[idx]

        gtimage = kf_viewpoint.original_image.cuda()
        rendering = render(kf_viewpoint, kf_viewpoint,
                        gaussians, pipeline_params, background,
                        use_flow=False)
        tstamp = int(tstamps[idx])

        image = rendering["render"]
        image_clamp = torch.clamp(image, 0.0, 1.0)
        image_ab = torch.clamp(image, 0.0, 1.0)

        if save_imgs:
            depth = rendering["depth"].detach().squeeze().cpu().numpy()
            pred = (image_ab.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{image_save_dir}/pred_{tstamp:06d}.jpg', pred)

            depth_disp = np.nan_to_num(depth.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            depth_disp = np.clip(depth_disp, 0, np.percentile(depth_disp, 99))
            depth_norm = cv2.normalize(depth_disp, None, 0, 255, cv2.NORM_MINMAX)
            if depth_norm.ndim == 3:
                depth_norm = depth_norm.squeeze()
            depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(f"{depth_save_dir}/jet_{tstamp:06d}.png", depth_color)

            gt = (gtimage.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{image_save_dir}/gt_{tstamp:06d}.jpg', gt)

        mask = gtimage > 0
        psnr_score = psnr((image_ab[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image_ab).unsqueeze(0), (gtimage).unsqueeze(0)).mean()
        lpips_score = cal_lpips((image_clamp).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
                      
    output = dict()
    output["iterations"] = iteration_count
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_file = os.path.join(save_dir, "psnr", "debug_results.jsonl")
    os.makedirs(os.path.dirname(psnr_save_file), exist_ok=True)

    with open(psnr_save_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output) + "\n")



# def eval_rendering_kf(
#     viewpoints,
#     gaussians,
#     save_dir,
#     background,
#     iteration="final",
# ):
#     psnr_array, ssim_array, lpips_array = [], [], []
#     cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
#     for frame in viewpoints.values():
#         gtimage = frame.original_image.cuda()

#         rendering = render(frame, gaussians, background)
#         image = (torch.exp(frame.exposure_a)) * rendering["render"] + frame.exposure_b
#         image = torch.clamp(image, 0.0, 1.0)

#         mask = gtimage > 0
#         psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
#         ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
#         lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

#         psnr_array.append(psnr_score.item())
#         ssim_array.append(ssim_score.item())
#         lpips_array.append(lpips_score.item())

#     output = dict()
#     output["mean_psnr"] = float(np.mean(psnr_array))
#     output["mean_ssim"] = float(np.mean(ssim_array))
#     output["mean_lpips"] = float(np.mean(lpips_array))

#     Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

#     psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
#     os.makedirs(psnr_save_dir, exist_ok=True)

#     json.dump(
#         output,
#         open(os.path.join(psnr_save_dir, "final_result_kf.json"), "w", encoding="utf-8"),
#         indent=4,
#     )
#     return output