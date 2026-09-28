

import time
import numpy as np
import torch
import torch.multiprocessing as mp
from lietorch import SE3
from munch import munchify
import random
# from tqdm import tqdm
from tqdm import trange

from utils.config_utils import load_config
from utils.flow_utils import normalize_weights, downsample_flow, downsample_disp
from utils.slam_utils import update_acm, solve_pose_update1, solve_pose_update2
from utils.logging_utils import Log

import os
import cv2
import gfslam_backends

from gaussian_splatting.gui import gui_utils, slam_gui
from utils.camera_utils import Camera, CameraVis, to_camera_vis
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, update_pose_by_delta
from utils.rgbd_utils import pose_matrix_to_quaternion
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, ssim_masked
from gaussian_splatting.utils.eval_utils import eval_rendering, eval_rendering_kf

import torch.nn.functional as F

from functools import partial
if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast

import time

from gs_mapper_fast import GSMapperFast

FLOAT_SCALING = 1.0

class GSMapper(GSMapperFast, mp.Process):
    def __init__(self, video, graph, args):
        super().__init__()
        self.video = video
        self.graph = graph
        self.config = load_config(args.gsconfig_path)
        self.use_gui = self.config["Results"]["use_gui"]
        # CLI --disable_vis overrides YAML use_gui: True. Lets the user
        # turn off the Open3D GUI (and avoid spawning slam_gui process)
        # without editing the config file.
        if getattr(args, "disable_vis", False):
            self.use_gui = False
        self.save_video = self.config["Results"]["save_video"]
        # Optional: skip eval-image saves (pred/gt/depth jpg/png) to save disk.
        # Read by eval_utils via env var; metric computation unaffected.
        os.environ["GFSLAM_SAVE_KF_IMAGES"] = (
            "1" if self.config["Results"].get("save_kf_images", True) else "0"
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.H, self.W  = self.video.ht, self.video.wd
        self.mask_shape = (1, self.H, self.W)

        self.opt_params = munchify(self.config["opt_params"])
        self.model_params = munchify(self.config["model_params"])
        self.pipeline_params = munchify(self.config["pipeline_params"])
        self.flow_func = self.config["Training"]["flow_func"]

        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        # self.model_params.sh_degree = 1 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(sh_degree=self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(1.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        self.debug_dir = args.output

        self.iteration_count = 0
        self.save_count = 0
        self.reset_opacity_count = 0
        self.remove_inactive_count = 0
        self.gsmap_initialized = False
        self.current_window = []
        self.occ_aware_visibility = {}

        self.cameras_extent = 6.0
        self.set_hyperparams()

        self.keyframe_optimizers = None

        self.edp_total_time = 0.0

        # Optional speed flags from YAML Training: section.
        # Synchronous DBA-coupled GS pipeline is preserved unless fast_mode
        # is opted in.
        train_cfg = self.config.get("Training", {})
        self.visible_only_adam = bool(train_cfg.get("visible_only_adam", False))
        fast_cfg = train_cfg.get("fast_mode", {}) or {}
        self.fast_mode_cfg = fast_cfg
        self.fast_mode_enabled = bool(fast_cfg.get("enabled", False))
        # Worker thread for async mapping (fast_mode only).
        # Persistent worker pulling from a queue; tracking enqueues
        # per-keyframe mapping tasks and blocks via wait_async after each
        # enqueue. Wait at terminate() drains anything still in flight.
        import threading, queue
        self._async_lock = threading.Lock()
        self._async_exception = None
        self._mapping_queue = None
        self._mapping_worker = None
        # Frontend will install a callable used by the worker for refinement
        # passes (insert vs refine split).
        self._refine_callback = None
        # Refinement iter budget. Each insert task grants `refine_per_insert`
        # iters; refinement steps consume from this counter. When 0, the
        # worker sleeps instead of tight-looping.
        self._refine_budget = 0
        # Per-insert refinement budget. Chunk size 100, so 300 → 3 chunks/KF
        # by default. Split and prune live in DIFFERENT chunks so newly-split
        # tiny gaussians get a full chunk of supervision before facing prune.
        self._refine_per_insert = int(fast_cfg.get("refine_per_insert", 300))
        # Counter incremented on every refinement chunk. With chunk size
        # 100 and YAML defaults (gaussian_update_every=30,
        # gaussian_prune_every=70), a `do_split_only` chunk triggers one
        # split at train_num=31, and a `do_prune_only` chunk triggers one
        # prune at train_num=71.
        self._refine_pass_count = 0
        # Mapping-fps probe: tally iters/sec over a rolling window.
        self._map_iter_counter_t0 = time.time()
        self._map_iter_counter_n0 = 0
        self._map_iter_last_log = self._map_iter_counter_t0
        self.last_mapping_fps = 0.0
        if self.fast_mode_enabled:
            # FIFO — process inserts in tracking order so every KF gets
            # its insert+refinement turn. Retired KFs (rm_keyframe) are
            # skipped on uid_to_slot lookup at pop time.
            self._mapping_queue = queue.Queue()
            # Worker is started lazily after _refine_callback is installed
            # by GFSLAMFrontend so it can call back into the frontend safely.

        self._print_mode_banner(fast_cfg)

        self.gui_process = None
        if self.use_gui:
            self.q_main2vis = mp.Queue(maxsize=1)
            self.q_vis2main = mp.Queue(maxsize=1)
            self.params_gui = gui_utils.ParamsGUI(
                pipe=self.pipeline_params,
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process.start()
            time.sleep(10)

        if self.save_video:
            filename_backcam_track = os.path.join(self.debug_dir, "video_backcam_track.mp4")
            filename_backcam_full = os.path.join(self.debug_dir, "video_backcam_full.mp4")
            filename_backcam_skip = os.path.join(self.debug_dir, "video_backcam_skip.mp4")
            filename_curr = os.path.join(self.debug_dir, "video_curr.mp4")
            filename_fixview = os.path.join(self.debug_dir, "video_fixview.mp4")
            filename_skip = os.path.join(self.debug_dir, "video_skip.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.fps = 30
            self.video_interval_tracking_only = 5
            self.video_interval_tracking = 5
            self.video_interval_mapping = 10

            gap   = int(max(4, self.H // 128))
            W_out = 2 * self.W + 2 * gap
            H_out = self.H

            if W_out % 2: W_out += 1
            if H_out % 2: H_out += 1

            self._mosaic_gap = gap
            self._mosaic_size = (W_out, H_out)

            # self.video_writer_backcam_track = cv2.VideoWriter(
            #     filename_backcam_track, fourcc, self.fps, (W_out, H_out)
            # )
            self.video_writer_backcam_full = cv2.VideoWriter(
                filename_backcam_full, fourcc, self.fps, (W_out, H_out)
            )
            self.video_writer_backcam_skip = cv2.VideoWriter(
                filename_backcam_skip, fourcc, self.fps, (W_out, H_out)
            )

            self.video_writer_curr = cv2.VideoWriter(
                filename_curr, fourcc, self.fps, (W_out, H_out)
            )
            # self.video_writer_fixview = cv2.VideoWriter(
            #     filename_fixview, fourcc, self.fps, (W_out, H_out)
            # )
            self.video_writer_skip = cv2.VideoWriter(
                filename_skip, fourcc, self.fps, (W_out, H_out)
            )

    @staticmethod
    def _print_mode_banner(fast_cfg):
        lines = ["==== GFSLAM mode ===="]
        if not fast_cfg.get("enabled", False):
            lines.append("fast_mode      : DISABLED (legacy synchronous)")
        else:
            use_fb = bool(fast_cfg.get("use_full_batch", True))
            fz = fast_cfg.get("freezing", {}) or {}
            ie = fast_cfg.get("inactive_edges", {}) or {}
            lines.append("fast_mode      : ENABLED")
            lines.append(f"  path          : {'full_batch' if use_fb else 'single_batch'}")
            if fz.get("enabled", False):
                lines.append(f"  freezing      : ON  (max_age_iters={fz.get('max_age_iters', 500)})")
            else:
                lines.append("  freezing      : OFF")
            if ie.get("disable", False):
                lines.append("  inactive_edges: OFF (render disabled)")
            else:
                lines.append(f"  inactive_edges: every {ie.get('every_n_iters', 1)} iter")
        lines.append("=========================")
        print("\n".join(lines), flush=True)

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_prune_every = self.config["Training"]["gaussian_prune_every"]
        self.gaussian_prune_offset = self.config["Training"]["gaussian_prune_offset"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]

        self.gui_interval = self.config["Training"]["gui_interval"]

        self.save_results = self.config["Results"]["save_results"]
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_trj = self.config["Results"]["save_trj"]

    def synchronize_poses_to_gs(self, unique_ij):
        """ Synchronize poses: video -> gs """
        with torch.no_grad():
            for i in (unique_ij):
                viewpoint = self.video.gs_viewpoints[i]
                # fast_mode tombstone: skip retired keyframe slots.
                if viewpoint is None:
                    continue
                Tcw_tensor = SE3(self.video.poses[i]).matrix()
                viewpoint.update_RT(Tcw_tensor[:3, :3], Tcw_tensor[:3, 3])
                # viewpoint.update_GTRT(Twc_tensor[:3, :3], Twc_tensor[:3, 3])
                # viewpoint.cam_trans_delta.fill_(0)
                # viewpoint.cam_rot_delta.fill_(0)

    def synchronize_poses_to_video(self, unique_ij):
        """ Synchronize poses: gs -> video """
        with torch.no_grad():
            for i in unique_ij:
                viewpoint = self.video.gs_viewpoints[i]
                # fast_mode tombstone: skip retired keyframe slots.
                if viewpoint is None:
                    continue
                Tcw = np.eye(4)
                Tcw[:3, :3] = viewpoint.R.cpu().numpy()
                Tcw[:3, 3] = viewpoint.T.cpu().numpy()
                Tcw_vec = torch.from_numpy(pose_matrix_to_quaternion(Tcw))

                self.video.poses[i] = Tcw_vec

                # viewpoint.cam_trans_delta.fill_(0)
                # viewpoint.cam_rot_delta.fill_(0)

    def synchronize_gsdepth_to_video(self, idx):
        with torch.no_grad():
            viewpoint = self.video.gs_viewpoints[idx]
            render_pkg = render(viewpoint, viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=False, update_flow_conf=False,
                                train_pose=False)
            depth = (render_pkg["depth"]).squeeze()
            silh = render_pkg["silh"]
            # silh_mask = (silh > 0.99).squeeze()
            disp = 1.0 / (depth + 1e-6)  # Convert depth to disparity
            self.video.disps_up[idx] = disp
            down_silh = downsample_disp(silh.unsqueeze(0), scale_factor=1/8).squeeze()
            self.video.disps[idx][down_silh > 0.9] = downsample_disp(self.video.disps_up[idx].unsqueeze(0).unsqueeze(0), scale_factor=1/8).squeeze()[down_silh > 0.9]

    def setup_keyframe_optimizers(self, unique_ij, use_only_color=False, opt_only_pose=False, is_refinement=False, fix_pose=False):

        self.keyframe_optimizers = None
        """ Setup optimizers for keyframes """
        lr_scale = 1.0
        if use_only_color:
            lr_scale = 0.5
        opt_params = []


        for idx in unique_ij.cpu().tolist():
        # for idx in range(self.video.counter.value):
            viewpoint = self.video.gs_viewpoints[idx]
            # fast_mode tombstone: skip retired keyframe slots.
            if viewpoint is None:
                continue

            # ---- Layer 3: fast_mode hands pose ownership to tracking. When
            # `fix_pose=True`, mapping does not optimize cam_rot/cam_trans
            # at all — gaussian params + (optional) exposure only. This
            # eliminates the tracking↔mapping pose write race.
            if not fix_pose:
                opt_params.append(
                    {
                        "params": [viewpoint.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                        * lr_scale,
                        "name": "rot_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.cam_trans_delta],
                        "lr": self.config["Training"]["lr"][
                            "cam_trans_delta"
                        ]
                        * lr_scale,
                        "name": "trans_{}".format(viewpoint.uid),
                    }
                )
            if not opt_only_pose:
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        # "lr": 0.01 if (is_refinement) else 0.000, #  or opt_only_pose
                        "lr": 0.00,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.00,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            if len(opt_params) > 0:
                self.keyframe_optimizers = torch.optim.Adam(opt_params)
            # self.keyframe_optimizers = torch.optim.Adam(opt_params, betas=(0.8, 0.96),eps=1e-6)
    
    def ba(self, train_num, edges, flows, weights, pose_fix_idx=0, flow_ba=True, \
           train_map=True, train_pose=False, add_color_target_idx=None, \
           N_dont_touch=3, do_densification=True):

        # torch.cuda.synchronize()
        # self.gaussians.optimizer.zero_grad(set_to_none=True)
        # self.keyframe_optimizers.zero_grad(set_to_none=True)

        # if train_map:
        #     self.iteration_count += len(edges)
        loss = 0
        loss_color = 0
        loss_flow = 0
        loss_silh = 0

        if train_map:
            LAMBDA_FLOW = 1.0
        else:
            LAMBDA_FLOW = 1.0
        
        acm_list = [
            viewspace_point_tensor_acm,
            visibility_filter_acm,
            radii_acm,
            n_touched_acm,
            n_found_acm,
            error_per_gs_acm,
            error_per_gs_2_acm,
            error_per_gs_3_acm
        ] = (
            {}, {}, {}, {}, {}, {}, {}, {}
        )

        update_gaussian = (
            train_num % self.gaussian_update_every
            == self.gaussian_update_offset and train_map
            and do_densification and False # and train_num > 0
        )
        do_prune = update_gaussian and train_num % self.gaussian_prune_every == self.gaussian_update_offset

        update_flowconfs = False
        self.gaussians.found_count.fill_(0)
        total_visibility_filter = torch.zeros((self.gaussians.get_opacity.shape[0],), device=self.device, dtype=torch.bool)

        first_gsflow = None
        first_flow_img = None

        with torch.enable_grad():

            # loss_color_terms = []
            # loss_flow_terms = []
            # loss_aux_terms = []
            # loss_scale_terms = []
            # loss_opacity_terms = []
            # loss_silh_terms = []

            loss_color_sum = torch.tensor(0.0, device=self.device)
            loss_flow_sum  = torch.tensor(0.0, device=self.device)
            loss_aux_sum   = torch.tensor(0.0, device=self.device)
            loss_scale_sum = torch.tensor(0.0, device=self.device)
            loss_opacity_sum = torch.tensor(0.0, device=self.device)
            loss_silh_sum = torch.tensor(0.0, device=self.device)

            for e, (i,j) in enumerate(edges):
                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]

                flow_conf_sum_prev = None
                if flow_ba:
                    # contiguous() to break the view-tie to ``flows`` so the
                    # source tensor (incl. ``self.graph.flow_ups`` slice that
                    # produced it) doesn't stay alive via ``flow_image``.
                    i_viewpoint.flow_image = flows[e].to(self.device, non_blocking=True).permute(2, 0, 1).contiguous()
                    w_e = weights[e].to(self.device, non_blocking=True)
                    flow_conf_sum_prev = w_e.detach().sum()

                i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
                render_pkg = render(i_viewpoint, j_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=flow_ba, update_flow_conf=update_flowconfs,
                                    flow_conf=w_e if flow_ba and self.flow_func == 'log-logistic' else None,
                                    train_pose=train_pose,
                                    use_flowraw_grad=False if self.flow_func == 'log-logistic' else True)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    silh,
                    gsflow,
                    flow_cost,  # donguk
                    error_per_gs,
                    error_per_gs_2,
                    error_per_gs_3,
                    aux_image,
                    aux_image2,
                    aux_image3,
                    n_touched,
                    n_found,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["silh"],
                    render_pkg["gsflow"], # donguk
                    render_pkg["flowcost"], # donguk
                    render_pkg["error_per_gs"],
                    render_pkg["error_per_gs_2"],
                    render_pkg["error_per_gs_3"],
                    render_pkg["aux_image"],
                    render_pkg["aux_image2"],
                    render_pkg["aux_image3"],
                    render_pkg["n_touched"], 
                    render_pkg["n_found"],
                )

                if first_gsflow is None:
                    first_gsflow = gsflow.detach().clone()
                if first_flow_img is None:
                    first_flow_img = i_viewpoint.flow_image.detach().clone()

                # image_ab = torch.exp(i_viewpoint.exposure_a) * image + i_viewpoint.exposure_b
                image_ab = image
                rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                Ll1 = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask
                # Lssim = (1.0 - ssim_masked(image_ab, i_gt_image, rgb_pixel_mask))
                loss1 = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                               + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
                loss_color_sum = loss_color_sum + loss1
                # loss_color_terms.append(loss1)
                # loss_color += ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                #                + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING


                if train_map and update_gaussian:
                    # loss_aux = (Lssim.detach() * aux_image).sum();
                    # loss_aux2 = (silh.detach() * aux_image2).sum();
                    # loss_aux3 = (flow_cost.detach() * aux_image3).sum()

                    # loss_aux_terms.append(loss_aux)
                    # loss_aux_terms.append(loss_aux2)
                    # loss_aux_terms.append(loss_aux3)
                    loss_aux_sum = loss_aux_sum + (Lssim.detach() * aux_image).sum()
                    loss_aux_sum = loss_aux_sum + (silh.detach() * aux_image2).sum()
                    loss_aux_sum = loss_aux_sum + (flow_cost.detach() * aux_image3).sum()

                    # loss_silh += 0.1 * FLOAT_SCALING * (1 - silh).mean()

                    radii_small_mask = torch.logical_and(radii < 7, radii > 0)
                    radii_small_mask = torch.logical_and(radii_small_mask, n_found > 0)
                    opacity_small = self.gaussians.get_opacity[radii_small_mask]
                    loss_small = torch.clamp(
                        # torch.log(opacity_small + 0.99),
                        (opacity_small - 0.01) * (opacity_small - 0.01),
                        min=0.0
                    ).mean() * FLOAT_SCALING
                    # loss_opacity_terms.append(loss_small)

                self.gaussians.found_count += (n_found > 0).to(torch.int32).detach()

                # total_visibility_filter = torch.logical_or(total_visibility_filter, visibility_filter)
                total_visibility_filter |= visibility_filter
                
                ## Flow optimization
                if flow_ba:
                    # if flow_conf_sum_prev < 1.0:
                    # loss_flow_terms.append(loss_f)

                    if self.flow_func == 'log-logistic':
                        loss_f = (flow_cost * rgb_pixel_mask).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                    elif self.flow_func == 'L1':
                        flow_diff = ((gsflow - i_viewpoint.flow_image.detach()).abs()) * w_e.detach().permute(2,0,1)
                        loss_f = (flow_diff).mean()
                    elif self.flow_func == 'Mahalanobis':
                        # flow_diff = ((gsflow - i_viewpoint.flow_image.detach()) **2) * w_e.detach().permute(2,0,1)
                        flow_diff = ((gsflow - i_viewpoint.flow_image.detach()).square() * w_e.detach().permute(2,0,1))
                        loss_f = (flow_diff).mean()
                    else:
                        raise ValueError("Unknown flow function: {}".format(self.flow_func))

                    loss_flow_sum = loss_flow_sum + loss_f

                ## Save data for the process if gaussians will be updated
                if update_gaussian:
                    data = (
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        n_touched,
                        n_found,
                        error_per_gs,
                        error_per_gs_2,
                        error_per_gs_3,
                    )
                    update_acm(e, acm_list, data)

                ## update flowconfs 
                if update_flowconfs:
                    weights[e] = i_viewpoint.flow_conf

            
            if add_color_target_idx is not None:
                add_viewpoint = self.video.gs_viewpoints[add_color_target_idx]
                add_gt_image = add_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
                rgb_pixel_mask = (add_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                render_pkg = render(add_viewpoint, add_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=False, update_flow_conf=False,
                                    train_pose=train_pose,
                                    use_flowraw_grad=False if self.flow_func == 'log-logistic' else True)
                image = render_pkg["render"]
                # image_ab = torch.exp(add_viewpoint.exposure_a) * image + add_viewpoint.exposure_b
                image_ab = image
                Ll1 = l1_loss(image_ab, add_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image_ab, add_gt_image)) * rgb_pixel_mask
                # Lssim = (1.0 - ssim_masked(image_ab, add_gt_image, rgb_pixel_mask))
                loss3 = len(edges) * ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                               + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
                # loss_color_terms.append(loss3)
                loss_color_sum = loss_color_sum + loss3


            ## Scale loss
            if train_map:
                scaling = self.gaussians.get_scaling
                scales_visible = scaling[total_visibility_filter]
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean() * FLOAT_SCALING
                max_scales = torch.max(scales_visible, dim=1).values  
                min_scales = torch.min(scales_visible, dim=1).values
                ratio_scales = max_scales / (min_scales + 1e-6)
                loss_scale_ratio = torch.clamp(ratio_scales - 5, min=0).mean()
                loss4 = len(edges) * loss_scale_ratio * FLOAT_SCALING
                # loss_scale_terms.append(isotropic_loss)
                # loss_scale_terms.append(loss4)
                loss_scale_sum = loss_scale_sum + isotropic_loss + loss4

                ## Opacity loss
                opacity = self.gaussians.get_opacity
                opacity_visible = opacity[total_visibility_filter]
                loss_opacity = torch.clamp(
                    -(opacity_visible - 0.1) * torch.log(opacity_visible + 0.1),
                    min=0.0
                ).mean() * len(edges) * FLOAT_SCALING
                # loss_opacity_terms.append(loss_opacity)
                # loss_opacity_sum = loss_opacity_sum + loss_opacity

                # loss += loss_silh

            # loss_color = torch.stack(loss_color_terms).sum() if len(loss_color_terms) > 0 else torch.tensor(0.0, device=self.device)
            # loss_flow = torch.stack(loss_flow_terms).sum() if flow_ba else torch.tensor(0.0, device=self.device)
            # loss_aux_sum = torch.stack(loss_aux_terms).sum() if len(loss_aux_terms) > 0 else torch.tensor(0.0, device=self.device)
            # loss_scale_sum = torch.stack(loss_scale_terms).sum() if len(loss_scale_terms) > 0 else torch.tensor(0.0, device=self.device)
            # loss_opacity_sum = torch.stack(loss_opacity_terms).sum() if len(loss_opacity_terms) > 0 else torch.tensor(0.0, device=self.device)
            # loss_silh_sum = torch.stack(loss_silh_terms).sum() if len(loss_silh_terms) > 0 else torch.tensor(0.0, device=self.device)
            # loss = loss_color + LAMBDA_FLOW * loss_flow + loss_aux_sum \
            #         + loss_scale_sum + loss_opacity_sum + loss_silh_sum
                
            loss = loss_color_sum + LAMBDA_FLOW * loss_flow_sum + loss_aux_sum + loss_scale_sum + loss_opacity_sum + loss_silh_sum
            # loss = LAMBDA_FLOW * loss_flow_sum

            # Gradient descent
            loss.backward()
            # torch.cuda.synchronize()

        with torch.no_grad():
            ii_torch = edges[:, 0]
            jj_torch = edges[:, 1]
            unique_list = torch.unique(edges)

            total_found_filter = (self.gaussians.found_count > 0).detach()
            self.gaussians.n_found += total_found_filter.to(torch.int32).detach()

            if update_gaussian:
                # Set occ_aware_visibility
                # self.occ_aware_visibility = set_occ_aware_visibility(n_touched_acm) # or n_found_acm
                torch.cuda.synchronize()
                self.gaussians.densify_and_prune_by_error(radii_acm, error_per_gs_acm, error_per_gs_2_acm, error_per_gs_3_acm,
                                                          viewspace_point_tensor_acm, self.gaussians.found_count, total_found_filter,
                                                          do_prune=do_prune)

            max_tau = 0.0
            if train_map:
                self.gaussians.optimizer.step()
                # self.gaussians.update_learning_rate(self.iteration_count)

            if True: # always train exposure
                self.keyframe_optimizers.step()
                if train_pose:
                    for e, idx in enumerate(unique_list):
                        # if idx == pose_fix_idx or e < N_dont_touch:
                        #     continue
                        
                        viewpoint = self.video.gs_viewpoints[idx]
                        if idx == pose_fix_idx or e < N_dont_touch:
                            viewpoint.cam_rot_delta.data.fill_(0)
                            viewpoint.cam_trans_delta.data.fill_(0)
                            continue
                        else:
                            tau_norm = update_pose(viewpoint)

                        if tau_norm > max_tau:
                            max_tau = tau_norm

            # torch.cuda.synchronize()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.keyframe_optimizers.zero_grad(set_to_none=True)

            if self.use_gui and train_num % self.gui_interval == 0 and not train_map:
                # cur_idx = ii_torch.max().item()
                cur_idx = self.video.counter.value - 1
                # gui_keyframes = [self.video.gs_viewpoints[i] for i in unique_list]
                # Skip tombstone (None) slots — fast_mode keeps stale slot
                # numbers stable so retired KFs leave gaps.
                gui_keyframes = [
                    self.video.gs_viewpoints[i]
                    for i in range(cur_idx + 1)
                    if i < len(self.video.gs_viewpoints) and self.video.gs_viewpoints[i] is not None
                ]
                edge_dict = {}
                for idx in unique_list.cpu().tolist():
                    edge_dict[int(self.video.tstamp[idx].cpu())] = self.video.tstamp[jj_torch[ii_torch == idx]].long().cpu().tolist()

                first_gtflow_vis = gui_utils.optical_flow_to_rgb(first_flow_img)
                first_gsflow_vis = gui_utils.optical_flow_to_rgb(first_gsflow)

                # self.q_main2vis.put(
                gui_utils.put_latest(
                    self.q_main2vis,
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians) if train_map else None,
                        # current_frame=self.video.gs_viewpoints[cur_idx],
                        current_frame=to_camera_vis(self.video.gs_viewpoints[cur_idx]),
                        gtcolor=self.video.gs_viewpoints[cur_idx].original_image if train_num == 0 else None,
                        gtflow=first_gtflow_vis,
                        gsflow=first_gsflow_vis,
                        # keyframes=gui_keyframes,
                        # kf_window=edge_dict,
                    )
                )

            if self.save_video and not train_map and \
            (train_num % self.video_interval_tracking_only == 0 or train_num % self.video_interval_tracking == 0):
                img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                        behind=True, frustum_overlay=True, return_flow=True)
                
                ### Save to video_writer_backcam_track
                # if train_num % self.video_interval_tracking_only == 0:
                #     self.save_frame_to_video(self.video_writer_backcam_track, 
                #                             img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
                
                ### Save to video_writer_backcam_full / video_writer_curr
                if train_num % self.video_interval_tracking == 0:
                    self.save_frame_to_video(self.video_writer_backcam_full, 
                                            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
                    img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=False, frustum_overlay=False, return_flow=True)
                    self.save_frame_to_video(self.video_writer_curr,
                                            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)

            for d in acm_list:
                d.clear()
            del acm_list

            return max_tau
        
    def mapping_one_iter(self, train_num, edge, flow, weight, use_flow=True, last_stage_for_edge=False, \
                        train_pose=False, add_color_target_idx=False, do_densification=False,
                        set_occ_aware_visibility=False, is_init=False,
                        i_cam=None, j_cam=None, occ_key=None):
        # ``edge`` contains (i_slot, j_slot) for legacy slot-based callers; D
        # mode passes ``i_cam``/``j_cam`` directly so we don't dereference the
        # live ``self.video.gs_viewpoints`` (which can shift under
        # rm_keyframe). ``occ_key`` is used as the dict key for
        # occ_aware_visibility (uid in fast_mode, slot in legacy mode).
        if i_cam is None or j_cam is None:
            (i, j) = edge
        else:
            i, j = (None, None)
        self.iteration_count += 1
        # update_gaussian = (
        #     train_num % self.gaussian_update_every
        #     == self.gaussian_update_offset and do_densification
        # )
        update_gaussian = False
        update_data = None

        loss = 0
        loss_color_i=0; loss_color_j=0
        loss_flow = 0
        loss_aux=0; loss_aux2=0; loss_aux3=0
        LAMBDA_FLOW = self.config["Training"]["lambda_flow"]
        update_flowconfs = False

        with torch.enable_grad():
            if i_cam is not None and j_cam is not None:
                i_viewpoint = i_cam
                j_viewpoint = j_cam
            else:
                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]
            flow_conf_sum_prev = None
            if use_flow:
                # ``.contiguous()`` breaks the view tie to the source flow
                # tensor (typically a slice of payload.flow_ups in fast_mode).
                # Without it, every Camera that ever receives a flow_image
                # holds a strong ref to its source payload's flow_ups
                # block, so payloads' GPU memory NEVER frees → unbounded
                # GPU growth on long sequences.
                i_viewpoint.flow_image = flow.permute(2, 0, 1).contiguous().to(
                    self.device, non_blocking=True
                )
                flow_conf_sum_prev = weight.to(self.device, non_blocking=True).sum() #.cuda().detach().sum()
                # if flow_conf_sum_prev < 1000.0:
                #     use_flow = False

            i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
            render_pkg = render(i_viewpoint, j_viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=use_flow, update_flow_conf=update_flowconfs,
                                flow_conf=weight.to(self.device, non_blocking=True) if use_flow and self.flow_func == 'log-logistic' else None,
                                train_pose=train_pose,
                                use_flowraw_grad=False if self.flow_func == 'log-logistic' else True)
            
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                silh,
                gsflow,
                flow_cost,  
                error_per_gs,
                error_per_gs_2,
                error_per_gs_3,
                aux_image,
                aux_image2,
                aux_image3,
                n_touched,
                n_found,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"], 
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["silh"],
                render_pkg["gsflow"], 
                render_pkg["flowcost"], 
                render_pkg["error_per_gs"],
                render_pkg["error_per_gs_2"],
                render_pkg["error_per_gs_3"],
                render_pkg["aux_image"],
                render_pkg["aux_image2"],
                render_pkg["aux_image3"],
                render_pkg["n_touched"], 
                render_pkg["n_found"],
            )

            found_filter = (n_found > 0)
            if set_occ_aware_visibility:
                key = occ_key if occ_key is not None else i.item()
                self.occ_aware_visibility[key] = (n_touched) > 0

            ## I viewpoint color loss
            # image_ab = torch.exp(i_viewpoint.exposure_a) * image + i_viewpoint.exposure_b
            image_ab = image
            rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
            Ll1 = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
            if self.config["Training"]["ssim_mask_eroded"]:
                mask_f = rgb_pixel_mask.float().unsqueeze(0)
                eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
            else:
                Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask
            # Lssim = (1.0 - ssim_masked(image_ab, i_gt_image, rgb_pixel_mask))
            loss_color_i = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                            + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
            
            if update_gaussian:
                loss_aux = (Lssim.detach() * aux_image).sum();
                loss_aux2 = (silh.detach() * aux_image2).sum();
                loss_aux3 = (flow_cost.detach() * aux_image3).sum()

            ## Small radii loss
            # radii_small_mask = torch.logical_and(radii < 7, radii > 0)
            # # radii_small_mask = torch.logical_and(radii_small_mask, n_found > 0)
            # opacity_small = self.gaussians.get_opacity[radii_small_mask]
            # loss_small = torch.clamp(
            #     # torch.log(opacity_small + 0.99),
            #     (opacity_small - 0.01) * (opacity_small - 0.01),
            #     min=0.0
            # ).mean() * FLOAT_SCALING
            
            if last_stage_for_edge:
                self.gaussians.found_count += (n_found > 0).to(torch.int32).detach()

            ## Flow loss
            if use_flow:
                # flow_thresholding = (i_viewpoint.flow_image.sum(dim=0) > 0.01).view(*self.mask_shape)
                # gsflow_thresholding = (gsflow.sum(dim=0) > 0.01).view(*self.mask_shape)
                # mask = flow_thresholding & gsflow_thresholding
                # if flow_conf_sum_prev < 1.0:

                # loss_flow = (flow_cost * rgb_pixel_mask).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                if self.flow_func == 'log-logistic':
                    loss_flow = (flow_cost * rgb_pixel_mask).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                elif self.flow_func == 'L1':
                    flow_diff = ((gsflow - i_viewpoint.flow_image.detach()).abs())* weight.detach().permute(2,0,1)
                    loss_flow = (flow_diff).mean()
                elif self.flow_func == 'Mahalanobis':
                    # flow_diff = ((gsflow - i_viewpoint.flow_image.detach()) **2) * weight.detach().permute(2,0,1)
                    flow_diff = ((gsflow - i_viewpoint.flow_image.detach()).square() * weight.detach().permute(2,0,1))
                    loss_flow = (flow_diff).mean()
                else:
                    raise ValueError("Unknown flow function: {}".format(self.flow_func))

            if update_flowconfs:
                weight = i_viewpoint.flow_conf

            loss_silh = FLOAT_SCALING * (1 - silh).mean()

            ## J viewpoint color loss
            if add_color_target_idx:
                j_gt_image = j_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
                render_pkg = render(j_viewpoint, j_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=False, update_flow_conf=False,
                                    train_pose=train_pose)
                image = render_pkg["render"]
                # image_ab = torch.exp(j_viewpoint.exposure_a) * image + j_viewpoint.exposure_b
                image_ab = image
                rgb_pixel_mask = (j_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                Ll1 = l1_loss(image_ab, j_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image_ab, j_gt_image)) * rgb_pixel_mask
                # Lssim = (1.0 - ssim_masked(image_ab, j_gt_image, rgb_pixel_mask))
                loss_color_j = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                               + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
            
            ## Scale loss
            scaling = self.gaussians.get_scaling
            # scales_visible = scaling[found_filter]
            mask_f = found_filter.to(scaling.dtype)
            # isotropic_loss = torch.abs(scales_visible - scales_visible.mean(dim=1, keepdim=True)).mean() * FLOAT_SCALING
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean() * FLOAT_SCALING
            max_scales = torch.max(scaling, dim=1).values  
            min_scales = torch.min(scaling, dim=1).values
            ratio_scales = max_scales / (min_scales + 1e-6)
            loss_scale_ratio = torch.clamp(ratio_scales - 3, min=0).mean() * FLOAT_SCALING

            ## Opacity loss
            opacity = self.gaussians.get_opacity.squeeze()
            loss_opacity_elem = torch.clamp(
                -(opacity - 0.1) * torch.log(opacity + 0.01), min=0.0
            ).squeeze(-1)                                   # → [N]
            # loss_opacity_elem = -(opacity - 0.1)*(opacity - 0.1) * torch.log(opacity) # → [N]
            # loss_opacity_elem = torch.clamp(
            #     -(opacity - 0.1)*(opacity - 0.1) * torch.log(opacity+0.01), min=0.0 
            # ).squeeze(-1)                                   # → [N]
            # loss_opacity_elem = torch.clamp(
            #     (opacity - 0.1)* (1.0 - opacity)**4, min=0.0 
            # ).squeeze(-1)                                   # → [N]
            # base = 1.0 - opacity          # new tensor
            # p2 = base * base               # (1 - a)^2
            # val = (opacity - 0.1) * p4     # multiply
            # loss_opacity_elem = val.squeeze(-1)

            mask_f = found_filter.to(loss_opacity_elem.dtype)  # [N]
            den = mask_f.sum().clamp_min(1)
            if is_init:
                loss_opacity = 0
            else:
                loss_opacity = (loss_opacity_elem * mask_f).sum() / den * FLOAT_SCALING
                # loss_opacity = (loss_opacity_elem * mask_f).mean()
            # opacity = self.gaussians.get_opacity
            # opacity_visible = opacity[found_filter]
            # loss_opacity = torch.clamp(
            #     -(opacity_visible - 0.1) * torch.log(opacity_visible + 0.01),
            #     min=0.0
            # ).mean() * FLOAT_SCALING
            # loss_opacity = torch.clamp(
            #     -(opacity_visible - 0.8) * torch.log(opacity_visible + 0.05),
            #     min=0.0
            # ).mean() * 10.0 * FLOAT_SCALING # 20.0
            # loss_opacity2 = torch.clamp(-torch.log(opacity_visible + 1e-6), min=0.0).mean() * FLOAT_SCALING # 20.0

            # scaling_mean = scaling.mean(dim=1, keepdim=True).detach()
            # mask = (opacity >= self.config["Training"]["opacity_th"]).squeeze(1)
            # if mask.any():
            #     o_m = opacity[mask]
            #     s_m = scaling_mean[mask]
            #     oc = o_m - o_m.mean()
            #     sc = s_m - s_m.mean()
            #     cov = (oc * sc).mean()
            # else:
            #     loss_opacity3 = torch.tensor(0.0, device=opacity.device)
            ##########################################################
                
            # loss_opacity_to_zero = opacity.sum() / found_filter.sum().clamp_min(1) * FLOAT_SCALING * 0.001

            # loss = LAMBDA_FLOW * loss_flow
            loss = loss_color_i + loss_color_j + LAMBDA_FLOW * loss_flow \
                + isotropic_loss + loss_opacity * 0.01 # + loss_scale_ratio  isotropic_loss
            loss.backward()
            # torch.cuda.synchronize()

        with torch.no_grad():
            self.gaussians.n_found += (n_found > 0).to(torch.int32).detach()
            # `edge` is None in fast_mode (cam-direct call); only used by the
            # train_pose branch below which fast_mode never enters.
            unique_list = torch.unique(edge) if edge is not None else None

            if update_gaussian:
                torch.cuda.synchronize()
            
            # kfIDs = self.gaussians.unique_kfIDs[found_filter]
            # kfIDs_f = kfIDs.to(torch.float32)
            # tstamps = self.video.tstamp[:self.video.counter.value].to(torch.float32)
            # diffs = torch.abs(kfIDs_f[:, None] - tstamps[None, :])
            # closest_idxs = diffs.argmin(dim=1)                               
            # mean_closest_idx = int(closest_idxs.float().min().item())            # Python float
            # curr_idx = i.item()

            # if train_num == 0:

            # self.gaussians.cutoff_gradients(n_found > 0)
            if self.visible_only_adam:
                # H5: skip Adam state/param updates for Gaussians not visible
                # in this iter. Saves ~70-80% of optimizer cost when typical
                # ~20% of Gaussians touch a frame.
                self.gaussians.masked_optimizer_step(visibility_filter)
            else:
                self.gaussians.optimizer.step()

            # always train exposure (None when fast_mode + fix_pose + opt_only_pose)
            if self.keyframe_optimizers is not None:
                self.keyframe_optimizers.step()
            if train_pose:
                for e, idx in enumerate(unique_list):
                    viewpoint = self.video.gs_viewpoints[idx]
                    if idx == 0:
                        viewpoint.cam_rot_delta.data.fill_(0)
                        viewpoint.cam_trans_delta.data.fill_(0)
                        continue
                    tau_norm = update_pose(viewpoint)

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.keyframe_optimizers is not None:
                self.keyframe_optimizers.zero_grad(set_to_none=True)

            if self.save_video and train_num % self.video_interval_mapping == 0:
                ### Save to video_writer_backcam_full
                img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=True, frustum_overlay=True, return_flow=True)
                self.save_frame_to_video(self.video_writer_backcam_full, 
                                        img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
                
                ### Save to video_writer_curr
                img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                        behind=False, frustum_overlay=False, return_flow=True)
                self.save_frame_to_video(self.video_writer_curr,
                                        img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)

    def get_past_frames(self, selected_indices):
        total_found_filter = None
        for e, idx in enumerate(selected_indices):
            viewpoint = self.video.gs_viewpoints[idx]
            # In fast_mode (tombstone rm_keyframe) some slots can be None;
            # skip them — their KF was retired and contributes no view.
            if viewpoint is None:
                continue
            render_pkg = render(viewpoint, viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=False, update_flow_conf=False,
                                train_pose=False)
            (
                n_touched,
                n_found,
            ) = (
                render_pkg["n_touched"],
                render_pkg["n_found"],
            )
            found_filter = (n_found > 0).detach()
            if total_found_filter is None:
                # total_found_filter = found_filter
                total_found_filter = found_filter.detach().clone()
            else:
                # total_found_filter = torch.logical_or(total_found_filter, found_filter)
                total_found_filter.logical_or_(found_filter)

        # selected_IDs = self.video.tstamp[selected_indices].to(torch.int32)
        # touched_kfIDs = torch.unique(self.gaussians.unique_kfIDs[total_found_filter]).to(torch.int32)
        # is_in_selected = torch.isin(touched_kfIDs, selected_IDs)
        # past_kfIDs = touched_kfIDs[~is_in_selected]
                
        # selected_IDs = torch.index_select(self.video.tstamp, 0, selected_indices).to(torch.int32)
        # midx = torch.nonzero(total_found_filter, as_tuple=True)[0]
        # touched_kfIDs = torch.index_select(self.gaussians.unique_kfIDs, 0, midx).to(torch.int32)
        # del midx, total_found_filter
        # selected_IDs_sorted = torch.unique(selected_IDs, sorted=True)
        # pos = torch.searchsorted(selected_IDs_sorted, touched_kfIDs)
        # in_bounds = pos < selected_IDs_sorted.numel()
        # is_in_selected = in_bounds & (selected_IDs_sorted[pos.clamp_max(selected_IDs_sorted.numel()-1)] == touched_kfIDs)
        # past_kfIDs = touched_kfIDs[~is_in_selected]
                
        # If every selected index pointed to a tombstoned slot (None
        # viewpoint), the loop above never wrote total_found_filter and
        # there are no past KFs to report.
        if total_found_filter is None:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        selected_IDs = torch.index_select(self.video.tstamp, 0, selected_indices).to(torch.int32)
        midx = torch.nonzero(total_found_filter, as_tuple=True)[0]
        touched_kfIDs = torch.index_select(self.gaussians.unique_kfIDs, 0, midx).to(torch.int32)
        touched_kfIDs = torch.unique(touched_kfIDs, sorted=False)
        is_in_selected = torch.isin(touched_kfIDs, torch.unique(selected_IDs))
        past_kfIDs = touched_kfIDs[~is_in_selected]

        return past_kfIDs

    def get_gsflows(self, edges, downsampled_gsflow=False):
        loss = 0
        loss_color = 0
        loss_flow = 0

        # gsflows = torch.zeros((1, 0, self.graph.ht, self.graph.wd, 2), device=self.device, dtype=torch.float)

        pieces = []
        # with torch.enable_grad():
        with torch.no_grad():
            for e, (i,j) in enumerate(edges):
                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]
                # i_gt_image = i_viewpoint.original_image.cuda()
                render_pkg = render(i_viewpoint, j_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=True, update_flow_conf=False,
                                    train_pose=False, downsampled_gsflow=downsampled_gsflow)
                (
                    # image, 
                    # depth,
                    # silh,
                    gsflow,
                ) = (
                    # render_pkg["render"],
                    # render_pkg["depth"],
                    # render_pkg["silh"],
                    render_pkg["gsflow"], 
                )

                if not downsampled_gsflow:
                    gsflow_down = downsample_flow(gsflow.unsqueeze(0))   # (1, 2, h, w)
                    gsflow_dw = gsflow_down.permute(0, 2, 3, 1)          # (1, h, w, 2)                
                else:
                    gsflow_dw = gsflow.unsqueeze(0).permute(0, 2, 3, 1)          # (1, h, w, 2)
                pieces.append(gsflow_dw.unsqueeze(1))                # (1, 1, h, w, 2)

        if pieces:
            gsflows = torch.cat(pieces, dim=1)
        else:
            gsflows = torch.zeros((1, 0, self.graph.ht, self.graph.wd, 2), device=self.device, dtype=torch.float32)

        return gsflows
    
    def get_losses(self, edges, flows, weights):
        # total_losses = []
        # loss_colors = []
        # loss_flows = []
        LAMBDA_FLOW = self.config["Training"]["lambda_flow"]

        n_edges = edges.shape[0] if torch.is_tensor(edges) else len(edges)
        total_losses = torch.empty(n_edges, device=self.device, dtype=torch.float32)

        # with torch.enable_grad():
        with torch.no_grad():
            for e, (i,j) in enumerate(edges):
                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]

                i_viewpoint.flow_image = flows[e].to(self.device, non_blocking=True).permute(2, 0, 1).contiguous()
                weights_e = weights[e].to(self.device, non_blocking=True)
                flow_conf_sum_prev = weights_e.sum()

                i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True)
                render_pkg = render(i_viewpoint, j_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    flow_conf=weights_e, use_flow=True, update_flow_conf=False,
                                    train_pose=False)
                (
                    image, 
                    # depth,
                    # silh,
                    # gsflow,
                    flow_cost,  
                ) = (
                    render_pkg["render"],
                    # render_pkg["depth"],
                    # render_pkg["silh"],
                    # render_pkg["gsflow"], 
                    render_pkg["flowcost"], 
                )

                # image_ab = torch.exp(i_viewpoint.exposure_a) * image + i_viewpoint.exposure_b
                image_ab = image
                rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                Ll1 = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask
                # Lssim = (1.0 - ssim_masked(image_ab, i_gt_image, rgb_pixel_mask))
                loss_color_i = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                            + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
                # loss_colors.append(loss_color_i)

                loss_flow = (flow_cost).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6)) * LAMBDA_FLOW
                # loss_flows.append(loss_flow)

                loss_total = loss_color_i + loss_flow
                total_losses[e] = (loss_color_i + loss_flow).to(torch.float32)
                # total_losses.append(loss_total)

        # return torch.tensor(loss_colors, device=self.device), \
        #         torch.tensor(loss_flows, device=self.device), torch.tensor(total_losses, device=self.device)
        # return torch.tensor(total_losses, device=self.device)
        return total_losses

    ##### DEPRECATED #####
    def get_pose_jacobian(self, edges, flows, weights, is_weighted=False, use_flow=True, use_color=False):
        total_visibility_filter = torch.zeros((self.gaussians.get_opacity.shape[0],), device=self.device, dtype=torch.bool)
        jacobian_tau_acm, jacobian_tau_next_acm, error_per_gs_acm, error_per_gs_2_acm = [], [], [], []

        LAMBDA_FLOW = 1.0
        pixel_num = self.video.ht * self.video.wd
        
        with torch.enable_grad():
            for e, (i,j) in enumerate(edges):
                loss = 0
                loss_color = 0
                loss_flow = 0
                loss_color_e = 0
                loss_flow_e=0

                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]
                i_viewpoint.flow_image = flows[e].permute(2, 0, 1).cuda()  # (H, W, 2) -> (2, H, W)
                if is_weighted:
                    i_viewpoint.flow_conf = weights[e].cuda()
                else:
                    i_viewpoint.flow_conf = torch.ones_like(weights[e])

                flow_conf_sum_prev = None
                if use_flow:
                    flow_conf_sum_prev = i_viewpoint.flow_conf.detach().sum()

                i_gt_image = i_viewpoint.original_image.cuda()
                render_pkg = render(i_viewpoint, j_viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=use_flow, update_flow_conf=False,
                                    train_pose=True)
                
                (
                    image,
                    visibility_filter,
                    gsflow,
                    flow_cost,
                    silh,
                    error_per_gs,
                    error_per_gs_2,
                    aux_image,
                    aux_image2,
                    raster_settings,
                ) = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["gsflow"], 
                    render_pkg["flowcost"], 
                    render_pkg["silh"],
                    render_pkg["error_per_gs"],
                    render_pkg["error_per_gs_2"],
                    render_pkg["aux_image"],
                    render_pkg["aux_image2"],
                    render_pkg["raster_settings"],
                )

                if use_color:
                    # image_ab = torch.exp(i_viewpoint.exposure_a) * image + i_viewpoint.exposure_b
                    image_ab = image
                    Ll1 = l1_loss(image_ab, i_gt_image)
                    Lssim = 1.0 - ssim(image_ab, i_gt_image)
                    loss_color_e = ((1.0 - self.opt_params.lambda_dssim) * Ll1
                                + self.opt_params.lambda_dssim * Lssim) * FLOAT_SCALING / pixel_num
                    loss_color += loss_color_e.mean()

                if use_flow:
                    loss_flow_e = (flow_cost) / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                    loss_flow += loss_flow_e.sum()

                total_visibility_filter = torch.logical_or(total_visibility_filter, visibility_filter)

                loss_for_error = loss_color_e + LAMBDA_FLOW * loss_flow_e
                loss_aux = (loss_for_error.detach() * aux_image).sum()
                loss_aux2 = (silh.detach() * weights[e].detach() * aux_image2).sum()
                loss += loss_aux
                loss += loss_aux2

                # if use_color and use_flow:

                loss += (loss_color + LAMBDA_FLOW * loss_flow)
                loss.backward() # retain_graph=True
                # torch.cuda.synchronize()

                jacobian_tau_acm.append(raster_settings.grad_tau_raw.detach().clone())
                jacobian_tau_next_acm.append(raster_settings.grad_tau_next_raw.detach().clone())
                error_per_gs_acm.append(error_per_gs.grad.detach().clone())

                w_per_gs = error_per_gs_2.grad.detach().clone()
                min_val = w_per_gs.min().item()
                max_val = w_per_gs.max().item()
                normalized_w_per_gs = (w_per_gs - min_val) / (max_val - min_val + 1e-6)
                error_per_gs_2_acm.append(normalized_w_per_gs)
                # error_per_gs_2_acm.append(error_per_gs_2.grad.detach().clone())
                
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.keyframe_optimizers.zero_grad(set_to_none=True)

            {
            ## Add Scale loss?
            # scaling = self.gaussians.get_scaling
            # scales_visible = scaling[total_visibility_filter]
            # max_scales = torch.max(scales_visible, dim=1).values  
            # min_scales = torch.min(scales_visible, dim=1).values
            # ratio_scales = max_scales / (min_scales + 1e-6)
            # loss_scale_ratio = torch.clamp(ratio_scales - 5, min=0).mean()
            # loss += len(edges) * loss_scale_ratio * FLOAT_SCALING

            ## Add Opacity loss?
            # opacity = self.gaussians.get_opacity
            # opacity_visible = opacity[total_visibility_filter]
            # loss_opacity = torch.clamp(
            #     -(opacity_visible - 0.1) * torch.log(opacity_visible + 0.3),
            #     min=0.0
            # )
            # loss += 10 * len(edges) * loss_opacity.mean() * FLOAT_SCALING
            }

        with torch.no_grad():
            jacobian_iis = torch.stack(jacobian_tau_acm, dim=0)
            jacobian_jjs = torch.stack(jacobian_tau_next_acm, dim=0)
            errors_all = torch.stack(error_per_gs_acm, dim=0)
            weights_all = torch.stack(error_per_gs_2_acm, dim=0)

            return jacobian_iis, jacobian_jjs, errors_all, weights_all

    def optimize_init_pose(self, last_t, cur_iter, limit_edges=False):
        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        # coords1=  self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
        # self.graph.update_by_gs(coords1, None, None, use_inactive=True)

        last_t_mask = (jj_torch == last_t)
        edge_use_num = 2
        if limit_edges and last_t_mask.sum() >= edge_use_num:
            # true_indices = torch.nonzero(last_t_mask, as_tuple=False).squeeze()
            # ii_selected = ii_torch[true_indices]
            # sorted_values, sorted_indices = torch.sort(ii_selected, descending=True)
            # topk_indices_in_true = sorted_indices[:edge_use_num]
            # keep_indices = true_indices[topk_indices_in_true]
            # new_mask = torch.zeros_like(last_t_mask, dtype=torch.bool)
            # new_mask[keep_indices] = True
            # last_t_mask = new_mask
            true_indices = torch.nonzero(last_t_mask, as_tuple=False).squeeze()
            ii_selected = torch.index_select(ii_torch, 0, true_indices)
            topk_values, topk_indices_in_true = torch.topk(ii_selected, k=edge_use_num, largest=True, sorted=False)
            keep_indices = torch.index_select(true_indices, 0, topk_indices_in_true)  # [edge_use_num]
            new_mask = torch.zeros_like(last_t_mask, dtype=torch.bool)
            new_mask[keep_indices] = True
            last_t_mask = new_mask

        # ii_masked = ii_torch[last_t_mask]
        # jj_masked = jj_torch[last_t_mask]
        masked_idx = torch.nonzero(last_t_mask, as_tuple=True)[0]
        ii_masked = torch.index_select(ii_torch, 0, masked_idx)
        jj_masked = torch.index_select(jj_torch, 0, masked_idx)
        edges_for_opt = torch.stack([ii_masked, jj_masked], dim=1)

        update_mask = torch.logical_or(ii_torch == last_t, last_t_mask)
        coords1=  self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
        self.graph.update_by_gs(coords1, None, None, use_inactive=False, selected_mask=update_mask)

        # ii_masked = ii_torch[last_t_mask]
        # jj_masked = jj_torch[last_t_mask]
        # edges_for_opt = torch.stack([ii_masked, jj_masked], dim=1)
        
        # coords1=  self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
        # self.graph.update_by_gs(coords1, None, None, use_inactive=True, selected_mask=last_t_mask)

        flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
        weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)
        with torch.no_grad():
            updating_indices = torch.unique(edges_for_opt)
        self.synchronize_poses_to_gs(updating_indices)

        masked_idx = torch.nonzero(last_t_mask, as_tuple=True)[0]
        flows_for_opt = torch.index_select(flow_ups, 0, masked_idx)
        selected_weight_ups = torch.index_select(weight_ups, 0, masked_idx)
        if self.flow_func == 'log-logistic':
            flowconfs_for_opt = normalize_weights(selected_weight_ups)
        else:
            flowconfs_for_opt = selected_weight_ups
        first_idx = updating_indices.min().item()
        target_idx = last_t

        # # self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)
        # flow_confs = normalize_weights(weight_ups).unsqueeze(0)
        # first_idx = updating_indices.min().item()
        # flows_for_opt = flow_ups[last_t_mask]
        # # flowconfs_for_opt = self.graph.flow_confs[:, last_t_mask, ...].squeeze(0)
        # flowconfs_for_opt = flow_confs[:, last_t_mask, ...].squeeze(0)
        # target_idx = last_t
        # # flowconfs_for_opt = torch.ones_like(flowconfs_for_opt, device=self.device)

        self.setup_keyframe_optimizers(updating_indices, opt_only_pose=True)
        max_tau = 0.0
        total_iter = 30 if cur_iter < 3 else self.tracking_itr_num
        for train_num in range(total_iter):
            max_tau = self.ba(train_num, edges_for_opt, flows_for_opt, flowconfs_for_opt, \
                        add_color_target_idx=target_idx,
                        pose_fix_idx=first_idx, train_map=False, \
                        train_pose=True, flow_ba=True, N_dont_touch=(len(updating_indices)-1))
        self.gaussians.n_found.fill_(0)
        self.synchronize_poses_to_video(updating_indices)

        ### Save to video_writer_backcam_skip / video_writer_skip
        if self.save_video:
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=True, frustum_overlay=True, return_flow=True)
            self.save_frame_to_video(self.video_writer_backcam_skip,
                                     img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=False, frustum_overlay=False, return_flow=True)
            self.save_frame_to_video(self.video_writer_skip,
                                    img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)

        return max_tau
    
    def check_overlap_visibility(self, check_idx):
        if len(self.occ_aware_visibility) == 0 or check_idx not in self.occ_aware_visibility:
            return None
        check_visibility = self.occ_aware_visibility[check_idx]
        other_visibility_union = None
        for idx, occ_visibility in self.occ_aware_visibility.items():
            if idx == check_idx:
                continue
            if idx < (check_idx-2) or idx > (check_idx+2):
                continue
            other_visibility_union = occ_visibility if other_visibility_union is None else torch.logical_or(other_visibility_union, occ_visibility)
        if other_visibility_union is None:
            return None
        
        check_other_visibility_intersect = torch.logical_and(check_visibility, other_visibility_union)

        return check_other_visibility_intersect.count_nonzero() / check_visibility.count_nonzero()            
    
    def optimize_map_randomly(self, insert_GS=False, fix_all_pose=False, selected_idx=None, do_densification=False, 
                              do_flowupdate=True, do_mapping=True,
                              total_iters=200, use_inactive_mapping=True, set_occ_aware_visibility=False):
        if set_occ_aware_visibility:
            self.occ_aware_visibility.clear()

        curr_idx = self.video.counter.value - 1
        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        # coords1 = self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
        # self.graph.update_by_gs(coords1, None, None, use_inactive=True)
        
        # flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
        # weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)
        # self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)

        edges_for_opt = None
        flows_for_opt = None
        flowconfs_for_opt = None
        selected_mask = None

        edges_all = torch.stack([ii_torch, jj_torch], dim=1)
        if selected_idx is not None and selected_idx >= 0 and selected_idx < self.video.counter.value:
            selected_mask1 = (ii_torch == selected_idx)
            selected_mask2 = (jj_torch == selected_idx)
            selected_mask = torch.logical_or(selected_mask1, selected_mask2)
            if selected_mask.any():
                idx = torch.nonzero(selected_mask, as_tuple=True)[0]
                edges_for_opt = torch.index_select(edges_all, 0, idx)
                del idx
                # ii_masked = ii_torch[selected_mask]
                # jj_masked = jj_torch[selected_mask]
                # edges_for_opt = torch.stack([ii_masked, jj_masked], dim=1)
                # flows_for_opt = flow_ups[selected_mask]
                # flowconfs_for_opt = self.graph.flow_confs[:, selected_mask, ...].squeeze(0)
            else:
                print("Wrong selected idx!!!")
        
        if edges_for_opt is None:
            edges_for_opt = edges_all
            # edges_for_opt = torch.stack([ii_torch, jj_torch], dim=1)
            # flows_for_opt = flow_ups
            # flowconfs_for_opt = self.graph.flow_confs.squeeze(0)

        if do_flowupdate:
            coords1 = self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
            self.graph.update_by_gs(coords1, None, None, use_inactive=False, 
                                    selected_mask=selected_mask if selected_mask is not None else None)

        # self.graph.update_by_gs(coords1, None, None, use_inactive=True, 
        #                         selected_mask=(ii_torch == curr_idx))
        
        flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
        weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)
        # self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)

        # flow_confs = normalize_weights(weight_ups).unsqueeze(0)
        # if selected_mask is not None:
        #     flows_for_opt = flow_ups[selected_mask]
        #     # flowconfs_for_opt = self.graph.flow_confs[:, selected_mask, ...].squeeze(0)
        #     flowconfs_for_opt = flow_confs[:, selected_mask, ...].squeeze(0)
        # else:
        #     flows_for_opt = flow_ups
        #     # flowconfs_for_opt = self.graph.flow_confs.squeeze(0)
        #     flowconfs_for_opt = flow_confs.squeeze(0)

        if selected_mask is not None and selected_mask.any():
            idx = torch.nonzero(selected_mask, as_tuple=True)[0]
            flows_for_opt        = torch.index_select(flow_ups,   0, idx)      # (E_sel, H, W, 2)
            selected_weight_ups  = torch.index_select(weight_ups, 0, idx) 
            if self.flow_func == 'log-logistic':
                flowconfs_for_opt = normalize_weights(selected_weight_ups)
            else:
                flowconfs_for_opt = selected_weight_ups
            del idx, selected_weight_ups
        else:
            flows_for_opt = flow_ups
            if self.flow_func == 'log-logistic':
                flowconfs_for_opt = normalize_weights(weight_ups)
            else:
                flowconfs_for_opt = weight_ups

        num_inac = self.graph.ii_inac.shape[0]

        if use_inactive_mapping and num_inac > 0:
            pass

        # if use_inactive_mapping and num_inac > 0:
        #     ii_inac = self.graph.ii_inac
        #     jj_inac = self.graph.jj_inac

        #     window_min_index = ii_torch.min().item()
        #     inac_mask1 = ii_inac >= window_min_index - 10
        #     inac_mask2 = jj_inac >= window_min_index - 10
        #     inac_mask3 = ii_inac < window_min_index
        #     # inac_mask = torch.logical_and(inac_mask1, inac_mask2)
        #     inac_mask = inac_mask1 & inac_mask2 & inac_mask3
        #     if inac_mask.sum() > 0:
        #         valid_indices = torch.nonzero(inac_mask, as_tuple=True)[0]
        #         valid_len = valid_indices.shape[0]

        #         rand_indices = torch.randperm(valid_len, device=ii_inac.device)[:min(4, valid_len)]
        #         selected_indices = valid_indices[rand_indices]

        #         ii_rand = ii_inac[selected_indices]
        #         jj_rand = jj_inac[selected_indices]
        #         edges_inac = torch.stack([ii_rand, jj_rand], dim=1)
        #         flows_inac = self.graph.flow_ups_inac.squeeze(0)[selected_indices]
        #         flowconfs_inac = self.graph.flow_confs_inac[:, selected_indices, ...].squeeze(0)

        #         edges_for_opt = torch.cat([edges_inac, edges_for_opt], dim=0)
        #         flows_for_opt = torch.cat([flows_inac, flows_for_opt], dim=0)
        #         flowconfs_for_opt = torch.cat([flowconfs_inac, flowconfs_for_opt], dim=0)

        updating_indices_window = torch.unique(edges_for_opt)
        past_kfIDs = self.get_past_frames(updating_indices_window).to(torch.int32)
        # past_kfIDs = self.get_past_frames(torch.tensor([self.video.counter.value - 1]))
        edges_inac = None
        if use_inactive_mapping and num_inac > 0:
            ii_inac = self.graph.ii_inac
            jj_inac = self.graph.jj_inac

            # updating_indices_window = torch.unique(edges_for_opt)
            # past_kfIDs = self.get_past_frames(updating_indices_window).to(torch.int32)
            # past_kfIDs = self.get_past_frames(torch.tensor([self.video.counter.value - 1]))
            
            # indices_inactive = torch.unique(torch.stack([self.graph.ii_inac, self.graph.jj_inac], dim=1))
            # inactive_kfIDs = self.video.tstamp[indices_inactive].to(torch.int32)
            # common_kfIDs = inactive_kfIDs[torch.isin(inactive_kfIDs, past_kfIDs)].to(torch.int32)

            indices_inactive = torch.unique(torch.cat([ii_inac, jj_inac], dim=0))
            inactive_kfIDs = torch.index_select(self.video.tstamp, 0, indices_inactive).to(torch.int32)

            # common_kfIDs = inactive_kfIDs ∩ past_kfIDs
            A = torch.unique(past_kfIDs.to(torch.int32), sorted=True)         
            q = inactive_kfIDs                                                
            pos = torch.searchsorted(A, q)
            in_bounds = pos < A.numel()
            common_mask = torch.zeros_like(in_bounds, dtype=torch.bool)
            if in_bounds.any():
                pos_in = pos[in_bounds]
                q_in   = q[in_bounds]
                common_mask[in_bounds] = torch.index_select(A, 0, pos_in).eq(q_in)
            common_kfIDs = q[common_mask].to(torch.int32)

            # mask = torch.isin(self.video.tstamp[ii_inac].to(torch.int32), common_kfIDs)
            # valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)

            ii_ids = torch.index_select(self.video.tstamp, 0, ii_inac).to(torch.int32)
            B = torch.unique(common_kfIDs, sorted=True)
            pos2 = torch.searchsorted(B, ii_ids)
            in_bounds2 = pos2 < B.numel()
            mask = torch.zeros_like(in_bounds2, dtype=torch.bool)
            if in_bounds2.any():
                pos2_in = pos2[in_bounds2]
                ids_in  = ii_ids[in_bounds2]
                mask[in_bounds2] = torch.index_select(B, 0, pos2_in).eq(ids_in)
            valid_indices = torch.nonzero(mask, as_tuple=True)[0]

            if valid_indices.numel() > 0:
                # ii_vals = ii_inac[valid_indices]
                # unique_ii_inac = torch.unique(ii_vals)

                ii_vals = torch.index_select(ii_inac, 0, valid_indices)
                unique_ii_inac = torch.unique(ii_vals)


                num_samples = min(20, unique_ii_inac.numel())
                sampled_ii = unique_ii_inac[torch.randperm(unique_ii_inac.numel(), device=unique_ii_inac.device)[:num_samples]]
                ##################### NEW train_num_per_edge #####################
                # sampled_ii = unique_ii_inac
                ##################################################################

                # sorted_ii, _ = torch.sort(unique_ii_inac)
                # num_small = min(5, sorted_ii.numel())
                # smallest_ii = sorted_ii[:num_small]
                # rest_ii = sorted_ii[num_small:]
                # num_random = min(5, rest_ii.numel())
                # if num_random > 0:
                #     rand_idx = torch.randperm(rest_ii.numel(), device=rest_ii.device)[:num_random]
                #     random_ii = rest_ii[rand_idx]
                # else:
                #     random_ii = rest_ii.new_empty((0,))
                # sampled_ii = torch.cat([smallest_ii, random_ii], dim=0)
                ###################################

                # sampled_ii = unique_ii_inac
                ####################################

                # selected_edge_indices = []
                # for ii_val in sampled_ii:
                #     candidates = valid_indices[ii_vals == ii_val]
                #     choice = candidates[torch.randint(candidates.numel(), (1,), device=candidates.device)]
                #     selected_edge_indices.append(choice)
                # selected_edge_indices = torch.cat(selected_edge_indices)

                selected_edge_indices = []
                for ii_val in sampled_ii:
                    cand_mask = (ii_vals == ii_val)
                    candidates = valid_indices[cand_mask]
                    choice = candidates[torch.randint(candidates.numel(), (1,), device=candidates.device)]
                    selected_edge_indices.append(choice)
                selected_edge_indices = torch.cat(selected_edge_indices)      # (S,)

                # selected_ii = ii_inac[selected_edge_indices]
                # selected_jj = jj_inac[selected_edge_indices]
                selected_ii = torch.index_select(ii_inac, 0, selected_edge_indices)
                selected_jj = torch.index_select(jj_inac, 0, selected_edge_indices)

                edges_inac = torch.stack([selected_ii, selected_jj], dim=1)
                # flows_inac = self.graph.flow_ups_inac.squeeze(0)[selected_edge_indices]
                selected_edge_indices_cpu = selected_edge_indices.to('cpu')
                flows_inac = torch.index_select(self.graph.flow_ups_inac.squeeze(0), 0, selected_edge_indices_cpu).to(self.device, non_blocking=True)
                # flowconfs_inac = self.graph.flow_confs_inac[:, selected_edge_indices, ...].squeeze(0)
                # flowconfs_inac = normalize_weights(self.graph.weight_ups_inac.squeeze(0))[selected_edge_indices]
                # selected_weight_ups = self.graph.weight_ups_inac.squeeze(0)[selected_edge_indices]
                selected_weight_ups = torch.index_select(self.graph.weight_ups_inac.squeeze(0), 0, selected_edge_indices_cpu).to(self.device, non_blocking=True)
                if self.flow_func == 'log-logistic':
                    flowconfs_inac = normalize_weights(selected_weight_ups)
                else:
                    flowconfs_inac = selected_weight_ups

                edges_for_opt = torch.cat([edges_inac, edges_for_opt], dim=0)
                flows_for_opt = torch.cat([flows_inac, flows_for_opt], dim=0)
                flowconfs_for_opt = torch.cat([flowconfs_inac, flowconfs_for_opt], dim=0)

        # updating_indices = torch.unique(edges_for_opt)
        updating_indices = torch.unique(edges_for_opt.reshape(-1))
        self.synchronize_poses_to_gs(updating_indices)

        if insert_GS and selected_idx is not None:
            insert_region, found_filter, silh_filter, depth_rendered = self.identify_insert_region(selected_idx)
            if insert_region is not None:
                self.create_new_gaussians(selected_idx, insert_region, found_filter, silh_filter, depth=depth_rendered, is_init=False)

        self.setup_keyframe_optimizers(updating_indices)
        
        num_for_edge = 10
        num_edges = len(edges_for_opt)
        total_iters = max(total_iters, num_edges * num_for_edge)
        ##################### NEW train_num_per_edge #####################
        # total_iters = max(total_iters, num_edges * 5)
        ##################################################################
        
        if self.config["Training"]["adaptive_training_edges"]:
            total_losses = self.get_losses(edges_for_opt, flows_for_opt, flowconfs_for_opt)
            sorted_indices = torch.argsort(total_losses)
            # edges_for_opt = edges_for_opt[sorted_indices]
            # flows_for_opt = flows_for_opt[sorted_indices]
            # flowconfs_for_opt = flowconfs_for_opt[sorted_indices]
            edges_for_opt      = torch.index_select(edges_for_opt,      0, sorted_indices)
            flows_for_opt      = torch.index_select(flows_for_opt,      0, sorted_indices)
            flowconfs_for_opt  = torch.index_select(flowconfs_for_opt,  0, sorted_indices)

            log_range = np.logspace(0.1, 1, num_edges, base=2.0)
            weights = log_range / log_range.sum()   
            float_alloc = weights * total_iters
            int_alloc = np.floor(float_alloc).astype(int)
            remaining = total_iters - int_alloc.sum()
            residuals = float_alloc - int_alloc
            top_indices = np.argsort(-residuals)[:remaining]        
            int_alloc[top_indices] += 1
            train_num_per_edge = int_alloc
            edge_idx_stack = [i for i in range(len(edges_for_opt))]
        else:
            # equal_alloc = total_iters // num_edges
            # remaining = total_iters % num_edges
            # int_alloc = np.full(num_edges, equal_alloc, dtype=int)
            # int_alloc[:remaining] += 1
            # train_num_per_edge = int_alloc
            # edge_idx_stack = [i for i in range(len(edges_for_opt))]

            num_for_edge = 5
            # num_for_inac = 10

            # N = edges_for_opt.size(0)
            # recent_thr = self.video.counter.value - 3
            # train_num_per_edge = np.where(i_vals >= recent_thr, num_for_edge, num_for_inac).astype(np.int32)
            # edge_idx_stack = list(range(N))

            total = edges_for_opt.size(0)
            n_inac = 0
            if edges_inac is not None and edges_inac.numel() > 0:
                n_inac = edges_inac.size(0)
            train_num_per_edge = np.full(total, num_for_edge, dtype=np.int32)
            if n_inac > 0:
                num_for_inac = int(num_for_edge * (total - n_inac) / n_inac )
                train_num_per_edge[:n_inac] = min(num_for_inac, 10)
            edge_idx_stack = list(range(total))

        ##################### NEW train_num_per_edge #####################
        # equal_alloc = total_iters // num_edges
        # remaining = total_iters % num_edges
        # int_alloc = np.full(num_edges, equal_alloc, dtype=int)
        # int_alloc[:remaining] += 1
        # train_num_per_edge = int_alloc
        # edge_idx_stack = [i for i in range(len(edges_for_opt))]

        # recent_frame_thresh = (self.video.counter.value - 1) - 5
        # ii_all = edges_for_opt[:, 0]
        # recent_mask = ii_all > recent_frame_thresh
        # recent_indices = recent_mask.nonzero(as_tuple=False).squeeze()
        # train_num_per_edge[recent_indices.cpu().numpy()] += 10
        ##################################################################

        done_split = False
        done_prune = False
        train_num = 0

        start_time = time.time()
        while True and do_mapping:
            edge_idx = random.choice(edge_idx_stack)
            if train_num_per_edge[edge_idx] == 0:
                continue
            update_gaussian = (
                train_num > self.gaussian_update_every and
                train_num % self.gaussian_update_every
                > self.gaussian_update_offset and do_densification \
                and not done_split
            )
            prune_gaussian = (train_num % self.gaussian_prune_every) > self.gaussian_prune_offset \
                              and (train_num > self.gaussian_prune_every) and do_densification and not done_prune
                        
            last_stage_for_edge = (train_num_per_edge[edge_idx] == 1)
            train_num_per_edge[edge_idx] -= 1
            curr_idx_edge = (edges_for_opt[edge_idx] == self.video.counter.value - 1).any()
            self.mapping_one_iter(train_num, edges_for_opt[edge_idx], flows_for_opt[edge_idx],
                                  flowconfs_for_opt[edge_idx], last_stage_for_edge=last_stage_for_edge,
                                  train_pose=False, add_color_target_idx=False, do_densification=do_densification,
                                  set_occ_aware_visibility=(set_occ_aware_visibility and train_num_per_edge[edge_idx] == 0))
            
            if (update_gaussian or prune_gaussian): #and np.max(train_num_per_edge) < (num_for_edge): #train_num_per_edge[edge_idx] > 3: # and np.min(train_num_per_edge) > 3
                start_time = time.time()
                if update_gaussian: # and curr_idx_edge
                    done_split = True
                    self.efficient_densify_and_prune(do_prune=False)
                elif prune_gaussian: # and curr_idx_edge
                    done_prune = True
                    self.efficient_densify_and_prune(do_prune=prune_gaussian)
                self.edp_total_time += time.time() - start_time

            train_num += 1

            is_finished = False
            if all(x == 0 for x in train_num_per_edge):
                is_finished = True

            if self.use_gui and (is_finished or self.iteration_count % self.gui_interval == 0):
                cur_idx = self.video.counter.value - 1
                # gui_keyframes = [to_camera_vis(self.video.gs_viewpoints[i]) for i in range(cur_idx-10, cur_idx + 1)]
                # Skip tombstone (None) slots — fast_mode leaves gaps after rm_keyframe.
                gui_keyframes = [
                    to_camera_vis(self.video.gs_viewpoints[i])
                    for i in range(self.video.counter.value)
                    if i < len(self.video.gs_viewpoints) and self.video.gs_viewpoints[i] is not None
                ]
                edge_dict = {}
                # for idx in updating_indices.cpu().tolist():
                #     edge_dict[int(self.video.tstamp[idx].cpu())] = self.video.tstamp[jj_torch[ii_torch == idx]].long().cpu().tolist()
                for ii_idx, jj_idx in edges_for_opt.cpu().tolist():
                    t_i = int(self.video.tstamp[ii_idx].item())
                    t_j = int(self.video.tstamp[jj_idx].item())
                    edge_dict.setdefault(t_i, []).append(t_j)
                # for t_i, neighbors in edge_dict.items():
                # Push to GUI. ``current_frame`` is the latest *non-None*
                # camera at or below counter.value-1. The list is
                # pre-allocated with None slots and tracking can also leave
                # transient None entries (rm_keyframe sets the last slot
                # to None during a shift). Walking back from counter-1
                # ensures we pick up the most-recent KF that actually has
                # a Camera, instead of falling through to None.
                latest_cam = None
                cur = self.video.counter.value - 1
                while cur >= 0:
                    if cur < len(self.video.gs_viewpoints) and self.video.gs_viewpoints[cur] is not None:
                        latest_cam = self.video.gs_viewpoints[cur]
                        break
                    cur -= 1
                gui_utils.put_latest(
                    self.q_main2vis,
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians) if self.iteration_count % self.gui_interval == 0 else None,
                        current_frame=to_camera_vis(latest_cam) if latest_cam is not None else None,
                        keyframes=gui_keyframes,
                        kf_window=edge_dict,
                    )
                )
            
            if is_finished:
                break
        elapsed_time = time.time() - start_time
        print("[MR] elapsed time for mapping: {:.4f} sec".format(elapsed_time))

        if self.save_video:
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=True, frustum_overlay=True, return_flow=True)
            self.save_frame_to_video(self.video_writer_backcam_skip,
                                     img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=False, frustum_overlay=False, return_flow=True)
            self.save_frame_to_video(self.video_writer_skip,
                                    img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
        
        min_past_seen_index = -1
        if self.config["Training"]["adaptive_mapping_window"]:
            current_tstamps = self.video.tstamp[:self.video.counter.value].to(torch.int32)
            past_gsmask = torch.isin(current_tstamps, past_kfIDs)
            if past_gsmask.sum() > 0:
                past_indices = torch.nonzero(past_gsmask, as_tuple=False)
                min_past_seen_index = past_indices.min().item()
        return min_past_seen_index
    
    def remove_inactive_gaussians(self, interval=1000):
        if (int(self.iteration_count / interval) == self.remove_inactive_count):
            return
        
        print("==== Removing inactive gaussians... ====")
        self.remove_inactive_count += 1

        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        ii_inac = self.graph.ii_inac
        jj_inac = self.graph.jj_inac

        unique_active_indices = torch.unique(torch.cat([ii_torch, jj_torch], dim=0))
        unique_inactive_indices = torch.unique(torch.cat([ii_inac, jj_inac], dim=0))

        weights_sum_thres = 1e-2 #1e-2 # 1e-3
        total_retained_mask = None
        total_found_filter = None
        with torch.no_grad():
            for idx in unique_active_indices.cpu().tolist():
                viewpoint = self.video.gs_viewpoints[idx]
                render_pkg = render(viewpoint, viewpoint,
                                            self.gaussians, self.pipeline_params, self.background,
                                            use_flow=False, update_flow_conf=False)
                n_found = render_pkg["n_found"]  # (P,)
                found_mask = n_found > 0

                if total_found_filter is None:
                    total_found_filter = found_mask
                else:
                    total_found_filter = torch.logical_or(total_found_filter, found_mask)
                del render_pkg, n_found, found_mask

            for idx in unique_inactive_indices.cpu().tolist():
                viewpoint = self.video.gs_viewpoints[idx]
                render_pkg = render(viewpoint, viewpoint,
                                            self.gaussians, self.pipeline_params, self.background,
                                            use_flow=False, update_flow_conf=False)
                weights_sum = render_pkg["weights_sum"]  # (P,)
                retained_mask = weights_sum > weights_sum_thres
                if total_retained_mask is None:
                    total_retained_mask = retained_mask
                else:
                    total_retained_mask = torch.logical_or(total_retained_mask, retained_mask)
                del render_pkg, weights_sum, retained_mask

        total_inactive_mask = torch.logical_and(~total_retained_mask, ~total_found_filter)
        self.gaussians.prune_points(total_inactive_mask)
    
    def efficient_densify_and_prune(self, do_prune):
        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        curr_idx = self.video.counter.value-1
        # ii_curr_mask = (ii_torch == curr_idx)
        # jj_curr_mask = (jj_torch == curr_idx)

        # candidate = jj_torch[ii_curr_mask]
        # selected_mask = None
        # if candidate.numel() > 0:
        #     max_jj_val, max_jj_idx = torch.max(candidate, dim=0)
        #     global_indices = ii_curr_mask.nonzero(as_tuple=False).squeeze()
        #     selected_idx = global_indices[max_jj_idx]
        #     selected_mask = torch.zeros_like(jj_curr_mask, dtype=torch.bool)
        #     selected_mask[selected_idx] = True

        # candidate2 = ii_torch[jj_curr_mask]
        # if candidate2.numel() > 0:
        #     top2_vals, top2_indices = torch.topk(candidate2, k=min(4, candidate2.numel()))
        #     global_indices2 = jj_curr_mask.nonzero(as_tuple=False).squeeze()
        #     selected_indices2 = global_indices2[top2_indices]
            
        #     selected_mask2 = torch.zeros_like(jj_curr_mask, dtype=torch.bool)
        #     selected_mask2[selected_indices2] = True
        # else:
        #     selected_mask2 = None

        # if selected_mask is not None and selected_mask2 is not None:
        #     total_mask = torch.logical_or(selected_mask, selected_mask2)
        # elif selected_mask is not None:
        #     total_mask = selected_mask
        # elif selected_mask2 is not None:
        #     total_mask = selected_mask2
        # else:
        #     total_mask = torch.zeros_like(jj_curr_mask, dtype=torch.bool)

        # ii_curr_mask = (ii_torch >= curr_idx-25)
        ii_curr_mask = (ii_torch >= curr_idx-15)
        total_mask = ii_curr_mask
        if total_mask.numel() > 0:
        # if True:
            flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
            weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)
            # self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)
            # flow_confs = normalize_weights(weight_ups).unsqueeze(0)

            # # edges = torch.stack([ii_torch, jj_torch], dim=1)
            # # flows = flow_ups
            # # weights = self.graph.flow_confs.squeeze(0)
            # edges = torch.stack([ii_torch[total_mask], jj_torch[total_mask]], dim=1)
            # flows = flow_ups[total_mask]
            # # weights = self.graph.flow_confs[:, total_mask, ...].squeeze(0)
            # weights = flow_confs[:, total_mask, ...].squeeze(0)

            sel_idx = torch.nonzero(total_mask, as_tuple=True)[0]
            ii_sel = torch.index_select(ii_torch, 0, sel_idx)
            jj_sel = torch.index_select(jj_torch, 0, sel_idx)
            edges  = torch.stack([ii_sel, jj_sel], dim=1)
            flows  = torch.index_select(flow_ups, 0, sel_idx)
            sel_weight_ups = torch.index_select(weight_ups, 0, sel_idx)  # (E_sel, H, W, 2)
            weights = normalize_weights(sel_weight_ups)

            acm_list = [
                viewspace_point_tensor_acm,
                visibility_filter_acm,
                radii_acm,
                n_touched_acm,
                n_found_acm,
                error_per_gs_acm,
                error_per_gs_2_acm,
                error_per_gs_3_acm
            ] = (
                {}, {}, {}, {}, {}, {}, {}, {}
            )

            with torch.enable_grad():
                # loss_color_terms = []
                # loss_flow_terms = []
                # loss_aux_terms = []
                found_count = None

                loss_color_sum = torch.tensor(0.0, device=self.device)
                loss_flow_sum  = torch.tensor(0.0, device=self.device)
                loss_aux_sum   = torch.tensor(0.0, device=self.device)

                for e, (i,j) in enumerate(edges):
                    i_viewpoint = self.video.gs_viewpoints[i]
                    j_viewpoint = self.video.gs_viewpoints[j]

                    flow_e = flows[e].to(self.device, non_blocking=True)
                    i_viewpoint.flow_image = flow_e.permute(2, 0, 1).contiguous()
                    w_e = weights[e].to(self.device, non_blocking=True)
                    flow_conf_sum_prev = w_e.detach().sum()

                    i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
                    render_pkg = render(i_viewpoint, j_viewpoint,
                                        self.gaussians, self.pipeline_params, self.background,
                                        use_flow=True, update_flow_conf=False,
                                        flow_conf=w_e,
                                        train_pose=False)
                    
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        silh,
                        gsflow,
                        flow_cost,  # donguk
                        error_per_gs,
                        error_per_gs_2,
                        error_per_gs_3,
                        aux_image,
                        aux_image2,
                        aux_image3,
                        n_touched,
                        n_found,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["silh"],
                        render_pkg["gsflow"], # donguk
                        render_pkg["flowcost"], # donguk
                        render_pkg["error_per_gs"],
                        render_pkg["error_per_gs_2"],
                        render_pkg["error_per_gs_3"],
                        render_pkg["aux_image"],
                        render_pkg["aux_image2"],
                        render_pkg["aux_image3"],
                        render_pkg["n_touched"], 
                        render_pkg["n_found"],
                    )

                    # image_ab = torch.exp(i_viewpoint.exposure_a) * image + i_viewpoint.exposure_b
                    image_ab = image
                    rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                    Ll1 = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
                    if self.config["Training"]["ssim_mask_eroded"]:
                        mask_f = rgb_pixel_mask.float().unsqueeze(0)
                        eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                        rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                        Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                    else:
                        Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask
                    # Lssim = (1.0 - ssim_masked(image_ab, i_gt_image, rgb_pixel_mask))
                    loss1 = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean() 
                               + self.opt_params.lambda_dssim * Lssim.mean())
                    # loss_color_terms.append(loss1)
                    # loss_color_sum = loss_color_sum + loss1

                    loss_f = (flow_cost).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                    # loss_flow_terms.append(loss_f)
                    # loss_flow_sum = loss_flow_sum + loss_f

                    loss_aux = (Lssim.detach() * aux_image).sum();
                    loss_aux2 = (silh.detach() * aux_image2).sum();
                    loss_aux3 = (flow_cost.detach() * aux_image3).sum()
                    # loss_aux_terms.append(loss_aux)
                    # loss_aux_terms.append(loss_aux2)
                    # loss_aux_terms.append(loss_aux3)
                    # loss_aux_sum = loss_aux_sum + loss_aux + loss_aux2 + loss_aux3

                    if found_count is None:
                        found_count = (n_found > 0).to(torch.int32).detach()
                    else:
                        found_count += (n_found > 0).to(torch.int32).detach()

                    data = (
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        n_touched,
                        n_found,
                        error_per_gs,
                        error_per_gs_2,
                        error_per_gs_3,
                    )
                    update_acm(e, acm_list, data)
                    loss = loss1 + loss_f + loss_aux + loss_aux2 + loss_aux3
                    loss.backward()

                # loss_color = torch.stack(loss_color_terms).sum()
                # loss_flow = torch.stack(loss_flow_terms).sum()
                # loss_aux_sum = torch.stack(loss_aux_terms).sum()
                # loss_color = loss_color_sum
                # loss_flow  = loss_flow_sum
                # loss_aux_sum = loss_aux_sum
                # loss = loss_color + loss_flow + loss_aux_sum
                # loss.backward()
                # torch.cuda.synchronize()

            with torch.no_grad():
                if found_count is not None:
                    torch.cuda.synchronize()
                    total_found_filter = (found_count > 0).detach()
                    prune_kf_id_thres = self.video.tstamp[curr_idx - 25].cpu().item() if curr_idx >= 25 else -1

                    self.gaussians.densify_and_prune_by_error(radii_acm, error_per_gs_acm, error_per_gs_2_acm, error_per_gs_3_acm,
                                                            viewspace_point_tensor_acm, found_count, total_found_filter,
                                                            kf_uid=int(self.video.tstamp[curr_idx].cpu().item()), do_prune=do_prune,
                                                            prune_kf_id_thres=prune_kf_id_thres)
                    
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                for d in acm_list:
                    d.clear()


    def reset_n_found(self):
        with torch.no_grad():
            self.gaussians.n_found.fill_(0)

    def reset_found_count(self):
        with torch.no_grad():
            self.gaussians.found_count.fill_(0)

    def prune_unstable_gaussians(self):
        # torch.cuda.synchronize()
        with torch.no_grad():
            found_count = self.gaussians.n_found > 0
            last_not_found = self.gaussians.found_count < 1
            unstable_mask = torch.logical_and(found_count, last_not_found).to(self.device, non_blocking=True) #.cuda()
            self.gaussians.prune_points(unstable_mask)

        #     self.gaussians.optimizer.zero_grad(set_to_none=True)
        #     self.keyframe_optimizers.zero_grad(set_to_none=True)
        # torch.cuda.synchronize()
        torch.cuda.empty_cache()


    def estimate_dense_depth_from_flow(self, edge, flow):
        curr_idx, comp_idx = edge        
        if curr_idx < 1 or comp_idx < 1:
            return
        
        Twc1_tensor = SE3(self.video.poses[curr_idx.item()]).inv().matrix()
        Tc2w_tensor = SE3(self.video.poses[comp_idx.item()]).matrix()

        K_tensor = torch.tensor([
            [self.fx,  0., self.cx],
            [0.,  self.fy, self.cy],
            [0.,  0., 1.]
        ], dtype=torch.float32, device=self.device).contiguous()
        Tc2c1 = torch.matmul(Tc2w_tensor, Twc1_tensor)

        depth = torch.zeros_like(flow[..., 0]).contiguous().to(self.device)
        Rc2c1 = Tc2c1[:3, :3].contiguous().to(self.device)
        tc2c1 = Tc2c1[:3, 3:4].contiguous().to(self.device)
        gfslam_backends.estimate_flow_depth(flow, K_tensor, Rc2c1, tc2c1, depth, 0.01, 100.0)

        return depth
        
    ##### WARNING! DEPRECATED #####
    def solve_pose_by_gsflow(self):
        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        coords1 = self.get_gsflows(torch.stack([ii_torch, jj_torch], dim=1)) + self.graph.coords0
        self.graph.update_by_gs(coords1, None, None, use_inactive=True)

        flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
        weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)

        edges = torch.stack([ii_torch, jj_torch], dim=1)
        flows = flow_ups
        flowconfs = self.graph.flow_confs.squeeze(0)

        updating_indices = torch.unique(edges)
        self.synchronize_poses_to_gs(updating_indices)
        self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)

        self.setup_keyframe_optimizers(updating_indices)
        ### Get weigted jacobian & residuals / J: (E, N, 6), r: (E, N, 1)
        wJii, wJjj, wr, wweights = self.get_pose_jacobian(edges, flows, flowconfs, is_weighted=True, use_color=True)

        ### Get Non-weighted jacobian & residuals / J: (E, N, 6), r: (E, N, 1)
        Jii, Jjj, r, weights = self.get_pose_jacobian(edges, flows, flowconfs, is_weighted=False, use_color=True)

        delta_x_1 = solve_pose_update1(updating_indices, edges, Jii, Jjj, r, wJii, wJjj, wr)
        # delta_x_2 = solve_pose_update2(updating_indices, edges, Jii, Jjj, r, weights)

        for e, idx in enumerate(updating_indices.cpu().tolist()):
            delta_x = delta_x_1[e]
            viewpoint = self.video.gs_viewpoints[idx]
            update_pose_by_delta(viewpoint, delta_x)

        self.synchronize_poses_to_video(updating_indices)

        if self.use_gui:
            last_idx = updating_indices.max().item()
            curr_viewpoint = self.video.gs_viewpoints[last_idx]
            # self.q_main2vis
            gui_utils.put_latest(
                    self.q_main2vis,
                gui_utils.GaussianPacket(
                    current_frame=to_camera_vis(curr_viewpoint),
                    gtcolor=curr_viewpoint.original_image,
                    gtdepth=np.zeros((curr_viewpoint.image_height, curr_viewpoint.image_width)),
                    # gtflow=first_viewpoint.flow_vis,
                )
            )


    def initialize_map(self):

        # Nodes (K: kfs num, E: edges num)
        # initial_Tcw_SE3 = SE3(self.video.poses[:self.video.counter.value]) # tensor, (K, 7) (x,y,z,qx,qy,qz,qw)

        # Get Edges
        ii_torch = self.graph.ii
        jj_torch = self.graph.jj
        flow_ups = self.graph.flow_ups.squeeze(0) # tensor, (E, H, W, 2)
        weight_ups = self.graph.weight_ups.squeeze(0) # tensor, (E, H, W, 2)

        # Update intrinsics
        if self.video.counter.value > 0 and not hasattr(self, 'fx'):
            self.fx, self.fy, self.cx, self.cy = self.video.fx, self.video.fy, self.video.cx, self.video.cy
            self.projection_matrix = self.video.projection_matrix

        # Synchronize poses & flow confs 
        updating_indices = torch.unique(ii_torch)
        self.synchronize_poses_to_gs(updating_indices)
        # self.graph.flow_confs = normalize_weights(weight_ups).unsqueeze(0)
        if self.flow_func == 'log-logistic':
            flow_confs = normalize_weights(weight_ups).unsqueeze(0)
        else:
            flow_confs = weight_ups
        
        # Insert 3DGS
        first_idx = updating_indices.min().item()
        for idx in sorted(updating_indices):
            if idx == first_idx:
                insert_region = torch.ones((self.H, self.W), dtype=torch.bool, device=self.device)
                found_filter = None
                silh_filter = None
                depth_rendered = None
            else:
                insert_region, found_filter, silh_filter, depth_rendered = self.identify_insert_region(idx)
                if insert_region is None:  # tombstone slot — skip
                    continue
            self.create_new_gaussians(idx, insert_region, found_filter, silh_filter, depth=depth_rendered, is_init=(idx==first_idx))

        init_edges = torch.stack([ii_torch, jj_torch], dim=1)
        init_flows = flow_ups
        # init_flowconfs = self.graph.flow_confs.squeeze(0)
        init_flowconfs = flow_confs.squeeze(0)
        
        if self.use_gui:
            first_viewpoint = self.video.gs_viewpoints[first_idx]
            # self.q_main2vis.put(
            gui_utils.put_latest(
                self.q_main2vis,
                gui_utils.GaussianPacket(
                    current_frame=to_camera_vis(first_viewpoint),
                    gtcolor=first_viewpoint.original_image,
                    # gtdepth=np.zeros((first_viewpoint.image_height, first_viewpoint.image_width)),
                    # gtflow=first_viewpoint.flow_vis,
                )
            )
        
        self.setup_keyframe_optimizers(updating_indices)
        ##################### Initial BA v1 ######################
        # for train_num in range(self.init_itr_num):
        #     self.ba(train_num, init_edges, init_flows, init_flowconfs, pose_fix_idx=first_idx, train_map=True, 
        #             train_pose=False, flow_ba=True) # False if train_num < 50 else True
        # self.graph.update_flowconfs(init_flowconfs, mask=None)        
        ##########################################################


        ##################### Initial BA v2 ######################
        train_num = 0
        iters_per_edge = int(self.init_itr_num / len(init_edges))
        train_num_per_edge = [iters_per_edge] * len(init_edges)
        edge_idx_stack = [i for i in range(len(init_edges))]

        while True:
            edge_idx = random.choice(edge_idx_stack)
            if all(x == 0 for x in train_num_per_edge):
                break
            if train_num_per_edge[edge_idx] == 0:
                continue
            update_gaussian = (
                train_num > self.config["Training"]["init_gaussian_update"] and
                train_num % self.config["Training"]["init_gaussian_update"] #self.gaussian_update_every
                == self.config["Training"]["init_gaussian_update_offset"]
            )
            prune_gaussian = train_num % self.config["Training"]["init_gaussian_prune"] == self.config["Training"]["init_gaussian_update_offset"] \
                    and train_num > self.config["Training"]["init_gaussian_prune"]
                        
            last_stage_for_edge = (train_num_per_edge[edge_idx] == 1)
            train_num_per_edge[edge_idx] -= 1
            self.mapping_one_iter(train_num, init_edges[edge_idx], init_flows[edge_idx],
                                  init_flowconfs[edge_idx], last_stage_for_edge=last_stage_for_edge,
                                  train_pose=False, add_color_target_idx=False, is_init=True) # is_init=True
            if update_gaussian or prune_gaussian:
                self.efficient_densify_and_prune(do_prune=prune_gaussian)
            train_num += 1

            if self.use_gui and train_num % self.gui_interval == 0:
                gui_keyframes = [to_camera_vis(self.video.gs_viewpoints[i]) for i in range(self.video.counter.value)]
                first_edge_dict = {}
                first_edge_dict[int(self.video.tstamp[first_idx].cpu())] = self.video.tstamp[jj_torch[ii_torch == first_idx]].long().cpu().tolist()
                # self.q_main2vis.put(
                gui_utils.put_latest(
                    self.q_main2vis,
                    gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    keyframes=gui_keyframes,
                    kf_window=first_edge_dict,
                    current_frame=to_camera_vis(self.video.gs_viewpoints[self.video.counter.value - 1]),
                    )
                )
        ##########################################################


        self.synchronize_poses_to_video(updating_indices)
        print("Initialized 3DGS Map with {} keyframes.".format(len(updating_indices)))
        self.gsmap_initialized = True

        # Gui sync
        if self.use_gui:
            unique_list = torch.unique(ii_torch).tolist()
            # gui_keyframes = [self.video.gs_viewpoints[i] for i in unique_list]
            gui_keyframes = [self.video.gs_viewpoints[i] for i in range(self.video.counter.value)]
            first_edge_dict = {}
            first_edge_dict[int(self.video.tstamp[first_idx].cpu())] = self.video.tstamp[jj_torch[ii_torch == first_idx]].long().cpu().tolist()
            
            # self.q_main2vis.put(
            gui_utils.put_latest(
                self.q_main2vis,
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=to_camera_vis(self.video.gs_viewpoints[first_idx]),
                    keyframes=gui_keyframes,
                    kf_window=first_edge_dict,
                )
            )

        ### Save to video_writer_backcam_skip / video_writer_skip
        if self.save_video:
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=True, frustum_overlay=True, return_flow=True)
            self.save_frame_to_video(self.video_writer_backcam_skip,
                                     img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)
            img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis \
                    = self.render_from_pose(self.video.gs_viewpoints[self.video.counter.value-1], None, 
                                            behind=False, frustum_overlay=False, return_flow=True)
            self.save_frame_to_video(self.video_writer_skip,
                                    img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis)

    def create_new_gaussians(self, idx, insert_region, found_filter, silh_filter, depth=None, is_init=False):
        gs_viewpoint = self.video.gs_viewpoints[idx]
        with torch.no_grad():
            # gt_img = gs_viewpoint.original_image.cuda()
            gt_img = gs_viewpoint.original_image.to(self.device, non_blocking=True)
            valid_rgb1 = (gt_img.sum(dim=0) > self.rgb_boundary_threshold)

            # initial_depth = 1.0 / self.video.disps_up[idx].cuda()
            disps = self.video.disps_up[idx].to(self.device, non_blocking=True)
            initial_depth = torch.where(disps > 0, disps.reciprocal(), torch.zeros_like(disps))
            if depth is not None and silh_filter is not None:
                # depth = depth.squeeze()
                # silh_filter = silh_filter.squeeze()
                # initial_depth[silh_filter] = depth[silh_filter]
                d = depth.squeeze().to(self.device, non_blocking=True)
                s = silh_filter.squeeze().to(self.device, non_blocking=True)
                initial_depth = torch.where(s, d, initial_depth)

            valid_rgb2 = (initial_depth > 0.0) & (initial_depth < 500.0)
            valid_rgb = valid_rgb1 & valid_rgb2

            # initial_depth[~valid_rgb] = 0.0  # Set invalid regions to median depth
            # initial_depth[~insert_region] = 0.0  # Set regions outside the insert region to zero depth
            initial_depth.masked_fill_(~valid_rgb, 0.0)
            initial_depth.masked_fill_(~insert_region.to(self.device), 0.0)
            initial_depth = initial_depth.detach().cpu().numpy()

        # with torch.no_grad():
            self.gaussians.extend_from_pcd_seq(gs_viewpoint, kf_id=int(self.video.tstamp[idx].cpu().item()), 
                                            init=is_init, scale=2.0, depthmap=initial_depth, found_filter=found_filter)


    def identify_insert_region(self, idx):
        viewpoint = self.video.gs_viewpoints[idx]
        # fast_mode tombstone: retired keyframe slot — caller should skip.
        if viewpoint is None:
            return None, None, None, None
        gt_image = viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()

        ssim_thres = 0.5
        energy_thres_max = 0.7 #0.7 #0.9
        energy_thres_min = 0.3

        # with torch.enable_grad():
        with torch.no_grad():
            render_pkg = render(viewpoint, viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=False, update_flow_conf=False)
            (
                image,
                visibility_filter,
                depth,
                silh,
                n_touched,
                n_found,
                
            ) = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["depth"],
                render_pkg["silh"],
                render_pkg["n_touched"], 
                render_pkg["n_found"],
            )

            # image_ab = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
            image_ab = image
            rgb_pixel_mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
            Ll1 = (l1_loss(image_ab, gt_image) * rgb_pixel_mask).detach()
            if self.config["Training"]["ssim_mask_eroded"]:
                mask_f = rgb_pixel_mask.float().unsqueeze(0)
                eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                Lssim = ((1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask_eroded).detach()
            else:
                Lssim = ((1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask).detach()
            # Lssim = (1.0 - ssim_masked(image_ab, gt_image, rgb_pixel_mask))

            silh_mask = silh.squeeze() < energy_thres_max
            Lssim_mask = Lssim.squeeze() > ssim_thres
            # Lssim_mask = Lssim.squeeze() > ssim_thres
            insert_region = torch.logical_or(silh_mask, Lssim_mask)#.to(self.device)
            # insert_region = torch.logical_and(silh_mask, Lssim_mask).to(self.device)
    
        return insert_region.detach(), (n_found > 0).detach(), (silh > 0.95).detach(), depth.detach()
    
    def reset_nonfound_gaussians(self, check_all_kfs=True, interval=5000, force_reset=False):
        total_filter = None
        if (int(self.iteration_count / interval) == self.reset_opacity_count) and not force_reset:
            return
        
        print("Resettting the opacity of non-found Gaussians....")
        self.reset_opacity_count += 1

        if check_all_kfs:
            with torch.no_grad():
                for idx in range(self.video.counter.value):
                    viewpoint = self.video.gs_viewpoints[idx]
                    render_pkg = render(viewpoint, viewpoint,
                                    self.gaussians, self.pipeline_params, self.background,
                                    use_flow=False, update_flow_conf=False)
                    (
                        visibility_filter,
                        n_touched,
                        n_found,
                    ) = (
                        render_pkg["visibility_filter"],
                        render_pkg["n_touched"], 
                        render_pkg["n_found"],
                    )
                    if total_filter is None:
                        total_filter = (n_found > 0).detach()
                    else:
                        total_filter = torch.logical_or(total_filter, (n_found > 0).detach())
        else:
            total_filter = (self.gaussians.n_found > 0).detach()

        # self.gaussians.reset_opacity_by_filter(total_filter)
                        
        reset_mask = ~total_filter
        self.gaussians.prune_points(reset_mask)

        if self.use_gui:
            # self.q_main2vis.put(
            gui_utils.put_latest(
                self.q_main2vis,
                gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians))
            )

    def color_refinement(self, use_flow=False):
        Log("Starting color refinement")

        iteration_total = 26000
        # for iteration in tqdm(range(1, iteration_total + 1)):

        # ii_torch = self.graph.ii
        # jj_torch = self.graph.jj
        # edges = torch.stack([ii_torch, jj_torch], dim=1)
        # updating_indices = torch.unique(edges)
        # self.setup_keyframe_optimizers(updating_indices, is_refinement=True)
        if not use_flow:
            for iteration in (pbar := trange(1, iteration_total + 1)):
                # viewpoint_idx_stack = [i for i, viewpoint in enumerate(self.video.gs_viewpoints) if viewpoint is not None]
                viewpoint_idx_stack = [i for i in range(self.video.counter.value)]
                viewpoint_cam_idx = viewpoint_idx_stack.pop(
                    random.randint(0, len(viewpoint_idx_stack) - 1)
                )
                viewpoint_cam = self.video.gs_viewpoints[viewpoint_cam_idx]
                render_pkg = render(
                    viewpoint_cam, viewpoint_cam, 
                    self.gaussians, self.pipeline_params, self.background
                )
                image, visibility_filter, radii, silh = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["silh"],
                )

                image_ab = image
                gt_image = viewpoint_cam.original_image.to(self.device, non_blocking=True) #.cuda()
                rgb_pixel_mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
                Ll1 = l1_loss(image_ab, gt_image) * rgb_pixel_mask
                # loss_silh = 0.02 * (1 - silh).mean()
                # self.opt_params.lambda_dssim
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = ((1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask)
                # Lssim = (1.0 - ssim_masked(image_ab, gt_image, rgb_pixel_mask))
                loss = (1.0 - self.opt_params.lambda_dssim) * Ll1.mean() + self.opt_params.lambda_dssim * Lssim.mean() #+ loss_silh
                loss.backward()
                # torch.cuda.synchronize()
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    lr = self.gaussians.update_learning_rate(iteration, is_refinement=(iteration==0))

                    # self.keyframe_optimizers.step()
                    # self.keyframe_optimizers.zero_grad(set_to_none=True)

                pbar.set_description(f"Color Refinement lr {lr:.3E} loss {loss.item():.3f}")

                if self.use_gui and iteration % 200 == 0:
                    # self.q_main2vis.put(
                    gui_utils.put_latest(
                        self.q_main2vis,
                        gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians))
                        )
        else:
            # edges_ac = torch.stack([self.graph.ii, self.graph.jj], dim=1)
            # flow_ups = self.graph.flow_ups.squeeze(0)
            # weight_ups = self.graph.weight_ups.squeeze(0)
            # flow_confs = normalize_weights(weight_ups)

            # edges_inac = torch.stack([self.graph.ii_inac, self.graph.jj_inac], dim=1)
            # flow_inac = self.graph.flow_ups_inac.squeeze(0).to(self.device, non_blocking=True)
            # weight_inac = self.graph.weight_ups_inac.squeeze(0).to(self.device, non_blocking=True)
            # flow_confs_inac = normalize_weights(weight_inac)

            # total_edges = torch.cat([edges_ac, edges_inac], dim=0)
            # total_flows = torch.cat([flow_ups, flow_inac], dim=0)
            # total_weights = torch.cat([flow_confs, flow_confs_inac], dim=0)

            # for iteration in (pbar := trange(1, iteration_total + 1)):
            #     edge_idx_stack = [i for i in range(total_edges.shape[0])]
            #     edge_idx = edge_idx_stack.pop(random.randint(0, len(edge_idx_stack) - 1))
            #     (i,j) = total_edges[edge_idx]
            #     flow = total_flows[edge_idx]
            #     weight = total_weights[edge_idx]

            #     i_viewpoint = self.video.gs_viewpoints[i]
            #     j_viewpoint = self.video.gs_viewpoints[j]

            #     i_viewpoint.flow_image = flow.to(self.device, non_blocking=True).permute(2, 0, 1) #.cuda()
            #     flow_conf_sum_prev = weight.to(self.device, non_blocking=True).sum() #.cuda().detach().sum()

            #     i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True) #.cuda()
            #     render_pkg = render(i_viewpoint, j_viewpoint,
            #                         self.gaussians, self.pipeline_params, self.background,
            #                         use_flow=True, update_flow_conf=False,
            #                         flow_conf=weight.to(self.device, non_blocking=True), #.cuda(),
            #                         train_pose=False)
                
            #     (
            #         image,
            #         gsflow,
            #         flow_cost,
            #     ) = (
            #         render_pkg["render"],
            #         render_pkg["gsflow"],
            #         render_pkg["flowcost"], 
            #     )

            #     image_ab = image
            #     rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
            #     Ll1 = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
            #     if self.config["Training"]["ssim_mask_eroded"]:
            #         mask_f = rgb_pixel_mask.float().unsqueeze(0)
            #         eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
            #         rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
            #         Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
            #     else:
            #         Lssim = ((1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask)
            #     loss_color_i = ((1.0 - 0.2) * Ll1.mean() + 0.2 * Lssim.mean()) * FLOAT_SCALING
                
            #     flow_thresholding = (i_viewpoint.flow_image.sum(dim=0) > 0.01).view(*self.mask_shape)
            #     gsflow_thresholding = (gsflow.sum(dim=0) > 0.01).view(*self.mask_shape)
            #     mask = flow_thresholding & gsflow_thresholding
            #     loss_flow = (flow_cost * mask * rgb_pixel_mask).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))

            #     scaling = self.gaussians.get_scaling
            #     isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean() * FLOAT_SCALING
            #     max_scales = torch.max(scaling, dim=1).values  
            #     min_scales = torch.min(scaling, dim=1).values
            #     ratio_scales = max_scales / (min_scales + 1e-6)
            #     loss_scale_ratio = torch.clamp(ratio_scales - 3, min=0).mean() * FLOAT_SCALING

            #     loss = loss_color_i + 0.1 * loss_flow + isotropic_loss # + loss_scale_ratio
            #     loss.backward()

            #     with torch.no_grad():
            #         self.gaussians.optimizer.step()
            #         self.gaussians.optimizer.zero_grad(set_to_none=True)

            #         # self.keyframe_optimizers.step()
            #         # self.keyframe_optimizers.zero_grad(set_to_none=True)

            #     pbar.set_description(f"Color Refinement lr {lr:.3E} loss {loss.item():.3f}")

            #     if self.use_gui and iteration % 200 == 0:
            #         self.q_main2vis.put(
            #             gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians))
            #             )

            edges_ac   = torch.stack([self.graph.ii, self.graph.jj], dim=1)                  # [E_ac, 2], small → ok on CPU
            flow_ac    = self.graph.flow_ups.squeeze(0).to(self.device, non_blocking=True)   # [E_ac, H, W, 2] (or C)
            weight_ac  = self.graph.weight_ups.squeeze(0).to(self.device, non_blocking=True) # [E_ac, H, W, 1] (or C)

            has_inac = (getattr(self.graph, "ii_inac", None) is not None) and \
                    (getattr(self.graph, "jj_inac", None) is not None) and \
                    (getattr(self.graph, "flow_ups_inac", None) is not None) and \
                    (getattr(self.graph, "weight_ups_inac", None) is not None)

            if has_inac:
                edges_inac = torch.stack([self.graph.ii_inac, self.graph.jj_inac], dim=1)    # [E_inac, 2], small
                flow_inac_cpu   = self.graph.flow_ups_inac.squeeze(0).pin_memory()           # [E_inac, H, W, 2]
                weight_inac_cpu = self.graph.weight_ups_inac.squeeze(0).pin_memory()         # [E_inac, H, W, 1]
                E_inac = edges_inac.shape[0]
            else:
                edges_inac = None
                E_inac = 0

            E_ac   = edges_ac.shape[0]
            E_all  = E_ac + E_inac

            with torch.no_grad():
                flow_conf_ac = normalize_weights(weight_ac)  # [E_ac, H, W, 1] on GPU

            for iteration in (pbar := trange(1, iteration_total + 1)):
                edge_idx_global = random.randint(0, E_all - 1)

                if edge_idx_global < E_ac:
                    # ACTIVE
                    k = edge_idx_global
                    i, j = edges_ac[k].tolist()
                    flow   = flow_ac[k]                      # already on GPU
                    weight = flow_conf_ac[k]                 # already normalized on GPU
                    flow_conf_sum_prev = weight.sum()        # GPU scalar
                else:
                    # INACTIVE (lazy upload + on-the-fly normalize)
                    k = edge_idx_global - E_ac
                    i, j = edges_inac[k].tolist()
                    with torch.no_grad():
                        flow_k_cpu   = flow_inac_cpu[k]      # [H, W, 2] on CPU (pinned)
                        weight_k_cpu = weight_inac_cpu[k]    # [H, W, 1] on CPU (pinned)
                        flow   = flow_k_cpu.to(self.device, non_blocking=True)
                        w_raw  = weight_k_cpu.to(self.device, non_blocking=True)
                        weight = normalize_weights(w_raw)     # [H, W, 1] on GPU
                        flow_conf_sum_prev = weight.sum()

                i_viewpoint = self.video.gs_viewpoints[i]
                j_viewpoint = self.video.gs_viewpoints[j]

                i_viewpoint.flow_image = flow.permute(2, 0, 1).contiguous()  # [2,H,W] or [C,H,W] on GPU

                i_gt_image = i_viewpoint.original_image.to(self.device, non_blocking=True)

                render_pkg = render(
                    i_viewpoint, j_viewpoint, self.gaussians, self.pipeline_params, self.background,
                    use_flow=True, update_flow_conf=False,
                    flow_conf=weight,
                    train_pose=False
                )

                image   = render_pkg["render"]
                gsflow  = render_pkg["gsflow"]
                flow_cost = render_pkg["flowcost"]

                image_ab = image
                rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)

                Ll1  = l1_loss(image_ab, i_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image_ab, i_gt_image)) * rgb_pixel_mask

                loss_color_i = ((1.0 - 0.2) * Ll1.mean() + 0.2 * Lssim.mean()) * FLOAT_SCALING

                # Flow is signed (u,v): use the componentwise absolute sum.
                # Algebraic u+v cancels opposite components and drops
                # negative motion even when the displacement is valid.
                flow_thresholding   = (i_viewpoint.flow_image.abs().sum(dim=0) > 0.01).view(*self.mask_shape)
                gsflow_thresholding = (gsflow.abs().sum(dim=0) > 0.01).view(*self.mask_shape)
                mask = flow_thresholding & gsflow_thresholding

                loss_flow = (flow_cost * mask * rgb_pixel_mask).sum() / ((flow_conf_sum_prev / FLOAT_SCALING) + 1e-6)

                scaling = self.gaussians.get_scaling
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean() * FLOAT_SCALING
                # max_scales = torch.max(scaling, dim=1).values
                # min_scales = torch.min(scaling, dim=1).values
                # loss_scale_ratio = torch.clamp(max_scales / (min_scales + 1e-6) - 3, min=0).mean() * FLOAT_SCALING

                loss = loss_color_i + 0.1 * loss_flow + isotropic_loss
                loss.backward()

                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    lr = self.gaussians.update_learning_rate(iteration, is_refinement=(iteration==0))

                pbar.set_description(f"Color Refinement lr {lr:.3E} loss {loss.item():.3f}")

                if self.use_gui and iteration % 200 == 0:
                    # self.q_main2vis.put(
                    gui_utils.put_latest(
                        self.q_main2vis,
                        gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians))
                    )

        Log("Map refinement done")

    def save_rendering(self, image_stream, traj, kf_indices, save_dir=None, iteration="before_opt"):
        eval_rendering(image_stream, traj, kf_indices, 
                       self.gaussians, self.pipeline_params, self.background,
                       self.projection_matrix, self.save_dir if save_dir is None else save_dir, iteration)
    

    def save_kf_images(self, interval, force_save=False):
        # if int(self.iteration_count / interval) > self.save_count:
        if int(self.video.counter.value / interval) > self.save_count or force_save:
            self.save_count += 1
            eval_rendering_kf(self.video.counter.value, self.video.gs_viewpoints,
                            self.video.tstamp, self.gaussians, self.pipeline_params, self.background,
                            self.debug_dir, self.iteration_count)
            return True
        
        else:
            return False
        
    def render_from_pose(self, viewpoint_input, Tcw_tensor=None, behind=False, frustum_overlay=False,
                         behind_offset=np.array([0.0, -2.5, -30.0]), frustum_size=0.03,
                         line_color=(0,255,0),line_thickness=2, return_flow=False):
        # behind_offset=np.array([0.0, -2.5, -30.0])

        use_flow = return_flow

        Tcw_behind = None
        gtcolor = None
        estflow_vis = None
        gsflow_vis = None
        if viewpoint_input is not None:
            gtcolor = viewpoint_input.original_image.detach().cpu().numpy().transpose(1, 2, 0)
            if behind:
                Tcw = torch.eye(4, device=self.device)
                Tcw[:3,:3] = viewpoint_input.R
                Tcw[:3,3] = viewpoint_input.T
                Tcw_behind = gui_utils.get_behind_view_Tcw(Tcw.to(self.device), offset_local=behind_offset)
                viewpoint = Camera.init_from_gui(uid=-1, 
                                             T=Tcw_behind,
                                             FoVx=self.video.fovx, FoVy=self.video.fovy,
                                             fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, 
                                             H=self.H, W=self.W)
                viewpoint.update_RT(Tcw_behind[:3,:3], Tcw_behind[:3,3])
            else:
                viewpoint = viewpoint_input
        elif Tcw_tensor is not None:
            use_flow = False
            viewpoint = Camera.init_from_gui(uid=-1, 
                                             T=Tcw_tensor,
                                             FoVx=self.video.fovx, FoVy=self.video.fovy,
                                             fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, 
                                             H=self.H, W=self.W)
            viewpoint.update_RT(Tcw_tensor[:3,:3], Tcw_tensor[:3,3])
            if behind:
                Tcw_behind = gui_utils.get_behind_view_Tcw(Tcw_tensor.to(self.device), offset_local=behind_offset)
                viewpoint.update_RT(Tcw_behind[:3,:3], Tcw_behind[:3,3])
        else:
            return None, None, None, None, None
        
        gsflow = None
        if use_flow and viewpoint_input is not None:
            flow_ups = self.graph.flow_ups.squeeze(0)
            mask = (self.video.tstamp == int(viewpoint_input.uid))
            idxs = torch.nonzero(mask, as_tuple=True)[0]
            if idxs.numel() > 0:
                idx = int(idxs[0].item())
                ii_torch = self.graph.ii
                jj_torch = self.graph.jj

                flow_ups = self.graph.flow_ups.squeeze(0)
                weight_ups = self.graph.weight_ups.squeeze(0)

                input_mask = (ii_torch == idx)
                masked_idx = torch.nonzero(input_mask, as_tuple=True)[0]
                first_edge = masked_idx[0]

                flows_for_idx = flow_ups[first_edge].to(self.device, non_blocking=True).permute(2,0,1).detach()
                i = int(ii_torch[first_edge].item())
                j = int(jj_torch[first_edge].item())
                comp_viewpoint = self.video.gs_viewpoints[j]
                gsflow = render(viewpoint_input, comp_viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=use_flow, update_flow_conf=False, train_pose=False)["gsflow"].detach()
                
                estflow_vis = gui_utils.optical_flow_to_rgb(flows_for_idx)
                gsflow_vis = gui_utils.optical_flow_to_rgb(gsflow)
                cv2.cvtColor(estflow_vis, cv2.COLOR_BGR2RGB, dst=estflow_vis)
                cv2.cvtColor(gsflow_vis, cv2.COLOR_BGR2RGB, dst=gsflow_vis)

        render_pkg = render(viewpoint, viewpoint,
                            self.gaussians, self.pipeline_params, self.background,
                            use_flow=use_flow, update_flow_conf=False, train_pose=False)
        (
            image,
            depth,
        ) = (
            render_pkg["render"],
            render_pkg["depth"],
        )

        img_t = torch.clamp(image, 0.0, 1.0)  # (H,W,3) float
        img_rgb = (img_t * 255.0).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)    # RGB uint8
        img_bgr = img_rgb[..., ::-1].copy()

        depth_disp = np.nan_to_num(depth.detach().squeeze().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        depth_disp = np.clip(depth_disp, 0, np.percentile(depth_disp, 99))
        depth_norm = cv2.normalize(depth_disp, None, 0, 255, cv2.NORM_MINMAX)
        if depth_norm.ndim == 3:
            depth_norm = depth_norm.squeeze()
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

        if viewpoint_input is not None and behind and frustum_overlay:
            cur_idx = self.video.counter.value - 1
            gui_keyframes = [self.video.gs_viewpoints[i] for i in range(cur_idx + 1)]

            for kf in gui_keyframes:
                if kf.uid == viewpoint_input.uid:
                    color = line_color
                    thickness = line_thickness
                else:
                    color = (255, 0, 0)
                    thickness = 1

                P_local, L = gui_utils.frustum_points_local(size=frustum_size)
                P_local = torch.from_numpy(P_local).to(kf.R.dtype).to(kf.R.device)

                Rcw_in = kf.R
                tcw_in = kf.T
                Rwc_in = Rcw_in.transpose(-1, -2)
                twc_in = (-Rwc_in @ tcw_in.unsqueeze(-1)).squeeze(-1)

                Rcw_b = viewpoint.R
                tcw_b = viewpoint.T

                # local -> world -> behind camera
                Pw = (Rwc_in @ P_local.T) + twc_in.unsqueeze(-1)  # (3,N)
                Pb = (Rcw_b @ Pw) + tcw_b.unsqueeze(-1)           # (3,N)

                # project
                Pb = Pb.T.cpu().numpy()
                K_np = np.array([[self.fx, 0, self.cx],
                                [0, self.fy, self.cy],
                                [0, 0, 1]], dtype=np.float32)
                pts2d = gui_utils.project_points(K_np, Pb)

                # draw
                gui_utils.draw_wireframe(img_bgr, pts2d, L, color=color, thickness=thickness)
            
        return img_bgr, gtcolor, depth_color, estflow_vis, gsflow_vis
    
    def _to_bgr_uint8(self, img, fallback_shape=None):
        """Pass through when possible, else normalize to BGR/HWC/uint8.

        Returns a black canvas when ``img`` is None.
        """
        if img is None:
            if fallback_shape is None:
                return None
            H, W = fallback_shape
            return np.zeros((H, W, 3), dtype=np.uint8)

        try:
            if isinstance(img, torch.Tensor):
                t = img.detach().cpu()
                # [C,H,W] -> [H,W,C]
                if t.dim() == 3 and t.shape[0] in (1,3):
                    t = t.permute(1,2,0)
                elif t.dim() == 4 and t.shape[0] == 1:
                    t = t.squeeze(0).permute(1,2,0)
                if t.dtype != torch.uint8:
                    t = (t.clamp(0,1) * 255).to(torch.uint8)
                img = t.numpy()
        except Exception:
            pass

        # numpy
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.dtype != np.uint8:
            if img.max() <= 1.01:
                img = (np.clip(img,0,1)*255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return img

    def _letterbox_bgr(self, img_bgr, W, H, bg=(255,255,255)):
        """Aspect-preserving resize plus white padding to exactly (H, W, 3)."""
        if img_bgr is None:
            return np.full((H, W, 3), bg, dtype=np.uint8)
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.full((H, W, 3), bg, dtype=np.uint8)
        scale = min(W / w, H / h)
        nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
        canvas = np.full((H, W, 3), bg, dtype=np.uint8)
        x0 = (W - nw) // 2
        y0 = (H - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def mosaic_frame_size(self, W, H, gap):
        """Final frame size: (H, 2W + 2*gap); the height stays at H."""
        W_out = 2*W + 2*gap
        H_out = H
        if W_out % 2: W_out += 1
        if H_out % 2: H_out += 1
        return W_out, H_out

    def save_frame_to_video(self, video_writer,
                            img_bgr, gtcolor, depth_color,
                            estflow_vis=None, gsflow_vis=None,
                            bg=(255,255,255), keep_aspect=False):
        """
        Final layout (height=H, width=2W+2*gap):
        [left W]      : img_bgr at its original size (not resized)
        [gap]         : spacing between the two columns
        [right W]     : 2x2 tiles (top: gt | depth, bottom: est | gsflow)
                        - no horizontal spacing inside (total width preserved)
                        - a single vertical gap between the two rows
        [trailing gap]: right margin
        """
        H, W = self.H, self.W
        gap  = getattr(self, "_mosaic_gap", max(4, H // 128))

        def to_bgr(img):
            return self._to_bgr_uint8(img, fallback_shape=(H, W))

        left_bgr = to_bgr(img_bgr)
        gt_bgr   = to_bgr(gtcolor)
        dep_bgr  = to_bgr(depth_color)
        est_bgr  = to_bgr(estflow_vis)
        gsf_bgr  = to_bgr(gsflow_vis)

        gt_bgr = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        W2 = max(1, W // 2)
        h_top = max(1, (H - gap) // 2)
        h_bot = max(1, H - gap - h_top)

        def resize_tile(img, w, h):
            if img is None:
                return np.full((h, w, 3), bg, dtype=np.uint8)
            if keep_aspect:
                ih, iw = img.shape[:2]
                if ih == 0 or iw == 0:
                    return np.full((h, w, 3), bg, dtype=np.uint8)
                s = min(w/iw, h/ih)
                nw, nh = max(1, int(round(iw*s))), max(1, int(round(ih*s)))
                inter = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                rs = cv2.resize(img, (nw, nh), interpolation=inter)
                canvas = np.full((h, w, 3), bg, dtype=np.uint8)
                x0 = (w - nw) // 2
                y0 = (h - nh) // 2
                canvas[y0:y0+nh, x0:x0+nw] = rs
                return canvas
            inter = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_LINEAR
            return cv2.resize(img, (w, h), interpolation=inter)

        col_left = left_bgr  # (H, W, 3)

        top_left   = resize_tile(gt_bgr,  W2, h_top)
        top_right  = resize_tile(dep_bgr, W2, h_top)
        bot_left   = resize_tile(est_bgr, W2, h_bot)
        bot_right  = resize_tile(gsf_bgr, W2, h_bot)

        row_top = np.hstack([top_left, top_right])      # (h_top, W, 3)
        row_bot = np.hstack([bot_left, bot_right])      # (h_bot, W, 3)

        hgap = np.full((gap, W, 3), bg, dtype=np.uint8)
        col_right = np.vstack([row_top, hgap, row_bot]) # (H, W, 3)

        vgap = np.full((H, gap, 3), bg, dtype=np.uint8)
        mosaic = np.hstack([col_left, vgap, col_right, vgap])  # (H, 2W+2*gap, 3)

        video_writer.write(np.ascontiguousarray(mosaic))
        return mosaic
    
    def release_video_writers(self):
        if self.save_video:
            self.video_writer_backcam_full.release()
            self.video_writer_backcam_skip.release()
            self.video_writer_curr.release()
            self.video_writer_skip.release()

        # self.video_writer_backcam_track.release()
        # self.video_writer_fixview.release()

    def shutdown_gui(self, timeout=2.0):
        """Close the Open3D child process and reap it before Python exits."""
        gui_process = getattr(self, "gui_process", None)
        if gui_process is None:
            return

        if gui_process.is_alive():
            try:
                while True:
                    self.q_main2vis.get_nowait()
            except Exception:
                pass

            try:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(finish=True), timeout=0.5
                )
            except Exception:
                pass
            gui_process.join(timeout=timeout)

        if gui_process.is_alive():
            gui_process.terminate()
            gui_process.join(timeout=1.0)

        if gui_process.is_alive() and hasattr(gui_process, "kill"):
            gui_process.kill()
            gui_process.join(timeout=1.0)

        for queue_name in ("q_main2vis", "q_vis2main"):
            process_queue = getattr(self, queue_name, None)
            if process_queue is not None:
                try:
                    process_queue.close()
                    process_queue.cancel_join_thread()
                except Exception:
                    pass

        try:
            gui_process.close()
        except Exception:
            pass
        self.gui_process = None

    def stamp_min_uid_for_gaussians(self):
        window_indices = torch.unique(self.graph.ii)
        min_uid = int(self.video.tstamp[window_indices].min().item())
        window_found_filter = (self.gaussians.n_found > 0).detach()
        self.gaussians.stamp_uids(min_uid, window_found_filter)
