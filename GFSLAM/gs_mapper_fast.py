"""fast_mode methods for GSMapper.

Worker thread (mapping queue) + fast_mode optimization passes. Designed
to be inherited by ``GSMapper`` so all methods share the same ``self``
and the same per-instance attributes (gaussians, video, config, ...).
"""

import os
import gc
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lietorch import SE3

import gfslam_backends
from utils.flow_utils import normalize_weights
from utils.slam_utils import (
    update_acm, vectorize_and_stack_values, vectorize_and_stack_grads,
    normalize_stacked_tensor, reduce_max, reduce_mean, reduce_median,
)
from utils.multiprocessing_utils import clone_obj
from utils.camera_utils import Camera, to_camera_vis
from gaussian_splatting.gui import gui_utils
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, ssim_masked


FLOAT_SCALING = 1.0


class GSMapperFast:
    # ----------------------------------------------------------------------
    # Async helpers (fast_mode).
    # `enqueue_mapping_task` pushes a task onto the persistent mapping queue;
    # tracking blocks via wait_async after each enqueue so mapping fully
    # processes the KF before tracking advances. `wait_async` drains
    # everything (called from terminate).
    # ----------------------------------------------------------------------
    def _mapping_worker_loop(self):
        """Persistent consumer for fast_mode.

        Behaviour:
          * If an *insert* task is queued, drain it (fast: ~0.5s/KF) and
            top up the refinement budget by `_refine_per_insert` iters.
          * If the queue is empty *and* the refinement budget is positive,
            run one refinement pass (chunk size 100 iters, consumes from
            the budget). Always reads the latest graph state.
          * Otherwise sleep briefly so we don't burn the CPU spinning.

        Refinement delegates back into the frontend via
        ``self._refine_callback``. Tracking blocks via wait_async after
        each enqueue, so the queue alternates between 0 and 1 items;
        refinement runs opportunistically during the tracking-side gap
        between enqueues (when the queue is empty and budget remains).
        """
        import queue as _queue
        import time as _time
        while True:
            try:
                task = self._mapping_queue.get(timeout=0.05)
            except _queue.Empty:
                if self._refine_budget > 0 and self._refine_callback is not None:
                    try:
                        self._refine_callback()
                    except BaseException as e:
                        self._async_exception = e
                else:
                    # Budget exhausted — wait for next insert before doing
                    # any more refinement work. Avoids tight-loop and lock
                    # starvation that previously had the tracking thread
                    # waiting on the worker.
                    _time.sleep(0.05)
                continue
            try:
                if task is None:  # sentinel for shutdown
                    return
                fn, args, kwargs = task
                fn(*args, **kwargs)
                # Top up refinement budget for this newly-inserted KF.
                self._refine_budget += self._refine_per_insert
            except BaseException as e:
                self._async_exception = e
            finally:
                self._mapping_queue.task_done()

    def enqueue_mapping_task(self, target, *args, **kwargs):
        if self._mapping_queue is None:
            raise RuntimeError(
                "enqueue_mapping_task called but fast_mode is off")
        if self._async_exception is not None:
            exc, self._async_exception = self._async_exception, None
            raise exc
        self._mapping_queue.put((target, args, kwargs))

    def wait_async(self):
        """Block until the mapping queue has fully drained. Used both per-KF
        (tracking-blocks-mapping cadence) and at terminate()."""
        if self._mapping_queue is not None:
            self._mapping_queue.join()
        if self._async_exception is not None:
            exc, self._async_exception = self._async_exception, None
            raise exc

    def install_refine_callback(self, callback):
        """GFSLAMFrontend calls this to register the refinement function and
        start the worker. Done lazily so the worker can call back into the
        frontend without ordering issues."""
        import threading
        self._refine_callback = callback
        if self._mapping_queue is not None and self._mapping_worker is None:
            self._mapping_worker = threading.Thread(
                target=self._mapping_worker_loop, daemon=True
            )
            self._mapping_worker.start()

    def shutdown_mapping_worker(self):
        """Signal the persistent mapping worker to exit (used at terminate)."""
        if self._mapping_queue is not None and self._mapping_worker is not None:
            self._mapping_queue.put(None)
            self._mapping_worker.join(timeout=5.0)
            self._mapping_worker = None
            self._mapping_queue = None
    # ------------------------------------------------------------------
    # fast_mode lock-free entry points.
    # All inputs come from a payload built by tracking under the video lock;
    # mapping uses only direct Camera refs + cloned tensors and never
    # dereferences live ``self.video.gs_viewpoints`` / ``self.graph`` once
    # the worker is past its short setup window. ``rm_keyframe`` then
    # cannot torpedo us mid-iter: we hold direct refs to the same Camera
    # objects, the snapshot edge tensors are clones, and ``fix_pose=True``
    # in setup keeps tracking the sole writer to ``video.poses[*]``.
    # ------------------------------------------------------------------

    def get_past_frames_fast(self, payload):
        """fast_mode version of ``get_past_frames``.

        Returns ``past_kfIDs`` — creation-tstamps of gaussians visible
        from the ACTIVE-EDGE window, EXCLUDING the active-edge-endpoint
        tstamps themselves. Used by the adaptive-window heuristic to
        expand ``frontend_window`` so that ``add_proximity_factors``
        can pick up long-range edges to older alive KFs whose
        gaussians the current window still observes.

        Mirrors legacy ``get_past_frames(updating_indices)`` semantics —
        renders only the active window (not every alive KF) so past_kfIDs
        contains "out-of-window but alive" tstamps that can be intersected
        with ``video.tstamp[:counter]`` to find the oldest alive slot
        still observed.
        """
        cams = payload.cam_by_uid
        # Active-edge-endpoint uids — analog of legacy's
        # ``updating_indices_window = torch.unique(edges_for_opt)``.
        active_uids = (set(payload.uid_ii) | set(payload.uid_jj)) \
            if payload.uid_ii else set()
        # Always include sel_uid (matches legacy: latest KF is always in
        # the active window).
        if payload.selected_uid is not None:
            active_uids.add(payload.selected_uid)
        if not active_uids:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        total_found_filter = None
        with torch.no_grad():
            for uid in active_uids:
                cam = cams.get(uid)
                if cam is None:
                    continue
                rp = render(cam, cam, self.gaussians,
                            self.pipeline_params, self.background,
                            use_flow=False, update_flow_conf=False,
                            train_pose=False)
                f = rp["n_found"] > 0
                if total_found_filter is None:
                    total_found_filter = f.detach().clone()
                else:
                    total_found_filter.logical_or_(f)
                del rp, f
        if total_found_filter is None:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        selected_tstamps = torch.tensor(
            [int(payload.tstamp_by_uid.get(u, -1)) for u in active_uids
             if cams.get(u) is not None],
            dtype=torch.int32, device=self.device,
        )
        midx = torch.nonzero(total_found_filter, as_tuple=True)[0]
        touched_kfIDs = torch.index_select(
            self.gaussians.unique_kfIDs, 0, midx
        ).to(torch.int32)
        del midx, total_found_filter
        touched_kfIDs = torch.unique(touched_kfIDs, sorted=False)
        is_in_selected = torch.isin(touched_kfIDs,
                                    torch.unique(selected_tstamps))
        past_kfIDs = touched_kfIDs[~is_in_selected]
        return past_kfIDs

    def mapping_full_batch(self, edge_uid_pairs, flow_active, flowconf_active,
                          inac_edge_uid_pairs, flow_inac, flowconf_inac,
                          cams, sel_uid, set_occ_aware_visibility=False,
                          flow_active_chw=None, flow_inac_chw=None):
        """Full-batch one-step optimization.

        Renders every active and inactive edge in the window, sums color +
        flow + ssim-aux + silh-aux + flow-aux losses across all of them,
        adds the once-per-pass isotropic + opacity regularizers, then
        runs a single ``loss.backward(); optimizer.step()``.

        ``flow_active_chw`` / ``flow_inac_chw`` are optional pre-permuted
        ``(E, 2, H, W)`` views provided by the caller — when supplied the
        per-iter ``flow.permute(...).contiguous()`` copy is avoided. Caller
        should compute these once per chunk before the iter loop.

        Returns the scalar total loss as a Python float so the caller
        can decide whether to stop early.
        """
        self.iteration_count += 1
        LAMBDA_FLOW = self.config["Training"]["lambda_flow"]
        total_loss = torch.zeros((), device=self.device)
        n_edges_used = 0

        # ---- Frozen-background tracking (opt-in via YAML) -------------
        # When ``fast_mode.freezing.enabled`` is True, accumulate the
        # union of n_touched>0 masks across every edge in this iter,
        # then after backward zero the gradients of gaussians that
        # haven't been seen in the last ``freezing.max_age_iters``
        # mapping iters. The render loop is unchanged; we only suppress
        # Adam updates on stale rows so their state stops contributing
        # gradient noise + step cost. OFF = no overhead.
        freezing_cfg = self.fast_mode_cfg.get("freezing", {}) or {}
        freeze_enabled = bool(freezing_cfg.get("enabled", False))
        freeze_max_age = int(freezing_cfg.get("max_age_iters", 500))
        if freeze_enabled:
            self.gaussians._freeze_tracking_enabled = True
            self.gaussians._freeze_current_iter = int(self.iteration_count)
        else:
            self.gaussians._freeze_tracking_enabled = False
        chunk_touched = None

        with torch.enable_grad():
            # ---- Active edges --------------------------------------
            if flow_active is not None and edge_uid_pairs:
                for ek, (ui, uj) in enumerate(edge_uid_pairs):
                    i_cam = cams.get(ui); j_cam = cams.get(uj)
                    if i_cam is None or j_cam is None:
                        continue
                    weight = flowconf_active[ek]
                    if flow_active_chw is not None:
                        i_cam.flow_image = flow_active_chw[ek]
                    else:
                        flow = flow_active[ek]
                        i_cam.flow_image = flow.permute(2, 0, 1).contiguous().to(
                            self.device, non_blocking=True
                        )
                    w_dev = weight.to(self.device, non_blocking=True)
                    flow_conf_sum_prev = w_dev.sum()
                    i_gt = i_cam.original_image.to(self.device, non_blocking=True)
                    render_pkg = render(
                        i_cam, j_cam, self.gaussians,
                        self.pipeline_params, self.background,
                        use_flow=True, update_flow_conf=False,
                        flow_conf=w_dev if self.flow_func == 'log-logistic' else None,
                        train_pose=False,
                        use_flowraw_grad=False if self.flow_func == 'log-logistic' else True,
                    )
                    image = render_pkg["render"]
                    flow_cost = render_pkg["flowcost"]
                    rgb_pixel_mask = (i_gt.sum(dim=0) > self.rgb_boundary_threshold
                                      ).view(*self.mask_shape)
                    Ll1 = l1_loss(image, i_gt) * rgb_pixel_mask
                    if self.config["Training"]["ssim_mask_eroded"]:
                        mask_f = rgb_pixel_mask.float().unsqueeze(0)
                        eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                        rgb_pixel_mask_e = eroded.squeeze(0).bool()
                        Lssim = (1.0 - ssim(image, i_gt)) * rgb_pixel_mask_e
                    else:
                        Lssim = (1.0 - ssim(image, i_gt)) * rgb_pixel_mask
                    loss_color = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean()
                                  + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
                    if self.flow_func == 'log-logistic':
                        loss_flow = (flow_cost * rgb_pixel_mask).sum() / (
                            flow_conf_sum_prev / FLOAT_SCALING + 1e-6
                        )
                    elif self.flow_func == 'L1':
                        gsflow = render_pkg["gsflow"]
                        loss_flow = (((gsflow - i_cam.flow_image.detach()).abs()
                                      * w_dev.detach().permute(2, 0, 1))).mean()
                    else:  # Mahalanobis
                        gsflow = render_pkg["gsflow"]
                        loss_flow = (((gsflow - i_cam.flow_image.detach()).square()
                                      * w_dev.detach().permute(2, 0, 1))).mean()
                    total_loss = total_loss + loss_color + LAMBDA_FLOW * loss_flow
                    n_edges_used += 1
                    if set_occ_aware_visibility and ui == sel_uid:
                        with torch.no_grad():
                            self.occ_aware_visibility[ui] = (
                                render_pkg["n_touched"] > 0
                            )
                    if freeze_enabled:
                        with torch.no_grad():
                            nt = render_pkg["n_touched"] > 0
                            if chunk_touched is None:
                                chunk_touched = nt.clone()
                            elif chunk_touched.shape[0] == nt.shape[0]:
                                chunk_touched |= nt

            # ---- Inactive edges ------------------------------------
            if flow_inac is not None and inac_edge_uid_pairs:
                for k, (ui, uj) in enumerate(inac_edge_uid_pairs):
                    i_cam = cams.get(ui); j_cam = cams.get(uj)
                    if i_cam is None or j_cam is None:
                        continue
                    weight = flowconf_inac[k]
                    if flow_inac_chw is not None:
                        i_cam.flow_image = flow_inac_chw[k]
                    else:
                        flow = flow_inac[k]
                        i_cam.flow_image = flow.permute(2, 0, 1).contiguous().to(
                            self.device, non_blocking=True
                        )
                    w_dev = weight.to(self.device, non_blocking=True)
                    flow_conf_sum_prev = w_dev.sum()
                    i_gt = i_cam.original_image.to(self.device, non_blocking=True)
                    render_pkg = render(
                        i_cam, j_cam, self.gaussians,
                        self.pipeline_params, self.background,
                        use_flow=True, update_flow_conf=False,
                        flow_conf=w_dev if self.flow_func == 'log-logistic' else None,
                        train_pose=False,
                        use_flowraw_grad=False if self.flow_func == 'log-logistic' else True,
                    )
                    image = render_pkg["render"]
                    flow_cost = render_pkg["flowcost"]
                    rgb_pixel_mask = (i_gt.sum(dim=0) > self.rgb_boundary_threshold
                                      ).view(*self.mask_shape)
                    Ll1 = l1_loss(image, i_gt) * rgb_pixel_mask
                    if self.config["Training"]["ssim_mask_eroded"]:
                        mask_f = rgb_pixel_mask.float().unsqueeze(0)
                        eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                        rgb_pixel_mask_e = eroded.squeeze(0).bool()
                        Lssim = (1.0 - ssim(image, i_gt)) * rgb_pixel_mask_e
                    else:
                        Lssim = (1.0 - ssim(image, i_gt)) * rgb_pixel_mask
                    loss_color = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean()
                                  + self.opt_params.lambda_dssim * Lssim.mean()) * FLOAT_SCALING
                    if self.flow_func == 'log-logistic':
                        loss_flow = (flow_cost * rgb_pixel_mask).sum() / (
                            flow_conf_sum_prev / FLOAT_SCALING + 1e-6
                        )
                    elif self.flow_func == 'L1':
                        gsflow = render_pkg["gsflow"]
                        loss_flow = (((gsflow - i_cam.flow_image.detach()).abs()
                                      * w_dev.detach().permute(2, 0, 1))).mean()
                    else:
                        gsflow = render_pkg["gsflow"]
                        loss_flow = (((gsflow - i_cam.flow_image.detach()).square()
                                      * w_dev.detach().permute(2, 0, 1))).mean()
                    total_loss = total_loss + loss_color + LAMBDA_FLOW * loss_flow
                    n_edges_used += 1
                    if freeze_enabled:
                        with torch.no_grad():
                            nt = render_pkg["n_touched"] > 0
                            if chunk_touched is None:
                                chunk_touched = nt.clone()
                            elif chunk_touched.shape[0] == nt.shape[0]:
                                chunk_touched |= nt

            if n_edges_used == 0:
                # Nothing to optimize; signal "converged" so caller exits.
                return 0.0

            # ---- Once-per-pass regularizers ------------------------
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(
                scaling - scaling.mean(dim=1).view(-1, 1)
            ).mean() * FLOAT_SCALING
            opacity = self.gaussians.get_opacity.squeeze()
            loss_opacity_elem = torch.clamp(
                -(opacity - 0.1) * torch.log(opacity + 0.01), min=0.0
            ).squeeze(-1)
            loss_opacity = loss_opacity_elem.mean() * FLOAT_SCALING
            total_loss = total_loss + isotropic_loss + 0.01 * loss_opacity

            # Average per-edge so loss magnitude is comparable across
            # chunks with different edge counts (the caller threshold
            # then has a stable meaning).
            total_loss = total_loss / max(1, n_edges_used)
            total_loss.backward()

        # ---- Frozen-gaussian gradient suppression --------------------
        # Bump last-seen for touched rows, then zero gradients on rows
        # not seen within ``freeze_max_age`` iters. Adam.step() still
        # runs but applies ~0 updates to frozen rows (m,v decay
        # geometrically). No-op when freeze_enabled is False.
        if freeze_enabled and chunk_touched is not None:
            self.gaussians.update_last_seen(chunk_touched)
            frozen_mask = self.gaussians.get_frozen_mask(freeze_max_age)
            n_frozen = 0
            if frozen_mask is not None and frozen_mask.any():
                self.gaussians.zero_grads_for_frozen(frozen_mask)
                n_frozen = int(frozen_mask.sum().item())

        with torch.no_grad():
            if self.visible_only_adam:
                # No per-edge visibility filter aggregated; fall back to
                # full optimizer step for the batch path.
                self.gaussians.optimizer.step()
            else:
                self.gaussians.optimizer.step()
            if self.keyframe_optimizers is not None:
                self.keyframe_optimizers.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.keyframe_optimizers is not None:
                self.keyframe_optimizers.zero_grad(set_to_none=True)

        return float(total_loss.detach().item())

    def _identify_insert_region_fast(self, viewpoint):
        """Same as ``identify_insert_region`` but takes a Camera ref so we
        don't touch live ``gs_viewpoints``."""
        if viewpoint is None:
            return None, None, None, None
        gt_image = viewpoint.original_image.to(self.device, non_blocking=True)
        ssim_thres = 0.5
        energy_thres_max = 0.7
        with torch.no_grad():
            render_pkg = render(viewpoint, viewpoint,
                                self.gaussians, self.pipeline_params, self.background,
                                use_flow=False, update_flow_conf=False)
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            silh  = render_pkg["silh"]
            n_found = render_pkg["n_found"]
            image_ab = image
            rgb_pixel_mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold).view(*self.mask_shape)
            if self.config["Training"]["ssim_mask_eroded"]:
                mask_f = rgb_pixel_mask.float().unsqueeze(0)
                eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                Lssim = ((1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask_eroded).detach()
            else:
                Lssim = ((1.0 - ssim(image_ab, gt_image)) * rgb_pixel_mask).detach()
            silh_mask = silh.squeeze() < energy_thres_max
            Lssim_mask = Lssim.squeeze() > ssim_thres
            insert_region = torch.logical_or(silh_mask, Lssim_mask)
        return insert_region.detach(), (n_found > 0).detach(), (silh > 0.95).detach(), depth.detach()

    def _create_new_gaussians_fast(self, viewpoint, kf_id, disps_up, insert_region,
                                found_filter, silh_filter, depth=None):
        """Cam-direct version of ``create_new_gaussians``."""
        with torch.no_grad():
            gt_img = viewpoint.original_image.to(self.device, non_blocking=True)
            valid_rgb1 = (gt_img.sum(dim=0) > self.rgb_boundary_threshold)
            disps = disps_up.to(self.device, non_blocking=True)
            initial_depth = torch.where(disps > 0, disps.reciprocal(),
                                        torch.zeros_like(disps))
            if depth is not None and silh_filter is not None:
                d = depth.squeeze().to(self.device, non_blocking=True)
                s = silh_filter.squeeze().to(self.device, non_blocking=True)
                initial_depth = torch.where(s, d, initial_depth)
            valid_rgb2 = (initial_depth > 0.0) & (initial_depth < 500.0)
            valid_rgb = valid_rgb1 & valid_rgb2
            initial_depth.masked_fill_(~valid_rgb, 0.0)
            initial_depth.masked_fill_(~insert_region.to(self.device), 0.0)
            initial_depth = initial_depth.detach().cpu().numpy()
            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=int(kf_id), init=False, scale=2.0,
                depthmap=initial_depth, found_filter=found_filter,
            )

    def _setup_keyframe_optimizers_fast(self, cam_list, fix_pose=True):
        """Cam-list version of ``setup_keyframe_optimizers``. With
        ``fix_pose=True`` we register only zero-lr exposure params (so
        ``keyframe_optimizers.step()`` is a noop). Returning a non-None
        Adam keeps the existing call site simple; alternatively we could
        leave it as None and rely on the None-checks added in
        ``mapping_one_iter``."""
        opt_params = []
        for cam in cam_list:
            if cam is None:
                continue
            if not fix_pose:
                opt_params.append({
                    "params": [cam.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(cam.uid),
                })
                opt_params.append({
                    "params": [cam.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(cam.uid),
                })
            opt_params.append({
                "params": [cam.exposure_a],
                "lr": 0.00,
                "name": "exposure_a_{}".format(cam.uid),
            })
            opt_params.append({
                "params": [cam.exposure_b],
                "lr": 0.00,
                "name": "exposure_b_{}".format(cam.uid),
            })
        self.keyframe_optimizers = torch.optim.Adam(opt_params) if opt_params else None

    def _push_gui_fast(self, payload, edge_uid_pairs, latest_cam,
                    flow_active=None, flowconf_active=None):
        """GUI push for fast_mode; uses cam refs from the payload, not
        live ``gs_viewpoints``. Renders Estimated Flow (DBA-target,
        from the captured snapshot) + GaussianFlow (from the GS map)
        for an edge incident on the latest KF, mirroring legacy.
        """
        if not self.use_gui:
            return
        # ``current_frame`` and ``kf_window`` should change ONLY when an
        # insert task pops from the mapping queue (sel_uid set), not on
        # every refinement chunk. During refinement we leave both fields
        # as None in the packet so the GUI keeps showing the last popped
        # insert's KF + its snapshot graph (sticky semantics).
        sel_uid = getattr(payload, "selected_uid", None)
        is_insert_pop = sel_uid is not None
        gui_keyframes = []
        for uid in payload.kf_uids_sorted:
            cam = payload.cam_by_uid.get(uid)
            if cam is not None:
                gui_keyframes.append(to_camera_vis(cam))
        edge_dict = None
        if is_insert_pop:
            # Payload is captured at enqueue time, so edges naturally
            # don't extend past sel_uid (sel_uid was the newest KF when
            # this snapshot was taken).
            edge_dict = {}
            for ui, uj in edge_uid_pairs:
                t_i = int(payload.tstamp_by_uid.get(ui, ui))
                t_j = int(payload.tstamp_by_uid.get(uj, uj))
                edge_dict.setdefault(t_i, []).append(t_j)

        # ---- Build Estimated Flow / GaussianFlow images for the CURRENT
        # keyframe (latest_cam). Prefer an edge with i_uid==latest_uid so
        # the rendered i-view is the current KF; if only j-side edges
        # exist, swap (i,j) so we still render at latest_cam. ----
        gtflow_vis = None
        gsflow_vis = None
        gtcolor_for_panel = None
        if latest_cam is not None and flow_active is not None and len(edge_uid_pairs) > 0:
            latest_uid = None
            for uid, cam in payload.cam_by_uid.items():
                if cam is latest_cam:
                    latest_uid = uid; break
            i_cam = j_cam = None
            flow_e = weight_e = None
            chosen = None  # ('i', edge_idx) or ('j', edge_idx)
            for k, (ui, uj) in enumerate(edge_uid_pairs):
                if ui == latest_uid:
                    chosen = ('i', k); break
            if chosen is None:
                for k, (ui, uj) in enumerate(edge_uid_pairs):
                    if uj == latest_uid:
                        chosen = ('j', k); break
            if chosen is not None:
                role, k = chosen
                ui_sel, uj_sel = edge_uid_pairs[k]
                if role == 'i':
                    i_cam = payload.cam_by_uid.get(ui_sel)
                    j_cam = payload.cam_by_uid.get(uj_sel)
                else:
                    # swap so latest_cam ends up as i
                    i_cam = payload.cam_by_uid.get(uj_sel)
                    j_cam = payload.cam_by_uid.get(ui_sel)
                flow_e = flow_active[k]
                weight_e = flowconf_active[k]
            if i_cam is not None and j_cam is not None and flow_e is not None:
                try:
                    # optical_flow_to_rgb expects (2, H, W). flow_e is (H, W, 2).
                    flow_2hw = flow_e.permute(2, 0, 1).contiguous()
                    gtflow_vis = gui_utils.optical_flow_to_rgb(flow_2hw)
                    with torch.no_grad():
                        i_cam.flow_image = flow_2hw.to(self.device)
                        render_pkg = render(
                            i_cam, j_cam, self.gaussians,
                            self.pipeline_params, self.background,
                            use_flow=True, update_flow_conf=False,
                            flow_conf=weight_e.to(self.device) if self.flow_func == 'log-logistic' else None,
                            train_pose=False,
                        )
                        gsflow_vis = gui_utils.optical_flow_to_rgb(render_pkg["gsflow"].detach())
                    gtcolor_for_panel = i_cam.original_image
                except BaseException as _ex:
                    print(f"[GUI flow build] {type(_ex).__name__}: {_ex}", flush=True)

        gui_utils.put_latest(
            self.q_main2vis,
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=(to_camera_vis(latest_cam)
                               if (is_insert_pop and latest_cam is not None)
                               else None),
                gtcolor=gtcolor_for_panel,
                gtflow=gtflow_vis,
                gsflow=gsflow_vis,
                keyframes=gui_keyframes,
                kf_window=edge_dict,
            )
        )

    def efficient_densify_and_prune_fast(self, payload, edge_uid_pairs,
                                      flow_active, flowconf_active, do_prune):
        """Lock-free densify+prune driven by the snapshot.

        Mirrors ``efficient_densify_and_prune`` but pulls all data from
        ``payload`` instead of ``self.graph`` / ``self.video``. Picks
        the last ~16 KFs (by uid → tstamp) as the densify window so
        new gaussians live near the active camera.
        """
        cams = payload.cam_by_uid
        if not cams or not edge_uid_pairs:
            return

        # Window: recent 16 uids by tstamp. The legacy heuristic uses
        # `ii_torch >= curr_idx-15` which is exactly "recent 16 KFs".
        sorted_uids = sorted(cams.keys(),
                             key=lambda u: payload.tstamp_by_uid.get(u, 0))
        window_uids = set(sorted_uids[-16:])
        # Accept either endpoint inside the window — factor graph adds
        # edges in pairs (i,j) and (j,i), but our edge filter may have
        # capped to one direction, so a pure ``ui in window`` check could
        # silently drop the densify window's edges.
        sel = [k for k, (ui, uj) in enumerate(edge_uid_pairs)
               if ui in window_uids or uj in window_uids]
        if not sel:
            return

        edges = [(edge_uid_pairs[k][0], edge_uid_pairs[k][1]) for k in sel]
        idx_t = torch.tensor(sel, dtype=torch.long, device=flow_active.device)
        flows = torch.index_select(flow_active, 0, idx_t)
        sel_w = torch.index_select(flowconf_active, 0, idx_t)
        # `flowconf_active` is already normalized in optimize_map_randomly_fast;
        # but legacy efficient_densify_and_prune renormalizes raw weight,
        # so re-normalize defensively only if log-logistic — otherwise the
        # weights pass through.
        weights = sel_w  # already normalized

        acm_list = [
            viewspace_point_tensor_acm,
            visibility_filter_acm,
            radii_acm,
            n_touched_acm,
            n_found_acm,
            error_per_gs_acm,
            error_per_gs_2_acm,
            error_per_gs_3_acm,
        ] = ({}, {}, {}, {}, {}, {}, {}, {})

        found_count = None
        with torch.enable_grad():
            for e, (ui, uj) in enumerate(edges):
                i_cam = cams.get(ui); j_cam = cams.get(uj)
                if i_cam is None or j_cam is None:
                    continue
                flow_e = flows[e].to(self.device, non_blocking=True)
                # contiguous() to break view-tie to payload.flow_ups
                i_cam.flow_image = flow_e.permute(2, 0, 1).contiguous()
                w_e = weights[e].to(self.device, non_blocking=True)
                flow_conf_sum_prev = w_e.detach().sum()
                i_gt_image = i_cam.original_image.to(self.device, non_blocking=True)
                render_pkg = render(
                    i_cam, j_cam, self.gaussians,
                    self.pipeline_params, self.background,
                    use_flow=True, update_flow_conf=False,
                    flow_conf=w_e, train_pose=False,
                )
                image          = render_pkg["render"]
                viewspace      = render_pkg["viewspace_points"]
                visibility     = render_pkg["visibility_filter"]
                radii          = render_pkg["radii"]
                silh           = render_pkg["silh"]
                gsflow         = render_pkg["gsflow"]
                flow_cost      = render_pkg["flowcost"]
                error_per_gs   = render_pkg["error_per_gs"]
                error_per_gs_2 = render_pkg["error_per_gs_2"]
                error_per_gs_3 = render_pkg["error_per_gs_3"]
                aux_image      = render_pkg["aux_image"]
                aux_image2     = render_pkg["aux_image2"]
                n_touched      = render_pkg["n_touched"]
                n_found        = render_pkg["n_found"]

                rgb_pixel_mask = (i_gt_image.sum(dim=0) > self.rgb_boundary_threshold
                                  ).view(*self.mask_shape)
                Ll1 = l1_loss(image, i_gt_image) * rgb_pixel_mask
                if self.config["Training"]["ssim_mask_eroded"]:
                    mask_f = rgb_pixel_mask.float().unsqueeze(0)
                    eroded = -F.max_pool2d(-mask_f, kernel_size=11, stride=1, padding=5)
                    rgb_pixel_mask_eroded = eroded.squeeze(0).bool()
                    Lssim = (1.0 - ssim(image, i_gt_image)) * rgb_pixel_mask_eroded
                else:
                    Lssim = (1.0 - ssim(image, i_gt_image)) * rgb_pixel_mask
                loss1 = ((1.0 - self.opt_params.lambda_dssim) * Ll1.mean()
                         + self.opt_params.lambda_dssim * Lssim.mean())
                loss_f = (flow_cost).sum() / ((flow_conf_sum_prev / FLOAT_SCALING + 1e-6))
                loss_aux  = (Lssim.detach() * aux_image).sum()
                loss_aux2 = (silh.detach() * aux_image2).sum()
                loss_aux3 = (flow_cost.detach() * render_pkg["aux_image3"]).sum()

                if found_count is None:
                    found_count = (n_found > 0).to(torch.int32).detach()
                else:
                    found_count += (n_found > 0).to(torch.int32).detach()

                data = (viewspace, visibility, radii, n_touched, n_found,
                        error_per_gs, error_per_gs_2, error_per_gs_3)
                update_acm(e, acm_list, data)
                loss = loss1 + loss_f + loss_aux + loss_aux2 + loss_aux3
                loss.backward()

        with torch.no_grad():
            if found_count is not None:
                torch.cuda.synchronize()
                total_found_filter = (found_count > 0).detach()
                # kf_uid threshold: legacy uses tstamp[curr_idx-25]; use
                # the same idea but in uid-space — 25 uids back from latest.
                latest_uid = sorted_uids[-1]
                kf_uid_now = int(payload.tstamp_by_uid.get(latest_uid, 0))
                if len(sorted_uids) > 25:
                    prune_thresh_uid = sorted_uids[-26]
                    prune_kf_id_thres = int(payload.tstamp_by_uid.get(prune_thresh_uid, -1))
                else:
                    prune_kf_id_thres = -1

                self.gaussians.densify_and_prune_by_error(
                    radii_acm, error_per_gs_acm, error_per_gs_2_acm,
                    error_per_gs_3_acm, viewspace_point_tensor_acm,
                    found_count, total_found_filter,
                    kf_uid=kf_uid_now, do_prune=do_prune,
                    prune_kf_id_thres=prune_kf_id_thres,
                )

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.keyframe_optimizers is not None:
                self.keyframe_optimizers.zero_grad(set_to_none=True)
            for d in acm_list:
                d.clear()

    def remove_inactive_gaussians_fast(self, payload, interval=1000):
        """Lock-free remove_inactive_gaussians using only the snapshot.

        Iterates all uids referenced by the payload (active + inactive
        edges). For each, renders the gaussian map and merges
        ``n_found > 0`` (kept by current views) and
        ``weights_sum > thr`` (kept by inactive views). Anything not
        seen by either is pruned.
        """
        if (int(self.iteration_count / interval) == self.remove_inactive_count):
            return
        self.remove_inactive_count += 1

        # Iterate over EVERY alive KF (cam_by_uid covers all live uids,
        # not just edge endpoints). Earlier versions used the payload's
        # active/inactive edge endpoint sets, but the inactive list is
        # capped at d_inac_max=64 in the payload builder — older KFs got
        # silently excluded → their visible gaussians were misclassified
        # as "inactive" and pruned, causing mid-run map collapse.
        if not payload.cam_by_uid:
            return

        weights_sum_thres = 1e-2
        total_retained_mask = None
        total_found_filter = None

        with torch.no_grad():
            for uid, cam in payload.cam_by_uid.items():
                if cam is None:
                    continue
                rp = render(cam, cam, self.gaussians,
                            self.pipeline_params, self.background,
                            use_flow=False, update_flow_conf=False)
                m_found = rp["n_found"] > 0
                m_retain = rp["weights_sum"] > weights_sum_thres
                total_found_filter = m_found if total_found_filter is None \
                    else torch.logical_or(total_found_filter, m_found)
                total_retained_mask = m_retain if total_retained_mask is None \
                    else torch.logical_or(total_retained_mask, m_retain)
                del rp, m_found, m_retain

        if total_found_filter is None and total_retained_mask is None:
            return
        if total_found_filter is None:
            total_found_filter = torch.zeros_like(total_retained_mask)
        if total_retained_mask is None:
            total_retained_mask = torch.zeros_like(total_found_filter)
        total_inactive_mask = torch.logical_and(~total_retained_mask, ~total_found_filter)
        print("[fast_mode] removing inactive gaussians: ",
              total_inactive_mask.sum().item())
        self.gaussians.prune_points(total_inactive_mask)

    def optimize_map_randomly_fast(self, payload, total_iters=100,
                                do_densification=False, fix_pose=True,
                                set_occ_aware_visibility=True,
                                do_split_only=False, do_prune_only=False):
        """fast_mode mapping pass. Lock-free; reads only the snapshot.

        ``payload`` is built by tracking and contains:
        - cam_by_uid: dict[uid → Camera]   (direct refs)
        - pose_by_uid: dict[uid → Tensor (7,)]  (snapshot quat — used to
              refresh cam.R/T at start; mapping is the sole writer of
              cam.R/T after that since fix_pose=True keeps tracking out)
        - tstamp_by_uid: dict[uid → float]
        - uid_ii / uid_jj: list[int]
        - flow_ups, weight_ups: Tensor clones (1, E, H, W, 2)
        - selected_uid, selected_disps_up
        - kf_uids_sorted: list[int] (display order)
        """
        cams = payload.cam_by_uid

        # Refresh cam.R/T from the enqueue-time pose snapshot. Tracking is
        # not allowed to write cam.R/T in fast_mode (no synchronize_poses_to_gs
        # is called from tracking), so this is a single-writer update.
        for uid, cam in cams.items():
            if cam is None:
                continue
            pose = payload.pose_by_uid.get(uid)
            if pose is None:
                continue
            Tcw = SE3(pose).matrix()
            cam.update_RT(Tcw[:3, :3], Tcw[:3, 3])

        if set_occ_aware_visibility:
            self.occ_aware_visibility.clear()

        # ---- Insert phase (only when an insert task) ----
        sel_uid = payload.selected_uid
        if sel_uid is not None and sel_uid in cams and payload.selected_disps_up is not None:
            sel_cam = cams[sel_uid]
            sel_kfID = int(payload.tstamp_by_uid.get(sel_uid, 0))
            insert_region, found_filter, silh_filter, depth = \
                self._identify_insert_region_fast(sel_cam)
            if insert_region is not None:
                self._create_new_gaussians_fast(
                    sel_cam, sel_kfID, payload.selected_disps_up,
                    insert_region, found_filter, silh_filter, depth,
                )

        # ---- Build active edge stack in uid space ----
        valid = [k for k, (ui, uj) in enumerate(zip(payload.uid_ii, payload.uid_jj))
                 if ui in cams and uj in cams and cams[ui] is not None and cams[uj] is not None]

        # Edge filtering for mapping speed. Two passes:
        # 1) Drop edges where BOTH endpoints fall outside the recent
        #    supervision window — those edges only constrain old gaussians
        #    that aren't being actively grown, so the render cost is wasted.
        # 2) Cap the surviving set at ``active_max_edges`` and prefer
        #    edges incident on sel_uid then by max-tstamp recency, so the
        #    chunk's most-relevant edges always make it through.
        n_recent = int(self.fast_mode_cfg.get("recent_window", 16))
        d_active_max = int(self.fast_mode_cfg.get("active_max_edges", 16))
        if valid and payload.tstamp_by_uid:
            sorted_uids_payload = sorted(
                payload.tstamp_by_uid.keys(),
                key=lambda u: payload.tstamp_by_uid.get(u, 0),
            )
            recent_uid_set = set(sorted_uids_payload[-n_recent:])
            if sel_uid is not None:
                recent_uid_set.add(sel_uid)
            valid_recent = [k for k in valid
                            if (payload.uid_ii[k] in recent_uid_set
                                or payload.uid_jj[k] in recent_uid_set)]
            if valid_recent:
                valid = valid_recent
            if len(valid) > d_active_max:
                def _edge_priority(k):
                    ui, uj = payload.uid_ii[k], payload.uid_jj[k]
                    incident = (sel_uid is not None
                                and (ui == sel_uid or uj == sel_uid))
                    recency = max(payload.tstamp_by_uid.get(ui, 0),
                                  payload.tstamp_by_uid.get(uj, 0))
                    return (-int(incident), -recency)
                valid = sorted(valid, key=_edge_priority)[:d_active_max]

        if valid:
            edge_uid_pairs = [(payload.uid_ii[k], payload.uid_jj[k]) for k in valid]
            flow_active = payload.flow_ups[:, valid].squeeze(0)
            weight_active = payload.weight_ups[:, valid].squeeze(0)
            if self.flow_func == 'log-logistic':
                flowconf_active = normalize_weights(weight_active)
            else:
                flowconf_active = weight_active
        else:
            edge_uid_pairs = []
            flow_active = None
            flowconf_active = None

        # ---- Sample inactive edges (mirrors legacy iters3 use_inactive_mapping=True).
        # Up to ``inac_max_edges`` edges, picked by unique ii (one edge per
        # unique ii), to give older KFs flow supervision so they don't drift.
        inac_edge_uid_pairs = []
        flow_inac = None
        flowconf_inac = None
        if (payload.flow_ups_inac is not None and len(payload.uid_ii_inac) > 0):
            # Inac edges are old-supervision anchors; they help drift but
            # aren't critical for new gaussian growth. Cap tighter than
            # before so chunk size shrinks and tracking-mapping ratio
            # improves.
            d_inac_max_edges = int(self.fast_mode_cfg.get("inac_max_edges", 8))
            inac_valid_idx = []
            seen_ii = set()
            for k, (ui, uj) in enumerate(zip(payload.uid_ii_inac, payload.uid_jj_inac)):
                if (ui in cams and uj in cams and cams[ui] is not None
                        and cams[uj] is not None and ui not in seen_ii):
                    seen_ii.add(ui)
                    inac_valid_idx.append(k)
                    if len(inac_valid_idx) >= d_inac_max_edges:
                        break
            if inac_valid_idx:
                inac_edge_uid_pairs = [(payload.uid_ii_inac[k], payload.uid_jj_inac[k])
                                       for k in inac_valid_idx]
                idx_t = torch.tensor(inac_valid_idx, dtype=torch.long,
                                     device=payload.flow_ups_inac.device)
                fi = payload.flow_ups_inac.index_select(1, idx_t).squeeze(0)
                wi = payload.weight_ups_inac.index_select(1, idx_t).squeeze(0)
                flow_inac = fi.to(self.device, dtype=torch.float32, non_blocking=True)
                w_inac_dev = wi.to(self.device, dtype=torch.float32, non_blocking=True)
                if self.flow_func == 'log-logistic':
                    flowconf_inac = normalize_weights(w_inac_dev)
                else:
                    flowconf_inac = w_inac_dev

        # ---- Setup optimizers (fix_pose: pose params not in optimizer) ----
        # Pool over all live cams (not just edge endpoints); KF iters
        # cover them all even when no edge connects them.
        unique_cams = [cams[u] for u in cams if cams[u] is not None]
        self._setup_keyframe_optimizers_fast(unique_cams, fix_pose=fix_pose)

        fast_cfg = self.fast_mode_cfg
        fb_cfg = fast_cfg.get("full_batch", {}) or {}
        inac_cfg = fast_cfg.get("inactive_edges", {}) or {}

        # ---- Full-batch path (default). When ``fast_mode.use_full_batch``
        # is True, we skip the per-edge SGD schedule below and instead run
        # ``mapping_full_batch`` for a small number of iterations: render
        # every active + inactive edge in the window, sum losses,
        # backward + step ONCE per iter. Densify thresholds are scaled
        # down accordingly (split at iter ``full_batch.split_iter``, prune
        # at iter ``full_batch.prune_iter``). Loop exits early if the
        # per-edge averaged loss drops below ``full_batch.loss_threshold``.
        # ----
        if bool(fast_cfg.get("use_full_batch", True)):
            max_iters = int(fb_cfg.get("max_iters", 30))
            loss_thresh = float(fb_cfg.get("loss_threshold", 0.1))
            split_iter = int(fb_cfg.get("split_iter", 5))
            prune_iter = int(fb_cfg.get("prune_iter", 12))

            unique_cams = [cams[u] for u in cams if cams[u] is not None]
            self._setup_keyframe_optimizers_fast(unique_cams, fix_pose=fix_pose)

            latest_cam = cams.get(sel_uid) if sel_uid is not None else None
            if latest_cam is None:
                latest_uid = max(payload.tstamp_by_uid,
                                 key=payload.tstamp_by_uid.get,
                                 default=None)
                latest_cam = cams.get(latest_uid) if latest_uid is not None else None

            # Early exit guard. If this chunk is supposed to fire a split
            # or prune, the loop must run at least up to that iter even
            # when the loss already dropped below ``loss_thresh`` — else
            # densify never gets a chance to trigger.
            must_run_until = 0
            if do_densification or do_split_only:
                must_run_until = max(must_run_until, split_iter + 1)
            if do_densification or do_prune_only:
                must_run_until = max(must_run_until, prune_iter + 1)

            # Pre-permute flow tensors once so each iter can use views
            # instead of repeating ``permute().contiguous()`` per edge
            # per iter. Active flow is GPU-resident; inactive is CPU
            # (fp16) so move-to-device once here too.
            flow_active_chw = None
            if flow_active is not None and edge_uid_pairs:
                flow_active_chw = (flow_active.permute(0, 3, 1, 2)
                                   .contiguous()
                                   .to(self.device, non_blocking=True))
            flow_inac_chw = None
            if flow_inac is not None and inac_edge_uid_pairs:
                flow_inac_chw = (flow_inac.permute(0, 3, 1, 2)
                                 .contiguous()
                                 .to(self.device, non_blocking=True))

            # Render inactive edges only every Nth outer iter. ~33% of edge
            # renders are inactive; throttling saves render cost with
            # minimal quality impact (inactive supervision is stale anyway).
            # 1 = every iter (legacy/baseline). N=999 ≈ iter-0 only.
            inactive_every = max(1, int(inac_cfg.get("every_n_iters", 1)))
            # Hard kill switch — when True, skip inactive renders entirely
            # (no iter ever runs them). Overrides ``every_n_iters``.
            inactive_disable = bool(inac_cfg.get("disable", False))

            done_split = False
            done_prune = False
            for it in range(max_iters):
                update_gaussian = (
                    it == split_iter
                    and (do_densification or do_split_only)
                    and not done_split
                )
                prune_gaussian = (
                    it == prune_iter
                    and (do_densification or do_prune_only)
                    and not done_prune
                )

                use_inactive_this_iter = (
                    (not inactive_disable)
                    and (it % inactive_every == 0)
                )
                if use_inactive_this_iter:
                    total_loss = self.mapping_full_batch(
                        edge_uid_pairs, flow_active, flowconf_active,
                        inac_edge_uid_pairs, flow_inac, flowconf_inac,
                        cams, sel_uid,
                        set_occ_aware_visibility=(set_occ_aware_visibility and it == 0),
                        flow_active_chw=flow_active_chw,
                        flow_inac_chw=flow_inac_chw,
                    )
                else:
                    # Skip inactive edge renders this iter.
                    total_loss = self.mapping_full_batch(
                        edge_uid_pairs, flow_active, flowconf_active,
                        [], None, None,
                        cams, sel_uid,
                        set_occ_aware_visibility=False,
                        flow_active_chw=flow_active_chw,
                        flow_inac_chw=None,
                    )

                if update_gaussian and flow_active is not None:
                    done_split = True
                    self.efficient_densify_and_prune_fast(
                        payload, edge_uid_pairs, flow_active, flowconf_active,
                        do_prune=False,
                    )
                elif prune_gaussian and flow_active is not None:
                    done_prune = True
                    self.efficient_densify_and_prune_fast(
                        payload, edge_uid_pairs, flow_active, flowconf_active,
                        do_prune=True,
                    )

                now = time.time()
                if now - self._map_iter_last_log >= 2.0:
                    dt = now - self._map_iter_counter_t0
                    dn = self.iteration_count - self._map_iter_counter_n0
                    if dt > 0:
                        self.last_mapping_fps = dn / dt
                    self._map_iter_counter_t0 = now
                    self._map_iter_counter_n0 = self.iteration_count
                    self._map_iter_last_log = now

                is_finished = (it == max_iters - 1)
                # In full-batch mode each chunk runs ~30 outer iters, far
                # fewer than the SGD path's hundreds. Tie GUI updates to
                # ``it`` (outer-iter counter) instead of the legacy
                # ``iteration_count % gui_interval`` so the GUI refreshes
                # every iter by default. Set ``full_batch.gui_every_iter``
                # > 1 to throttle if the gaussian-clone cost hurts fps.
                gui_every = max(1, int(fb_cfg.get("gui_every_iter", 1)))
                if self.use_gui and (is_finished or it % gui_every == 0):
                    self._push_gui_fast(payload, edge_uid_pairs, latest_cam,
                                     flow_active=flow_active,
                                     flowconf_active=flowconf_active)

                if it >= must_run_until and total_loss < loss_thresh:
                    break

            # ---- Adaptive window: compute min_past_seen_index so caller
            # can expand the tracker's frontend_window when current
            # gaussians link back to old KFs (legacy parity). ----
            min_past_seen_index = -1
            if self.config["Training"].get("adaptive_mapping_window", False):
                past_kfIDs = self.get_past_frames_fast(payload)
                if past_kfIDs.numel() > 0:
                    current_tstamps = self.video.tstamp[
                        :self.video.counter.value
                    ].to(torch.int32)
                    past_gsmask = torch.isin(current_tstamps, past_kfIDs)
                    if past_gsmask.sum() > 0:
                        past_indices = torch.nonzero(past_gsmask, as_tuple=False)
                        min_past_seen_index = int(past_indices.min().item())
            return min_past_seen_index

        # ---- Per-edge SGD schedule below (when use_full_batch=False) ----
        # Each edge is processed individually as a single-batch SGD step
        # (batch size = 1). Active edges incident on the new KF (sel_uid)
        # are bursted at `edge_iters_per_incident` to give sel_uid extra
        # supervision. Inactive edges anchor older KFs against drift. ----
        if not edge_uid_pairs and not inac_edge_uid_pairs:
            return
        sb_cfg = fast_cfg.get("single_batch", {}) or {}
        # Incident on selected KF (for flow anchor on new KF)
        incident_edge_idxs = []
        if sel_uid is not None and edge_uid_pairs:
            for k, (ui, uj) in enumerate(edge_uid_pairs):
                if ui == sel_uid or uj == sel_uid:
                    incident_edge_idxs.append(k)
        edge_iters_per_incident = int(sb_cfg.get("edge_iters_per_incident", 5))

        edge_schedule = []
        for ek in incident_edge_idxs:
            edge_schedule.extend([ek] * edge_iters_per_incident)
        random.shuffle(edge_schedule)

        # Inactive edges supervise older KFs that fell out of the window.
        # Lowered default 5→3 to shrink chunk wall-time.
        inac_iters_per_edge = int(sb_cfg.get("inac_iters_per_edge", 3))
        inac_schedule = []
        for k in range(len(inac_edge_uid_pairs)):
            inac_schedule.extend([k] * inac_iters_per_edge)
        random.shuffle(inac_schedule)

        full_schedule = ([("edge", e) for e in edge_schedule]
                         + [("inac", k) for k in inac_schedule])

        # GUI follows the mapping queue: latest_cam = the KF being
        # processed (selected_uid from queue). Panel images (Input
        # Color / Estimated Flow / GaussianFlow) all show this KF.
        # Refinement (no insert) falls back to highest-tstamp cam.
        latest_cam = cams.get(sel_uid) if sel_uid is not None else None
        if latest_cam is None:
            latest_uid = max(payload.tstamp_by_uid, key=payload.tstamp_by_uid.get,
                             default=None)
            latest_cam = cams.get(latest_uid) if latest_uid is not None else None

        train_num = 0
        done_split = False
        done_prune = False

        while full_schedule:
            kind, item = full_schedule.pop()
            # do_densification = legacy split+prune in one chunk (unused by
            #                    `_refinement_step` after the mod-3 rework;
            #                    kept for any external caller).
            # do_split_only   = split allowed, prune blocked.
            # do_prune_only   = prune allowed, split blocked.
            # The refinement loop now uses split-only and prune-only chunks
            # in alternation so newly-split gaussians never meet prune in
            # the same chunk.
            update_gaussian = (
                train_num > self.gaussian_update_every and
                train_num % self.gaussian_update_every > self.gaussian_update_offset
                and (do_densification or do_split_only) and not done_split
            )
            prune_gaussian = (
                (train_num % self.gaussian_prune_every) > self.gaussian_prune_offset
                and (train_num > self.gaussian_prune_every)
                and (do_densification or do_prune_only) and not done_prune
            )

            if kind == "edge" and flow_active is not None:
                ek = item
                ui, uj = edge_uid_pairs[ek]
                i_cam = cams[ui]; j_cam = cams[uj]
                flow = flow_active[ek]
                weight = flowconf_active[ek]
                self.mapping_one_iter(
                    train_num, None, flow, weight,
                    last_stage_for_edge=False, train_pose=False,
                    add_color_target_idx=False, do_densification=do_densification,
                    set_occ_aware_visibility=(set_occ_aware_visibility and ui == sel_uid),
                    i_cam=i_cam, j_cam=j_cam, occ_key=ui,
                )
            elif kind == "inac" and flow_inac is not None:
                k = item
                ui, uj = inac_edge_uid_pairs[k]
                i_cam = cams[ui]; j_cam = cams[uj]
                flow = flow_inac[k]
                weight = flowconf_inac[k]
                self.mapping_one_iter(
                    train_num, None, flow, weight,
                    last_stage_for_edge=False, train_pose=False,
                    add_color_target_idx=False, do_densification=False,
                    set_occ_aware_visibility=False,
                    i_cam=i_cam, j_cam=j_cam, occ_key=ui,
                )

            if update_gaussian or prune_gaussian:
                if flow_active is not None:
                    if update_gaussian:
                        done_split = True
                        self.efficient_densify_and_prune_fast(
                            payload, edge_uid_pairs, flow_active, flowconf_active,
                            do_prune=False,
                        )
                    elif prune_gaussian:
                        done_prune = True
                        self.efficient_densify_and_prune_fast(
                            payload, edge_uid_pairs, flow_active, flowconf_active,
                            do_prune=True,
                        )

            train_num += 1

            now = time.time()
            if now - self._map_iter_last_log >= 2.0:
                dt = now - self._map_iter_counter_t0
                dn = self.iteration_count - self._map_iter_counter_n0
                if dt > 0:
                    self.last_mapping_fps = dn / dt
                self._map_iter_counter_t0 = now
                self._map_iter_counter_n0 = self.iteration_count
                self._map_iter_last_log = now

            is_finished = (len(full_schedule) == 0)
            if self.use_gui and (is_finished or self.iteration_count % self.gui_interval == 0):
                self._push_gui_fast(payload, edge_uid_pairs, latest_cam,
                                 flow_active=flow_active,
                                 flowconf_active=flowconf_active)

        # ---- Adaptive window: compute min_past_seen_index so caller
        # can expand the tracker's frontend_window when current
        # gaussians link back to old KFs (same logic as full_batch). ----
        min_past_seen_index = -1
        if self.config["Training"].get("adaptive_mapping_window", False):
            past_kfIDs = self.get_past_frames_fast(payload)
            if past_kfIDs.numel() > 0:
                current_tstamps = self.video.tstamp[
                    :self.video.counter.value
                ].to(torch.int32)
                past_gsmask = torch.isin(current_tstamps, past_kfIDs)
                if past_gsmask.sum() > 0:
                    past_indices = torch.nonzero(past_gsmask, as_tuple=False)
                    min_past_seen_index = int(past_indices.min().item())
        return min_past_seen_index
