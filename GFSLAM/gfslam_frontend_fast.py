"""fast_mode-only entry points for GFSLAMFrontend.

Tracking-and-mapping decoupled: tracking thread runs DBA only,
mapping worker thread handles per-keyframe insert + continuous
background refinement. Tracking blocks via wait_async after each
enqueue, so mapping fully processes the KF before tracking advances.

This module exposes ``FastModeFuncs`` which is meant to be inherited by
``GFSLAMFrontend`` so all four methods share the same ``self`` and the
same per-instance attributes (graph, video, gs_mapper, t1, …).
"""

import gc

import torch


class FastModeFuncs:
    """fast_mode methods. Designed to be inherited by GFSLAMFrontend."""

    @staticmethod
    def _build_mapping_payload(graph, video, selected_uid=None,
                               include_disps_for_selected=False):
        """Build the full fast_mode mapping payload under the video lock.

        Returns an object holding direct Camera refs + cloned tensors;
        mapping can run lock-free against it. ``rm_keyframe`` after this
        point cannot break us because:
          - cam_by_uid holds direct refs (Cameras stay alive in py)
          - flow_ups / weight_ups are clones
          - poses are quaternion clones (used to refresh cam.R/T inside
            mapping; mapping is the sole writer of cam.R/T thereafter)
        """
        class _Payload:
            __slots__ = (
                "selected_uid", "uid_ii", "uid_jj",
                "flow_ups", "weight_ups",
                "uid_ii_inac", "uid_jj_inac",
                "flow_ups_inac", "weight_ups_inac",
                "cam_by_uid", "pose_by_uid", "tstamp_by_uid",
                "selected_disps_up", "kf_uids_sorted",
                "active_uid_set",
            )

        p = _Payload()
        p.selected_uid = selected_uid
        kf_uids = video.kf_uids
        cnt = video.counter.value

        # Active edges → uid space
        if graph.ii.numel() > 0:
            ii_cpu = graph.ii.cpu().tolist()
            jj_cpu = graph.jj.cpu().tolist()
            keep = [k for k, (i, j) in enumerate(zip(ii_cpu, jj_cpu))
                    if 0 <= i < cnt and 0 <= j < cnt
                    and kf_uids[i] >= 0 and kf_uids[j] >= 0]
            if len(keep) == len(ii_cpu):
                p.uid_ii = [kf_uids[i] for i in ii_cpu]
                p.uid_jj = [kf_uids[j] for j in jj_cpu]
                p.flow_ups = graph.flow_ups.detach().clone()
                p.weight_ups = graph.weight_ups.detach().clone()
            else:
                p.uid_ii = [kf_uids[ii_cpu[k]] for k in keep]
                p.uid_jj = [kf_uids[jj_cpu[k]] for k in keep]
                idx_t = torch.tensor(keep, dtype=torch.long,
                                     device=graph.flow_ups.device)
                p.flow_ups   = graph.flow_ups.index_select(1, idx_t).detach().clone()
                p.weight_ups = graph.weight_ups.index_select(1, idx_t).detach().clone()
        else:
            p.uid_ii, p.uid_jj = [], []
            p.flow_ups   = graph.flow_ups.detach().clone()
            p.weight_ups = graph.weight_ups.detach().clone()

        # Inactive edges — uid lists + flow tensors (CPU fp16).
        # We sample to bound clone size on long sequences.
        inac_max = 64
        if graph.ii_inac.numel() > 0:
            ii_inac_cpu = graph.ii_inac.cpu().tolist()
            jj_inac_cpu = graph.jj_inac.cpu().tolist()
            keep_i = [k for k, (i, j) in enumerate(zip(ii_inac_cpu, jj_inac_cpu))
                      if 0 <= i < cnt and 0 <= j < cnt
                      and kf_uids[i] >= 0 and kf_uids[j] >= 0]
            if len(keep_i) > inac_max:
                keep_i = keep_i[-inac_max:]
            p.uid_ii_inac = [kf_uids[ii_inac_cpu[k]] for k in keep_i]
            p.uid_jj_inac = [kf_uids[jj_inac_cpu[k]] for k in keep_i]
            if keep_i:
                idx_t = torch.tensor(keep_i, dtype=torch.long,
                                     device=graph.flow_ups_inac.device)
                p.flow_ups_inac   = graph.flow_ups_inac.index_select(1, idx_t).detach().clone()
                p.weight_ups_inac = graph.weight_ups_inac.index_select(1, idx_t).detach().clone()
            else:
                p.flow_ups_inac = None
                p.weight_ups_inac = None
        else:
            p.uid_ii_inac, p.uid_jj_inac = [], []
            p.flow_ups_inac = None
            p.weight_ups_inac = None

        # Cam refs / pose / tstamp snapshot — include EVERY live KF
        # (any uid in uid_to_slot) so GUI can render the full trajectory
        # of frustums and refinement KF iters can reach older KFs too.
        # Camera objects are shared with video.gs_viewpoints (no GPU
        # copy), so this is just a dict of pointers — bounded by #
        # alive KFs.
        all_uids = set(video.uid_to_slot.keys())
        if selected_uid is not None:
            all_uids.add(selected_uid)
        p.cam_by_uid = {}
        p.pose_by_uid = {}
        p.tstamp_by_uid = {}
        # Active set = endpoints of active edges + selected_uid. Used by
        # the schedule to focus KF iters on the recent window.
        p.active_uid_set = set(p.uid_ii) | set(p.uid_jj)
        if selected_uid is not None:
            p.active_uid_set.add(selected_uid)
        for uid in all_uids:
            slot = video.uid_to_slot.get(uid)
            if slot is None or slot >= cnt:
                continue
            cam = video.gs_viewpoints[slot]
            if cam is None:
                continue
            p.cam_by_uid[uid] = cam
            p.pose_by_uid[uid] = video.poses[slot].detach().clone()
            p.tstamp_by_uid[uid] = float(video.tstamp[slot].item())

        # Disps clone for the selected KF (needed by create_new_gaussians).
        if include_disps_for_selected and selected_uid is not None:
            slot = video.uid_to_slot.get(selected_uid)
            if slot is not None and slot < cnt:
                p.selected_disps_up = video.disps_up[slot].detach().clone()
            else:
                p.selected_disps_up = None
        else:
            p.selected_disps_up = None

        # Display order for GUI.
        p.kf_uids_sorted = sorted(p.cam_by_uid.keys(),
                                  key=lambda u: p.tstamp_by_uid.get(u, 0))
        return p

    def _update_tracking_only(self):
        """DBA-only update on the tracking thread, then enqueue mapping.

        Build a snapshot payload before releasing the lock and enqueue
        it. Mapping runs lock-free against the payload.
        """
        with self.video.get_lock():
            if self.graph.corr is not None:
                self.graph.rm_factors(self.graph.age > self.max_age, store=True)

            self.graph.add_proximity_factors(
                self.t1 - self.frontend_window_offset,
                max(self.t1 - self.frontend_window, 0),
                rad=self.frontend_radius,
                nms=self.frontend_nms,
                thresh=self.frontend_thresh,
                min_gap_sample=self.frontend_gap,
                beta=self.beta,
                remove=True,
            )

            self.video.disps[self.t1 - 1] = torch.where(
                self.video.disps_sens[self.t1 - 1] > 0,
                self.video.disps_sens[self.t1 - 1],
                self.video.disps[self.t1 - 1],
            )

            for _ in range(self.iters1):
                self.graph.update(None, None, use_inactive=True)

            d = self.video.distance([self.t1 - 4], [self.t1 - 3], bidirectional=True)
            d_covis = self.video.distance_covis([self.t1 - 3])
            cri1 = d.item() < self.keyframe_thresh
            cri2 = d_covis.item() < 0.1
            removed_uid = None
            if cri1 and cri2:
                # Capture uid before shift (rm_keyframe will retire it).
                removed_slot = self.t1 - 3
                if 0 <= removed_slot < len(self.video.kf_uids):
                    removed_uid = self.video.kf_uids[removed_slot]
                self.graph.rm_keyframe(removed_slot)
                self.video.counter.value -= 1
                self.t1 -= 1
            else:
                for _ in range(self.iters2):
                    self.graph.update(None, None, use_inactive=True)

            # Just capture the selected uid; the worker will build the
            # payload at processing time (briefly under lock) so that the
            # ~35 MB GPU clones don't accumulate while waiting in queue.
            kf_slot = self.t1 - 1
            selected_uid = self.video.kf_uids[kf_slot] if 0 <= kf_slot < len(self.video.kf_uids) else None

            # Pose / disps prep for the next foreground frame.
            self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
            self.video.disps[self.t1] = torch.quantile(
                self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
            )
            self.video.dirty[self.graph.ii.min() : self.t1] = True

        # Lock released. Queue holds tiny tuples (uid, removed_uid).
        self.gs_mapper.enqueue_mapping_task(self._mapping_task, selected_uid, removed_uid)
        # Block tracking until the worker finishes insert + forced
        # refinement chunks for this KF. Same threading layout as a
        # pure-async run, but no payload/pose races.
        self.gs_mapper.wait_async()

    def _mapping_task(self, selected_uid, removed_uid):
        """Insert task — runs on the mapping worker thread.

        Builds the payload at *processing time* (briefly under the video
        lock) instead of carrying a 35 MB GPU clone in the queue. This
        keeps queue items tiny (~16 B per task) and prevents payload
        accumulation OOMs on long sequences.

        If ``selected_uid`` was retired between enqueue and now, we
        skip — its KF no longer exists.
        """
        if selected_uid is None:
            return
        with self.video.get_lock():
            if self.video.uid_to_slot.get(selected_uid) is None:
                return
            payload = self._build_mapping_payload(
                self.graph, self.video,
                selected_uid=selected_uid,
                include_disps_for_selected=True,
            )
        # Insert task = NO densify (matches legacy initial call which
        # is do_densification=False). Newly inserted gaussians need to
        # settle for ~250 iters before split/prune. Densify happens in
        # the next refinement chunk (split-only) and the chunk after
        # (prune-only).
        min_past_seen_index = self.gs_mapper.optimize_map_randomly_fast(
            payload,
            total_iters=100,
            do_densification=False,
            do_split_only=False,
            fix_pose=True,
            set_occ_aware_visibility=True,
        )
        # Periodic KF eval (every N KFs by counter.value) — prints
        # ``kf mean psnr: ...`` to terminal so we can monitor render
        # quality during the run, not just at terminate. Internally
        # gated by ``save_kf_images``'s counter so we can call it on
        # every insert without it actually rendering every time.
        # When the eval fires, also run ATE evaluation on the current
        # KF trajectory (legacy parity).
        if self.gs_mapper.save_results:
            kf_saved = self.gs_mapper.save_kf_images(interval=10)
            if kf_saved and self.gs_mapper.iteration_count > 0:
                self.eval_kf_traj(self.gs_mapper.iteration_count)
        # Adaptive window (legacy parity): expand frontend_window when
        # current gaussians link back to old KFs so the tracker's
        # add_proximity_factors can pull long-range edges (loop closure).
        # Reading int from worker thread is GIL-safe; tracking thread
        # picks up the new value on its next _update_tracking_only.
        prev_fw = self.frontend_window
        if (min_past_seen_index is not None and min_past_seen_index >= 0):
            adaptive_window_size = self.video.counter.value - 1 - min_past_seen_index
            self.frontend_window = max(25, adaptive_window_size)
        else:
            self.frontend_window = 25
        if self.frontend_window != prev_fw:
            print(f"[adaptive] frontend_window {prev_fw} -> "
                  f"{self.frontend_window} (min_past={min_past_seen_index}, "
                  f"counter={self.video.counter.value})", flush=True)
        # Drop the payload reference promptly so its GPU clones can free
        # before the next task — without this, the local var lives until
        # the next task pop and we'd hold ~35 MB extra.
        del payload
        self._gc_kf_counter += 1
        if self._gc_kf_counter >= self._gc_kf_interval:
            self._gc_kf_counter = 0
            gc.collect()
            torch.cuda.empty_cache()

    def _refinement_step(self):
        """Refinement pass — fast_mode, lock-free.

        We capture a fresh payload at the start (briefly under lock) and
        run the gaussian-only pass on it. The captured snapshot is
        immune to subsequent tracking mutations, so this can run while
        tracking adds/removes KFs without race.
        """
        if not self.is_initialized:
            return
        budget = self.gs_mapper._refine_budget
        if budget <= 0:
            return

        self.gs_mapper._refine_pass_count += 1
        # Densify schedule — split and prune are NEVER in the same chunk
        # so newly-split tiny gaussians get a full chunk (~100 iters) of
        # supervision before the next prune. Mod 2 cycle (alternating):
        #   mod 1 → split-only
        #   mod 0 → prune-only
        # 1:1 split:prune ratio. Under heavy backpressure where only one
        # refinement chunk fires per insert, this gives KF-by-KF
        # alternation (split, prune, split, prune…), which keeps the
        # gaussian count from drifting upward as fast as the legacy 2:1
        # ratio did.
        mod2 = self.gs_mapper._refine_pass_count % 2
        do_split_only = (mod2 == 1) and budget >= 100
        do_prune_only = (mod2 == 0) and budget >= 100
        do_densification = False
        chunk_iters = 100 if (do_split_only or do_prune_only) else min(100, budget)

        # Brief lock window: build a fresh payload over the current
        # window. After this we operate on snap clones and Camera refs.
        with self.video.get_lock():
            if self.graph.ii is None or self.graph.ii.numel() == 0:
                return
            payload = self._build_mapping_payload(
                self.graph, self.video,
                selected_uid=None, include_disps_for_selected=False,
            )

        # Lock released. Heavy gaussian-only work runs concurrent with tracking.
        self.gs_mapper.optimize_map_randomly_fast(
            payload,
            total_iters=chunk_iters,
            do_densification=do_densification,
            do_split_only=do_split_only,
            do_prune_only=do_prune_only,
            fix_pose=True,
            set_occ_aware_visibility=False,
        )
        self.gs_mapper._refine_budget -= chunk_iters

        # remove_inactive_gaussians_fast iterates ALL alive KFs (cam_by_uid)
        # instead of the payload's capped edge endpoint sets, so it
        # doesn't misclassify gaussians visible from older KFs.
        # Interval is hardcoded to 1000 (fires every ~1000 mapping iters
        # ≈ ~10-30 KFs depending on chunk cadence).
        self.gs_mapper.remove_inactive_gaussians_fast(payload, interval=1000)
        self._gc_kf_counter += 1
        if self._gc_kf_counter >= self._gc_kf_interval:
            self._gc_kf_counter = 0
            gc.collect()
            torch.cuda.empty_cache()
