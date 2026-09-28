import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from utils.flow_utils import vis_flow
from gs_mapper import GSMapper
from gfslam_frontend_fast import FastModeFuncs
from utils.logging_utils import print_gpu_mem

import gc
import re
######### evo eval #########
from evo.core.trajectory import PoseTrajectory3D, align_trajectory
from evo.core.sync import associate_trajectories
from evo.core.metrics import APE, PoseRelation, StatisticsType
from evo.tools import plot, file_interface
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.logging_utils import Log
############################

import os
import cv2
GREEN = "\033[92m"   # bright green
BLUE = "\033[94m"    # bright blue
RED = "\033[91m"     # bright red
RESET = "\033[0m"    # reset to default


class GFSLAMFrontend(FastModeFuncs):
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=args.max_factors, upsample=args.upsample
        )
        self.gs_mapper = GSMapper(self.video, self.graph, args)
        # Install the refinement callback used by the mapping worker (fast_mode).
        # Worker is only started after this so it can call back safely.
        self.gs_mapper.install_refine_callback(self._refinement_step)

        self.image_dir = args.imagedir
        self.stride = args.stride
        self.gt_path = args.gt_path
        self.traj_gt = None
        if self.gt_path is not None:
            self.load_gt_trajectory(self.gt_path)
        self.scale_timestamps = args.scale_timestamps
        self.out_path = args.output

        self.start_t0 = args.t0

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.iters1 = 3
        self.iters2 = 2
        self.iters3 = 3
        self.max_age = (self.iters1 + self.iters2 + (self.iters3-1)) * args.age_kf_num #20  (DROID-SLAM: 20/5 = 4, HI-SLAM2 25/6 = 4.16)
        # self.max_age = 25

        self.keyframe_removal_index = 3

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius
        self.frontend_gap = args.frontend_gap
        self.frontend_window_offset = args.frontend_window_offset

        self.depth_window = 3

        self.motion_damping = 0.0
        if hasattr(args, "motion_damping"):
            self.motion_damping = args.motion_damping

        self.removed_kf_num = 0

        # Run gc/empty_cache every N keyframes instead of every keyframe.
        # Each gc.collect cost ~130ms; running it on every KF cost ~7%
        # of the per-keyframe budget for no measurable correctness gain.
        self._gc_kf_interval = 5
        self._gc_kf_counter = 0

    def _init_next_state(self):

        with self.video.get_lock():
            # set pose / depth for next iteration
            self.video.poses[self.t1] = self.video.poses[self.t1 - 1]

            self.video.disps[self.t1] = torch.quantile(
                self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
            )
            self.video.disps_up[self.t1] = torch.quantile(
                self.video.disps_up[self.t1 - 3 : self.t1 - 1], 0.5
            )

            # damped linear velocity model
            if self.motion_damping >= 0:
                poses = SE3(self.video.poses)
                vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()
                damped_vel = self.motion_damping * vel
                next_pose = SE3.exp(damped_vel) * poses[self.t1 - 1]
                self.video.poses[self.t1] = next_pose.data

    def _update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1

        if self.gs_mapper.fast_mode_enabled:
            # fast_mode:
            #   tracking thread → DBA only (no init_pose, no GS flow update)
            #   mapping thread → reads tracking-decided pose, runs GS work
            self._update_tracking_only()
            return

        with self.video.get_lock():
            if self.graph.corr is not None:
                self.graph.rm_factors(self.graph.age > self.max_age, store=True)

            self.graph.add_proximity_factors(
                # self.t1 - 5,
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

            # Optimize last pose using GS
            print(f"{GREEN}Track initial pose{RESET}", "of {} frame".format(int(self.video.tstamp[self.video.counter.value-1])))
            init_pose_iters = 5
            itr = 0
            total_itr = 0
            while itr < init_pose_iters:
                total_itr += 1
                max_tau = self.gs_mapper.optimize_init_pose(self.t1 - 1, cur_iter=itr, limit_edges=True)
                if total_itr > 10:
                    break
                if max_tau > 1e-4 and itr == (init_pose_iters - 1):
                    continue
                if max_tau < 1e-4 and itr > 2:
                    break
                itr += 1

            # Optimize map and last pose
            print(f"\n{BLUE}Perform color + flow BA{RESET}")
            self.gs_mapper.optimize_map_randomly(insert_GS=True, selected_idx=self.t1-1,
                                                 do_densification=False, total_iters=100,
                                                 use_inactive_mapping=False, set_occ_aware_visibility=True)

            curr_idx_mask = torch.logical_or((self.graph.ii == self.t1-1), (self.graph.jj == self.t1-1))
            print(f"\n{RED}Perform 2nd-order DBA{RESET}")
            for itr in range(self.iters1):
                coords1 = self.gs_mapper.get_gsflows(torch.stack([self.graph.ii, self.graph.jj], dim=1)) + self.graph.coords0
                self.graph.update_by_gs(coords1, None, None, use_inactive=True, process_dba=True, selected_mask=curr_idx_mask)

            d = self.video.distance([self.t1-4], [self.t1-3], bidirectional=True)
            d_covis = self.video.distance_covis([self.t1-3])
            if self.gs_mapper.save_results:
                with open(os.path.join(self.gs_mapper.debug_dir, "d_covis.txt"), "a") as f:
                    f.write(f"{int(self.video.tstamp[self.t1-3])} {d_covis.item():.6f} {d.item()}\n")

            covis_thresh = 0.1
            cri1 = d.item() < self.keyframe_thresh
            cri2 = d_covis.item() < covis_thresh
            if cri1 and cri2:
                self.graph.rm_keyframe(self.t1 - 3)
                with self.video.get_lock():
                    self.video.counter.value -= 1
                    self.t1 -= 1
            else:
                for itr in range(self.iters2):
                    coords1 = self.gs_mapper.get_gsflows(torch.stack([self.graph.ii, self.graph.jj], dim=1)) + self.graph.coords0
                    self.graph.update_by_gs(coords1, None, None, use_inactive=True, process_dba=True, selected_mask=curr_idx_mask)

            print(f"{BLUE}Perform color + flow BA{RESET}", "of iteration: [{}]".format(self.gs_mapper.iteration_count))
            # curr_idx_mask = (self.graph.ii == self.t1-1)
            curr_idx_mask = torch.logical_or((self.graph.ii == self.t1-1), (self.graph.jj == self.t1-1))
            # update_mask = curr_idx_mask #self.graph.ii > (self.t1-1) - 3
            # update_mask = torch.logical_or((self.graph.ii > (self.t1-1) - 10), (self.graph.jj > (self.t1-1) - 10))
            # update_mask = self.graph.ii >= ((self.t1 -1) - 25)
            update_mask = None
            self.gs_mapper.reset_n_found()
            self._run_iters3_and_post()

            # Update max,min uids in the sliding window
            # window_indices = torch.unique(self.graph.ii)
            # self.video.set_sliding_uids(window_indices)

    def _run_iters3_and_post(self):
        """The heavy iters3 mapping loop + post-keyframe cleanup."""
        min_past_seen_index = None
        for itr in range(self.iters3):
            min_past_seen_index = self.gs_mapper.optimize_map_randomly(
                insert_GS=False, do_densification=(itr == 0),
                do_flowupdate=False, total_iters=200,
                use_inactive_mapping=True,
            )
            if itr > 1:
                self.graph.update(None, None, use_inactive=True)

        self.gs_mapper.remove_inactive_gaussians(interval=10000)

        print("Total edges num: ", self.graph.ii.shape[0])
        print("Total gaussians: ", self.gs_mapper.gaussians._xyz.shape[0])
        print("Total keyframes: ", self.video.counter.value - 1)
        print("Total removed keyframes: ", self.removed_kf_num)

        if (min_past_seen_index is not None) and min_past_seen_index >= 0:
            adaptive_window_size = self.video.counter.value - 1 - min_past_seen_index
            self.frontend_window = max(25, adaptive_window_size)
        else:
            self.frontend_window = 25
        print("Current frontend window size: ", self.frontend_window)

        # Debug keyframe images
        if self.gs_mapper.save_results:
            kf_saved = self.gs_mapper.save_kf_images(interval=5)
            if kf_saved and self.gs_mapper.iteration_count > 0:
                self.eval_kf_traj(self.gs_mapper.iteration_count)

        self._gc_kf_counter += 1
        if self._gc_kf_counter >= self._gc_kf_interval:
            self._gc_kf_counter = 0
            gc.collect()
            torch.cuda.empty_cache()

        # set pose for next iteration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, min_gap_sample=self.frontend_gap, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # initialize GS map
        self.gs_mapper.reset_n_found()
        self.gs_mapper.initialize_map()
        self.gs_mapper.prune_unstable_gaussians()

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

        # Update max,min uids in the sliding window
        window_indices = torch.unique(self.graph.ii)
        self.video.set_sliding_uids(window_indices)

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self._initialize()
            self._init_next_state()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self._update()
            # _update_tracking_only (fast_mode) handles its own next-frame
            # pose / disps prep; only run _init_next_state for the
            # synchronous path.
            if not self.gs_mapper.fast_mode_enabled:
                self._init_next_state()

    def load_gt_trajectory(self, gt_path):
        gt_timestamps = []
        gt_poses = []

        with open(gt_path, "r") as f:
            for line in f:
                if line.startswith("#") or len(line.strip()) == 0:
                    continue
                parts = line.strip().split()
                if len(parts) != 8:
                    continue
                t = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                rotmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rotmat
                T[:3, 3] = [tx, ty, tz]
                gt_timestamps.append(t)
                gt_poses.append(T)
        self.traj_gt = PoseTrajectory3D(poses_se3=gt_poses, timestamps=np.array(gt_timestamps))
        

    def eval_kf_traj(self, iterations):
        if self.image_dir is None:
            print("No path to image directory")
            return

        if self.traj_gt is None or len(self.traj_gt.poses_se3) == 0:
            print("[eval] Ground-truth trajectory unavailable; skipping trajectory evaluation")
            return
        
        t = self.video.counter.value
        tstamps = self.video.tstamp[:t].cpu().numpy().astype(int) + self.start_t0
        poses_wc = lietorch.SE3(self.video.poses[:t]).inv().data

        # tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(self.image_dir))])[..., np.newaxis][::self.stride]
        tstamps_full = np.array([
            float(re.findall(r"\d+\.\d+|\d+", x)[-1])
            for x in sorted(os.listdir(self.image_dir), key=lambda x: float(re.findall(r"\d+\.\d+|\d+", x)[-1]))
        ])[..., np.newaxis][::self.stride]
        tstamps_kf = tstamps_full[tstamps]

        if self.scale_timestamps:
            ttraj_kf = np.concatenate([(tstamps_kf)*1e-9, poses_wc.cpu().numpy()], axis=1)
        else:
            ttraj_kf = np.concatenate([(tstamps_kf), poses_wc.cpu().numpy()], axis=1)


        est_timestamps = []
        est_poses = []
        for row in ttraj_kf:
            t = row[0]
            tx, ty, tz, qx, qy, qz, qw = row[1:]
            rotmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rotmat
            T[:3, 3] = [tx, ty, tz]
            est_timestamps.append(t)
            est_poses.append(T)
        traj_est = PoseTrajectory3D(poses_se3=est_poses, timestamps=np.array(est_timestamps))

        traj_gt_matched, traj_est_matched = associate_trajectories(self.traj_gt, traj_est) # max_diff=0.05
        traj_est_aligned = align_trajectory(
            traj_est_matched,
            traj_gt_matched,
            correct_scale=True
        )
        traj_gt_aligned = traj_gt_matched

        ape_metric = APE(PoseRelation.translation_part)
        ape_metric.process_data((traj_gt_aligned, traj_est_aligned))
        ape_rmse = ape_metric.get_statistic(StatisticsType.rmse)
        ape_stats = ape_metric.get_all_statistics()

        Log(f"Iters: {iterations} RMSE ATE \[m]", ape_rmse, tag="Eval")

        if self.out_path is not None:
            path_save_dir = f'{self.out_path}/debug/trajs/'
            os.makedirs(path_save_dir, exist_ok=True)

            with open(os.path.join(path_save_dir, "ape_rmse_log.txt"), "a") as f:
                f.write(f"{iterations} {ape_rmse:.6f}\n")

            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot.PlotMode.xy)
            ax.set_title(f"ATE RMSE: {ape_rmse:.4f} m")
            plot.traj(ax, plot.PlotMode.xy, traj_gt_aligned, "--", "gray", "GT")
            plot.traj_colormap(
                ax, traj_est_aligned, ape_metric.error, plot.PlotMode.xy,
                min_map=ape_stats["min"], max_map=ape_stats["max"]
            )
            ax.legend()
            plt.savefig(os.path.join(path_save_dir, f"ape_{iterations}.png"), dpi=100)
            plt.close(fig)

        return ape_rmse
