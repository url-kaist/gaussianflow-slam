import torch
import lietorch
import numpy as np

from gfslam_net import GFSLAMNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from gfslam_frontend import GFSLAMFrontend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

import os
import time
import atexit

class GFSLAM:
    def __init__(self, args):
        super(GFSLAM, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis
        self.images = {}

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = GFSLAMFrontend(self.net, self.video, self.args)
        self._shutdown_complete = False
        atexit.register(self.shutdown)

        self.stats_path = os.path.join(args.output, "runtime_stats.txt")
        
        # backend process

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        self.net = GFSLAMNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None, is_last=False):
        """ main thread - update map """

        with torch.no_grad():
            self.images[tstamp] = image

            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics, is_initialized=self.frontend.is_initialized, is_last=is_last)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream_data=None, save_dir=None):
        try:
            return self._terminate(stream_data=stream_data, save_dir=save_dir)
        finally:
            self.shutdown()

    def shutdown(self):
        """Stop background mapping and GUI workers. Safe to call repeatedly."""
        if getattr(self, "_shutdown_complete", False):
            return

        frontend = getattr(self, "frontend", None)
        mapper = getattr(frontend, "gs_mapper", None)
        if mapper is not None:
            try:
                mapper.shutdown_mapping_worker()
            except Exception:
                pass
            try:
                mapper.shutdown_gui()
            except Exception:
                pass
        self._shutdown_complete = True

    def _terminate(self, stream_data=None, save_dir=None):
        """ terminate the visualization process, return poses [t, q] """

        # Drain any pending async mapping work (fast_mode) before we
        # touch trajectories, gaussians or write outputs.
        if hasattr(self.frontend, "gs_mapper") and \
                self.frontend.gs_mapper.fast_mode_enabled:
            self.frontend.gs_mapper.wait_async()
            # Tear down the persistent mapping worker so the
            # process can exit cleanly.
            self.frontend.gs_mapper.shutdown_mapping_worker()

        torch.cuda.empty_cache()
        print("#" * 32)
        # self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        # self.backend(12)

        self.frontend.eval_kf_traj(self.frontend.gs_mapper.iteration_count)
        self.frontend.video.skip_exposure_copy = True

        start_time = time.time()
        deltas = np.add.accumulate(self.filterx.deltas)
        camera_trajectory, camera_dips = self.traj_filler(iter(stream_data))
        del self.filterx
        traj_filler_time = time.time() - start_time
        with open(self.stats_path, 'a') as f:
            f.write(f"Trajectory filler time: {traj_filler_time:.4f} sec\n")

        mapper = self.frontend.gs_mapper
        gaussian_count = int(mapper.gaussians.get_xyz.shape[0])
        refinement_time = 0.0
        if gaussian_count == 0:
            print(
                "[mapping] No Gaussians were created; skipping rendering, "
                "color refinement, and PLY export.",
                flush=True,
            )
        else:
            if mapper.save_results:
                mapper.save_kf_images(interval=5, force_save=True)
                # self.frontend.gs_mapper.reset_nonfound_gaussians(check_all_kfs=True, interval=3000, force_reset=True) #interval=2000
                mapper.save_rendering(iter(stream_data), traj=camera_trajectory.matrix().data,
                                    kf_indices=self.video.tstamp[:self.video.counter.value].to(device='cpu'),
                                    save_dir=save_dir, iteration="before_opt")
            start_time = time.time()
            mapper.color_refinement(use_flow=False)
            refinement_time = time.time() - start_time
            mapper.save_rendering(iter(stream_data), traj=camera_trajectory.matrix().data,
                                kf_indices=self.video.tstamp[:self.video.counter.value].to(device='cpu'),
                                save_dir=save_dir, iteration="after_opt")

        with open(self.stats_path, 'a') as f:
            f.write(f"Color refinement time: {refinement_time:.4f} sec\n")

        if mapper.save_results:
            # self.frontend.gs_mapper.save_rendering(iter(stream_data), traj=camera_trajectory.matrix().data, 
            #                                     kf_indices=self.video.tstamp[:self.video.counter.value].to(device='cpu'), 
            #                                     save_dir=save_dir, iteration="after_opt")
            mapper.release_video_writers()
            if gaussian_count > 0:
                mapper.gaussians.save_ply(save_dir)

        with open(self.stats_path, 'a') as f:
            f.write(f"Total keyframes: {self.frontend.video.counter.value - 1}\n")
            f.write(f"Total Gaussians: {self.frontend.gs_mapper.gaussians._xyz.shape[0]}\n")

        return camera_trajectory.inv().data.cpu().numpy()
