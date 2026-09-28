import os
import sys

# Make GFSLAM importable no matter what the working directory is.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GFSLAM'))

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:2048"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import matplotlib
matplotlib.use("Agg")

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import re

from torch.multiprocessing import Process
from gfslam import GFSLAM

import torch.nn.functional as F


def save_trajectory(gfslam, traj_full, imagedir, output, stride=1, scale_timestamps=False, t0=0, t1=-1):
    t = gfslam.video.counter.value
    tstamps = gfslam.video.tstamp[:t]
    poses_wc = lietorch.SE3(gfslam.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), gfslam.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))])[..., np.newaxis][t0:t1:stride]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    if scale_timestamps:
        ttraj_kf = np.concatenate([(tstamps_kf)*1e-9, poses_wc.cpu().numpy()], axis=1)
    else:
        ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf, fmt="%.9f")
    if traj_full is not None:
        if scale_timestamps:
            ttraj_full = np.concatenate([tstamps_full[:len(traj_full)]*1e-9, traj_full], axis=1)
        else:
            ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full, fmt="%.9f")

def save_full_traj(gfslam, traj_full, imagedir, output, stride=1, scale_timestamps=False, t0=0, t1=-1):
    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))])[..., np.newaxis][t0:t1:stride]
    if traj_full is not None:
        if scale_timestamps:
            ttraj_full = np.concatenate([tstamps_full[:len(traj_full)]*1e-9, traj_full], axis=1)
        else:
            ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full, fmt="%.9f")

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, t0, t1):
    """ image generator """
    RES = 341 * 640
    # RES = 520 * 640

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # image_list = sorted(os.listdir(imagedir))[t0:t1][::stride]
    image_list = sorted(
        # [f for f in os.listdir(imagedir) if f.endswith(".png")],
        [f for f in os.listdir(imagedir) if f.lower().endswith((".png", ".jpg"))],
        key=lambda x: float(os.path.splitext(x)[0])
    )[t0:t1][::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
            
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        is_last = False
        yield t, image[None], intrinsics, is_last


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame idx")
    parser.add_argument("--t1", default=int(1e8), type=int, help="end frame idx")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=700)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe") # 2.4
    parser.add_argument("--warmup", type=int, default=12, help="number of warmup frames") # default=8; euroc=15
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe") # default=4.0; euroc=3.0
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance") # default=16.0 # euroc=17.5
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window") #50 #30 # default=25 # euroc=20
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius") #2
    parser.add_argument("--frontend_nms", type=int, default=2, help="non-maximal supression of edges") #1
    parser.add_argument("--gsconfig_path", type=str, default="configs/mono.yaml", help="path to gs_mapper config file")

    parser.add_argument("--max_factors", type=int, default=72, help="max number of flow edges") #48
    parser.add_argument("--age_kf_num", type=int, default=6, help="max number of flow edges") #8
    parser.add_argument("--frontend_gap", type=int, default=3, help="min gap for flow edges") #2
    parser.add_argument("--frontend_window_offset", type=int, default=10, help="optimization window offset") #2

    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--output", default='results/traj', help="path to save output")
    parser.add_argument("--gt_path", type=str, default=None, help="path to ground truth trajectory for evaluation")
    parser.add_argument("--scale_timestamps", action="store_true", help="if timestamps from filename should be converted to seconds(*1e-9)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    gfslam = None

    tstamps = []
    run_start_time = time.time()
    fps_start_time = None
    fps_start_n = None
    last_log_t = None
    last_log_n = None
    pbar = tqdm(image_stream(args.imagedir, args.calib, args.stride, args.t0, args.t1))
    for n, (t, image, intrinsics, is_last) in enumerate(pbar):
        if gfslam is None:
            args.image_size = [image.shape[2], image.shape[3]]
            gfslam = GFSLAM(args)

        gfslam.track(t, image, intrinsics=intrinsics, is_last=is_last)

        # Start FPS accounting only after frontend/map initialization has
        # completely returned. The frame that triggered initialization and
        # all time spent processing it are intentionally excluded.
        now = time.time()
        if fps_start_time is None and gfslam.frontend.is_initialized:
            fps_start_time = now
            fps_start_n = n + 1
            last_log_t = now
            last_log_n = n + 1
            init_message = (
                f"[fps] initialization_complete frame={t}; "
                "total_fps measurement starts now"
            )
            pbar.set_postfix_str(init_message, refresh=False)
            print(init_message, flush=True)
        elif fps_start_time is not None and now - last_log_t >= 2.0:
            dt = now - last_log_t
            dn = (n + 1) - last_log_n
            window_fps = dn / dt if dt > 0 else 0.0
            measured_frames = (n + 1) - fps_start_n
            measured_time = now - fps_start_time
            total_fps = measured_frames / measured_time if measured_time > 0 else 0.0
            mapping_fps = getattr(gfslam.frontend.gs_mapper, "last_mapping_fps", 0.0)
            postfix = (f"[fps] frame={t} total_fps={total_fps:.2f} "
                       f"window_fps={window_fps:.2f} mapping_fps={mapping_fps:.2f}")
            pbar.set_postfix_str(postfix, refresh=False)
            print(postfix, flush=True)
            last_log_t = now
            last_log_n = n + 1

    tracking_end_time = time.time()
    stream_data = list(image_stream(args.imagedir, args.calib, args.stride, args.t0, args.t1))
    save_trajectory(gfslam, None, args.imagedir, args.output, args.stride, args.scale_timestamps, args.t0, args.t1)
    traj_est = gfslam.terminate(stream_data, save_dir=args.output)
    save_full_traj(gfslam, traj_est, args.imagedir, args.output, args.stride, args.scale_timestamps, args.t0, args.t1)
    print("Done")

    full_pipeline_time = time.time() - run_start_time
    num_images = len(stream_data) if stream_data is not None else 0
    full_pipeline_fps = (
        num_images / full_pipeline_time if full_pipeline_time > 0 else 0.0
    )
    if fps_start_time is not None:
        initialization_frames = min(fps_start_n, num_images)
        measured_frames = max(0, num_images - initialization_frames)
        measured_time = max(0.0, tracking_end_time - fps_start_time)
        total_fps = measured_frames / measured_time if measured_time > 0 else 0.0
    else:
        initialization_frames = num_images
        measured_frames = 0
        measured_time = 0.0
        total_fps = 0.0
    print(
        f"[runtime] measured_frames={measured_frames} "
        f"initialization_frames_excluded={initialization_frames} "
        f"tracking_time_sec={measured_time:.3f} total_fps={total_fps:.2f} "
        "(initialization excluded)",
        flush=True,
    )
    stats_path = os.path.join(args.output, "runtime_stats.txt")
    with open(stats_path, "a") as f:
        f.write(f"total_time_sec: {full_pipeline_time:.6f}\n")
        f.write(f"tracking_time_sec_excluding_initialization: {measured_time:.6f}\n")
        f.write(f"initialization_frames_excluded: {initialization_frames}\n")
        f.write(f"measured_frames: {measured_frames}\n")
        f.write(f"total_fps: {total_fps:.6f}\n")
        f.write(
            "full_pipeline_fps_including_initialization: "
            f"{full_pipeline_fps:.6f}\n"
        )
        f.write(
            "fps (num_images / total_time): "
            f"{full_pipeline_fps:.6f}\n"
        )
        f.write(f"num_images: {num_images}\n")
