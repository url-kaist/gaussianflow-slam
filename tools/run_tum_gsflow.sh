#!/usr/bin/env bash
set -Eeuo pipefail

# Point DATA_PATH at your local TUM-RGBD root. It must contain
#   <seq>/rgb_modified/        RGB frames named so they sort in capture order
#   <seq>/groundtruth.txt      ground truth in TUM format
DATA_PATH=/data2/tum

# Sequences to run (override from the shell with SEQUENCES="seq1 seq2").
read -r -a SEQUENCES <<< "${SEQUENCES:-rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg2_xyz rgbd_dataset_freiburg3_long_office_household}"

# Set USE_GUI=0 for a headless run.
USE_GUI=${USE_GUI:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

vis_args=()
if [ "$USE_GUI" = "1" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=""
else
    export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
    vis_args=(--disable_vis)
fi

for seq in "${SEQUENCES[@]}"; do
    # freiburg1/2 need the distortion model; freiburg3 frames are undistorted.
    case "$seq" in
        *freiburg1*) calib=calib/tum1.txt; config=configs/tum_distort.yaml ;;
        *freiburg2*) calib=calib/tum2.txt; config=configs/tum_distort.yaml ;;
        *freiburg3*) calib=calib/tum3.txt; config=configs/tum_undistort.yaml ;;
        *) echo "[!] unknown sequence: $seq" >&2; continue ;;
    esac

    echo "[run] $seq ($calib)"
    python3 demo.py \
        --imagedir="$DATA_PATH/$seq/rgb_modified" \
        --calib="$calib" \
        --gsconfig_path="$config" \
        --gt_path="$DATA_PATH/$seq/groundtruth.txt" \
        --output="results/$seq/" \
        --warmup=12 --upsample --scale_timestamps \
        --frontend_gap=2 --frontend_nms=1 --age_kf_num=12 --keyframe_thresh=4.0 \
        "${vis_args[@]}" "$@"
done
