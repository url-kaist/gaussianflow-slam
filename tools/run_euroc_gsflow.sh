#!/usr/bin/env bash
set -Eeuo pipefail

# Point DATA_PATH at your local EuRoC root. It must contain
#   <seq>/mav0/cam0/data/            RGB frames
#   gt_converted/<seq>_gt.txt        ground truth in TUM format
DATA_PATH=/data2/euroc

# Sequences to run (override from the shell with SEQUENCES="seq1 seq2").
read -r -a SEQUENCES <<< "${SEQUENCES:-V101 MH01}"

# Set USE_GUI=0 for a headless run.
USE_GUI=${USE_GUI:-1}

# SLAM mode. Use configs/mono_mapping_only.yaml for mapping-only.
GS_CONFIG=configs/mono.yaml

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
    echo "[run] $seq"
    python3 demo.py \
        --imagedir="$DATA_PATH/$seq/mav0/cam0/data" \
        --calib=calib/euroc.txt \
        --gsconfig_path="$GS_CONFIG" \
        --gt_path="$DATA_PATH/gt_converted/${seq}_gt.txt" \
        --output="results/$seq/" \
        --warmup=15 --upsample --scale_timestamps \
        "${vis_args[@]}" "$@"
done
