#!/usr/bin/env bash
set -Eeuo pipefail

# Batch runner for the representative EuRoC and TUM-RGBD sequences used by
# this repository. The mode can be selected explicitly:
#
#   mapping_only: DROID owns pose; 3DGS mapping keeps pose fixed.
#   slam:         synchronous GaussianFlow pose updates + 3DGS mapping.
#   both:         run mapping_only first, then slam.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  tools/run_gaussianflow_modes.sh [mapping_only|slam|both] [sequence ...] [demo.py options]
  tools/run_gaussianflow_mapping_only.sh [sequence ...] [demo.py options]
  tools/run_gaussianflow_slam.sh [sequence ...] [demo.py options]
  tools/run_gaussianflow_mapping_only_nogui.sh [sequence ...] [demo.py options]
  tools/run_gaussianflow_slam_nogui.sh [sequence ...] [demo.py options]

Defaults:
  mode       both
  sequences  V101 MH01 fr1/desk fr2 fr3

TUM aliases:
  fr1/desk -> rgbd_dataset_freiburg1_desk
  fr2      -> rgbd_dataset_freiburg2_xyz
  fr3      -> rgbd_dataset_freiburg3_long_office_household

Runner options (place before demo.py options):
  --no-gui, --headless  Disable the Open3D GUI
  --gui                 Force-enable the Open3D GUI

Useful environment variables:
  SEQUENCES    Space-separated aliases/full sequence names
  EUROC_PATH   EuRoC root (default: /data2/euroc)
  TUM_PATH     TUM-RGBD root (default: /data2/tum)
  OUTPUT_ROOT  Result root (default: <project>/results)
  T0, T1       Frame range; an empty T1 means the full sequence
  STRIDE       Input frame stride (default: 1)
  REPEATS      Number of runs per mode/sequence (default: 1)
  RUN_TAG      Optional output tag override
  PYTHON_BIN   Python executable (default: python3)
  DRY_RUN      Set to 1 to validate inputs and print commands only
  USE_GUI      Show the Open3D GUI by default; set to 0 for headless mode

Docker (opt-in; set GFSLAM_USE_DOCKER=1):
  GFSLAM_CONTAINER       Container name (default: gsflow_setup)
  GFSLAM_DOCKER_IMAGE    Image used when creating the container
                         (default: dongukseo/gs_test:cuda12.8)
  GFSLAM_CONTAINER_PROJECT  Project path in the container
                         (default: /workspace/<project directory name>)
  GFSLAM_USE_DOCKER=1    Run inside the container instead of the current shell

Examples:
  tools/run_gaussianflow_mapping_only.sh V101
  tools/run_gaussianflow_slam.sh MH01 fr1/desk fr3
  tools/run_gaussianflow_mapping_only_nogui.sh V101 fr2
  tools/run_gaussianflow_slam_nogui.sh MH01 fr3
  tools/run_gaussianflow_modes.sh both V101 fr2 --no-gui
  tools/run_gaussianflow_modes.sh both V101 fr2
  SEQUENCES='V101 fr1/desk' T1=600 DRY_RUN=1 \
    tools/run_gaussianflow_modes.sh both

Runner options listed above are consumed by this script. The first other
option beginning with '--' and everything after it are appended to every
demo.py command. An explicit '--' delimiter is also accepted.
EOF
}

die() {
    echo "[error] $*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

launch_in_docker() {
    local container_name container_image container_project container_running
    local env_name env_value data_root
    local -a create_args exec_env_args tty_args

    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    docker info >/dev/null 2>&1 || die "cannot connect to the Docker daemon"

    container_name="${GFSLAM_CONTAINER:-gsflow_setup}"
    container_image="${GFSLAM_DOCKER_IMAGE:-dongukseo/gs_test:cuda12.8}"
    container_project="${GFSLAM_CONTAINER_PROJECT:-/workspace/$(basename "$PROJECT_ROOT")}"

    if ! docker inspect --type container "$container_name" >/dev/null 2>&1; then
        echo "[docker] creating container '$container_name' from $container_image"
        create_args=(
            run -d
            --name "$container_name"
            --gpus all
            --ipc=host
            -w "$container_project"
            -v "$PROJECT_ROOT:$container_project"
        )

        if [[ -d /tmp/.X11-unix ]]; then
            create_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
        fi

        # Dataset roots are mounted at identical paths so the same command
        # line works both on the host and in the container.
        for data_root in "${EUROC_PATH:-/data2/euroc}" "${TUM_PATH:-/data2/tum}"; do
            if [[ "$data_root" == /* && -d "$data_root" ]]; then
                create_args+=(-v "$data_root:$data_root")
            fi
        done

        docker "${create_args[@]}" "$container_image" sleep infinity >/dev/null \
            || die "failed to create Docker container '$container_name'"
    fi

    container_running="$(docker inspect --type container -f '{{.State.Running}}' "$container_name")"
    if [[ "$container_running" != "true" ]]; then
        echo "[docker] starting container '$container_name'"
        docker start "$container_name" >/dev/null \
            || die "failed to start Docker container '$container_name'"
    fi

    docker exec "$container_name" test -d "$container_project" \
        || die "project is not visible at $container_project in '$container_name'"

    exec_env_args=(
        -e GFSLAM_IN_DOCKER=1
        -e "DISPLAY=${DISPLAY:-:0}"
        -e QT_X11_NO_MITSHM=1
    )
    for env_name in \
        MODE SEQUENCES EUROC_PATH TUM_PATH OUTPUT_ROOT T0 T1 STRIDE REPEATS \
        RUN_TAG DRY_RUN USE_GUI EUROC_WARMUP TUM_WARMUP PYTHON_BIN \
        CUDA_VISIBLE_DEVICES MPLBACKEND PYTORCH_CUDA_ALLOC_CONF; do
        if [[ -v $env_name ]]; then
            env_value="${!env_name}"
            if [[ "$env_name" == "OUTPUT_ROOT" && "$env_value" == "$PROJECT_ROOT"* ]]; then
                env_value="$container_project${env_value#"$PROJECT_ROOT"}"
            fi
            exec_env_args+=(-e "$env_name=$env_value")
        fi
    done

    tty_args=()
    if [[ -t 0 && -t 1 ]]; then
        tty_args=(-it)
    fi

    echo "[docker] container=$container_name project=$container_project"
    exec docker exec "${tty_args[@]}" "${exec_env_args[@]}" \
        -w "$container_project" "$container_name" \
        bash "$container_project/tools/run_gaussianflow_modes.sh" "$@"
}

# Help does not need to enter Docker, including wrapper calls such as
# run_gaussianflow_slam.sh --help.
for help_arg in "$@"; do
    case "$help_arg" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

docker_enabled="${GFSLAM_USE_DOCKER:-0}"
case "${docker_enabled,,}" in
    0|1|false|true|no|yes|off|on) ;;
    *) die "GFSLAM_USE_DOCKER must be a boolean" ;;
esac

if [[ "${GFSLAM_IN_DOCKER:-0}" != "1" && ! -f /.dockerenv ]] \
        && is_true "$docker_enabled"; then
    launch_in_docker "$@"
fi

selected_mode="${MODE:-both}"
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    mapping_only|mapping-only|map)
        selected_mode="mapping_only"
        shift
        ;;
    slam|pose|full)
        selected_mode="slam"
        shift
        ;;
    both|all)
        selected_mode="both"
        shift
        ;;
    ""|--*)
        ;;
    *)
        # No mode was given; treat the first positional value as a sequence
        # and keep the default mode (both).
        ;;
esac

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

case "$selected_mode" in
    mapping_only|mapping-only|map)
        run_modes=(mapping_only)
        ;;
    slam|pose|full)
        run_modes=(slam)
        ;;
    both|all)
        run_modes=(mapping_only slam)
        ;;
    *)
        die "invalid MODE '$selected_mode' (expected mapping_only, slam, or both)"
        ;;
esac

# Positional values after the mode are sequence aliases. Once a demo.py
# option (or an explicit '--') is encountered, the rest is forwarded as-is.
cli_sequences=()
demo_args=()
while (($# > 0)); do
    case "$1" in
        --no-gui|--headless)
            USE_GUI=0
            shift
            ;;
        --gui)
            USE_GUI=1
            shift
            ;;
        --sequences)
            (($# >= 2)) || die "--sequences requires a value"
            sequence_spec="${2//,/ }"
            read -r -a parsed_sequences <<< "$sequence_spec"
            cli_sequences+=("${parsed_sequences[@]}")
            shift 2
            ;;
        --sequences=*)
            sequence_spec="${1#*=}"
            sequence_spec="${sequence_spec//,/ }"
            read -r -a parsed_sequences <<< "$sequence_spec"
            cli_sequences+=("${parsed_sequences[@]}")
            shift
            ;;
        --)
            shift
            demo_args+=("$@")
            break
            ;;
        --*)
            demo_args+=("$@")
            break
            ;;
        *)
            sequence_spec="${1//,/ }"
            read -r -a parsed_sequences <<< "$sequence_spec"
            cli_sequences+=("${parsed_sequences[@]}")
            shift
            ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
EUROC_PATH="${EUROC_PATH:-/data2/euroc}"
TUM_PATH="${TUM_PATH:-/data2/tum}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results}"
SEQUENCES="${SEQUENCES:-V101 MH01 fr1/desk fr2 fr3}"
T0="${T0:-0}"
T1="${T1:-}"
STRIDE="${STRIDE:-1}"
REPEATS="${REPEATS:-1}"
RUN_TAG="${RUN_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
USE_GUI="${USE_GUI:-1}"
EUROC_WARMUP="${EUROC_WARMUP:-15}"
TUM_WARMUP="${TUM_WARMUP:-12}"

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
[[ "$T0" =~ ^[0-9]+$ ]] || die "T0 must be a non-negative integer"
[[ -z "$T1" || "$T1" =~ ^[0-9]+$ ]] || die "T1 must be empty or a non-negative integer"
[[ "$STRIDE" =~ ^[1-9][0-9]*$ ]] || die "STRIDE must be a positive integer"
case "${DRY_RUN,,}" in
    0|1|false|true|no|yes|off|on) ;;
    *) die "DRY_RUN must be a boolean (0/1, false/true, no/yes, or off/on)" ;;
esac
case "${USE_GUI,,}" in
    0|1|false|true|no|yes|off|on) ;;
    *) die "USE_GUI must be a boolean (0/1, false/true, no/yes, or off/on)" ;;
esac

if ((${#cli_sequences[@]} > 0)); then
    requested_sequences=("${cli_sequences[@]}")
else
    sequence_spec="${SEQUENCES//,/ }"
    read -r -a requested_sequences <<< "$sequence_spec"
fi
((${#requested_sequences[@]} > 0)) || die "SEQUENCES is empty"

export MPLBACKEND="${MPLBACKEND:-Agg}"
if [[ ! -v PYTORCH_CUDA_ALLOC_CONF ]]; then
    if is_true "$USE_GUI"; then
        export PYTORCH_CUDA_ALLOC_CONF=""
    else
        export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
    fi
fi
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/GFSLAM:$PROJECT_ROOT/thirdparty/diff-gaussian-rasterization:$PROJECT_ROOT/thirdparty/simple-knn/build/lib.linux-x86_64-3.10:$PROJECT_ROOT/thirdparty/lietorch:$PROJECT_ROOT/thirdparty/lietorch/build/lib.linux-x86_64-3.10${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
if command -v evo_config >/dev/null 2>&1; then
    evo_config set plot_backend Agg >/dev/null 2>&1 \
        || die "failed to configure evo for headless rendering"
fi

runtime_summary="$("$PYTHON_BIN" -W ignore::FutureWarning -c '
import matplotlib
import numpy
import torch
import demo

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the execution environment")

print(
    "[env] python={} numpy={} matplotlib={} torch={} cuda={}".format(
        __import__("sys").executable,
        numpy.__version__,
        matplotlib.__version__,
        torch.__version__,
        torch.cuda.get_device_name(0),
    )
)
' 2>&1)" || {
    echo "$runtime_summary" >&2
    die "GaussianFlow runtime preflight failed"
}
echo "$runtime_summary"

range_args=("--t0=$T0")
range_tag="t${T0}-full"
if [[ -n "$T1" ]]; then
    ((T1 > T0)) || die "T1 must be greater than T0"
    range_args+=("--t1=$T1")
    range_tag="t${T0}-${T1}"
fi

visualization_args=()
if ! is_true "$USE_GUI"; then
    visualization_args=(--disable_vis)
fi

resolve_sequence() {
    local requested="$1"

    case "$requested" in
        V101|v101|V1_01_easy|v1_01_easy)
            sequence_id="V101"
            dataset_kind="euroc"
            ;;
        MH01|mh01|MH_01_easy|mh_01_easy)
            sequence_id="MH01"
            dataset_kind="euroc"
            ;;
        fr1/desk|fr1_desk|tum_fr1_desk|rgbd_dataset_freiburg1_desk)
            sequence_id="rgbd_dataset_freiburg1_desk"
            dataset_kind="tum_distort"
            ;;
        fr2|fr2/xyz|fr2_xyz|tum_fr2_xyz|rgbd_dataset_freiburg2_xyz)
            sequence_id="rgbd_dataset_freiburg2_xyz"
            dataset_kind="tum_distort"
            ;;
        fr3|fr3/long_office_household|fr3_long_office_household|tum_fr3_long_office_household|rgbd_dataset_freiburg3_long_office_household)
            sequence_id="rgbd_dataset_freiburg3_long_office_household"
            dataset_kind="tum_undistort"
            ;;
        *)
            die "unsupported sequence '$requested'"
            ;;
    esac
}

config_for() {
    local run_mode="$1"
    local kind="$2"

    case "$run_mode:$kind" in
        mapping_only:euroc)
            echo "$PROJECT_ROOT/configs/mono_mapping_only.yaml"
            ;;
        mapping_only:tum_distort)
            echo "$PROJECT_ROOT/configs/tum_distort_mapping_only.yaml"
            ;;
        mapping_only:tum_undistort)
            echo "$PROJECT_ROOT/configs/tum_undistort_mapping_only.yaml"
            ;;
        slam:euroc)
            echo "$PROJECT_ROOT/configs/mono_slam.yaml"
            ;;
        slam:tum_distort)
            echo "$PROJECT_ROOT/configs/tum_distort_slam.yaml"
            ;;
        slam:tum_undistort)
            echo "$PROJECT_ROOT/configs/tum_undistort_slam.yaml"
            ;;
        *)
            die "no config for mode '$run_mode' and dataset kind '$kind'"
            ;;
    esac
}

print_command() {
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
}

# All runs are deliberately sequential so a single GPU is safe.
for run_mode in "${run_modes[@]}"; do
    for ((repeat_index = 1; repeat_index <= REPEATS; repeat_index++)); do
        for requested in "${requested_sequences[@]}"; do
            resolve_sequence "$requested"
            gs_config="$(config_for "$run_mode" "$dataset_kind")"

            if [[ "$dataset_kind" == "euroc" ]]; then
                image_dir="$EUROC_PATH/$sequence_id/mav0/cam0/data"
                gt_file="$EUROC_PATH/gt_converted/${sequence_id}_gt.txt"
                calib_file="$PROJECT_ROOT/calib/euroc.txt"
                warmup="$EUROC_WARMUP"
                dataset_args=()
            else
                image_dir="$TUM_PATH/$sequence_id/rgb_modified"
                gt_file="$TUM_PATH/$sequence_id/groundtruth.txt"
                warmup="$TUM_WARMUP"
                dataset_args=(--frontend_gap=2 --frontend_nms=1 --age_kf_num=12 --keyframe_thresh=4.0)
                case "$sequence_id" in
                    *freiburg1*) calib_file="$PROJECT_ROOT/calib/tum1.txt" ;;
                    *freiburg2*) calib_file="$PROJECT_ROOT/calib/tum2.txt" ;;
                    *freiburg3*) calib_file="$PROJECT_ROOT/calib/tum3.txt" ;;
                esac
            fi

            [[ -d "$image_dir" ]] || die "missing image directory: $image_dir"
            [[ -f "$gt_file" ]] || die "missing ground truth: $gt_file"
            [[ -f "$calib_file" ]] || die "missing calibration: $calib_file"
            [[ -f "$gs_config" ]] || die "missing config: $gs_config"

            if [[ -n "$RUN_TAG" ]]; then
                run_tag="$RUN_TAG"
                if ((${#run_modes[@]} > 1)); then
                    run_tag="${run_tag}_${run_mode}"
                fi
            else
                run_tag="gfslam_${run_mode}"
            fi

            output_dir="$OUTPUT_ROOT/${sequence_id}_${run_tag}_${range_tag}"
            if ((REPEATS > 1)); then
                output_dir="${output_dir}_run${repeat_index}"
            fi
            log_file="$output_dir/run.log"

            command_args=(
                "$PYTHON_BIN" "$PROJECT_ROOT/demo.py"
                "--imagedir=$image_dir"
                "--calib=$calib_file"
                "--gsconfig_path=$gs_config"
                "--output=$output_dir"
                "--gt_path=$gt_file"
                "--warmup=$warmup"
                "--stride=$STRIDE"
                --upsample
                --scale_timestamps
                "${visualization_args[@]}"
                "${range_args[@]}"
                "${dataset_args[@]}"
                "${demo_args[@]}"
            )

            echo "[run] mode=$run_mode sequence=$sequence_id gui=$USE_GUI repeat=$repeat_index/$REPEATS"
            echo "[run] output=$output_dir"
            if is_true "$DRY_RUN"; then
                print_command "${command_args[@]}"
            else
                mkdir -p "$output_dir"
                "${command_args[@]}" 2>&1 | tee "$log_file"
            fi
        done
    done
done
