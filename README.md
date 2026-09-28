<p align="center">
  <h1 align="center">GaussianFlow SLAM</h1>
  <p align="center">
    <strong>Dong-Uk Seo</strong>
    ·
    <strong>Jinwoo Jeon</strong>
    ·
    <strong>Eungchang Mason Lee</strong>
    ·
    <strong>Hyun Myung</strong>
  </p>
  <h3 align="center">IEEE RA-L 2026</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2604.15612">Paper</a> | <a href="https://www.youtube.com/watch?v=3JJMqJIWMBg&t=1s">Video</a> | <a href="https://gaussianflow-slam.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <img src="./media/fr1_desk.gif" alt="TUM fr1/desk" width="48%">
  <img src="./media/MH05.gif" alt="EuRoC MH05" width="48%">
</p>
<p align="center">
  Given a monocular video sequence, GaussianFlow SLAM tracks the camera trajectory and reconstructs a photorealistic 3DGS map by leveraging optical flow as geometric supervision, without any depth measurements.
</p>

# Getting Started

## Installation

```bash
git clone --recursive https://github.com/url-kaist/gaussianflow-slam.git
cd gaussianflow-slam
```

System packages:

```bash
sudo apt install build-essential libeigen3-dev libopencv-dev python3-tk \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Adjust the `torch`, `torchvision`, `torchaudio` versions and the `--extra-index-url` in `requirements.txt` to match your CUDA version.

CUDA extensions (built locally from `thirdparty/` and `src/`):

```bash
pip install thirdparty/lietorch
pip install thirdparty/pytorch_scatter
pip install thirdparty/simple-knn
pip install -e thirdparty/diff-gaussian-rasterization \
    --use-pep517 --no-build-isolation --force-reinstall
pip install -e .
```

Pre-trained weights:

```bash
./tools/download_model.sh
```

Reference environment: Ubuntu 20.04, Python 3.10, CUDA 11.8, PyTorch 2.1.2+cu118.

## Downloading Datasets

### EuRoC MAV

Download the ASL-format sequences from the
[EuRoC MAV page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
and convert the ground truth to TUM format (`gt_converted/<seq>_gt.txt`).

### TUM-RGBD

```bash
./tools/download_tum.sh
```

## Run

### SLAM mode

```bash
./tools/run_gaussianflow_slam.sh V101          # GUI
./tools/run_gaussianflow_slam_nogui.sh V101    # headless
```

### Mapping-only mode

```bash
./tools/run_gaussianflow_mapping_only.sh V101          # GUI
./tools/run_gaussianflow_mapping_only_nogui.sh V101    # headless
```

Sequences: `V101`, `MH01`, `fr1/desk`, `fr2`, `fr3`. Options and environment
variables: `tools/run_gaussianflow_modes.sh --help`.

### Dataset scripts

Edit `DATA_PATH` at the top of each script to point to your local dataset, then:

```bash
./tools/run_euroc_gsflow.sh     # EuRoC
./tools/run_tum_gsflow.sh       # TUM-RGBD
```

Extra `demo.py` flags can be appended to the script call, e.g.:

```bash
./tools/run_euroc_gsflow.sh --t1=600 --stride=2
```

### Custom data

You only need (1) a folder of RGB frames named so they sort in capture order
and (2) a one-line calibration file `fx fy cx cy [k1 k2 p1 p2 [k3 [k4 k5 k6]]]`.
Then:

```bash
python3 demo.py --imagedir=<path/to/rgb> --calib=<calib.txt> \
    --gsconfig_path=configs/mono_slam.yaml --upsample --disable_vis \
    --output=results/<run>/
```

# Evaluation

Passing `--gt_path` makes `demo.py` report ATE and write `traj_full.txt` /
`traj_kf.txt` to the output directory. To recompute ATE from a finished run:

```bash
python3 tools/compute_ate.py results/<run>/traj_full.txt <gt.txt>
```

# Acknowledgement

This work builds on many open-source projects. We extend our gratitude to the
authors:

- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [MonoGS](https://github.com/muskie82/MonoGS)
- [GaussianFlow](https://arxiv.org/abs/2403.12365)

# License

GaussianFlow SLAM is released under the **[GPL-3.0 License](LICENSE)**. For a list of code dependencies which are not property of the authors of GaussianFlow SLAM, please check **[Dependencies.md](Dependencies.md)**.

# Citation

If you found this work useful in your own research, please consider
citing:

```bibtex
@article{seo2026gaussianflow,
  title={GaussianFlow SLAM: Monocular Gaussian Splatting SLAM Guided by GaussianFlow},
  author={Seo, Dong-Uk and Jeon, Jinwoo and Lee, Eungchang Mason and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE}
}
```
