<p align="center">
  <img src="media/arrow_logo.png" alt="Arrow logo" width="240">
</p>

<h1 align="center">Arrow: 360° Panoramic Perception and Visual SLAM</h1>

Arrow is a perception and visualization pipeline for 360° panoramic video. It combines:

- panoramic visual SLAM with Stella VSLAM;
- multi-view object detection and tracking with Ultralytics YOLO;
- panoramic depth estimation with DAP (Depth Any Panoramas);
- metric map-scale estimation using a calibrated stereo camera pair;
- Open3D visualization of the SLAM map, camera trajectory, and object tracks.

## 🧰 Requirements

- Linux (Ubuntu 22.04 recommended)
- Conda with Python 3.10
- FFmpeg
- Docker
- NVIDIA GPU, NVIDIA driver, and NVIDIA Container Toolkit
- A desktop session with display access for the Stella VSLAM and Open3D viewers

## 📦 Get the Code

```bash
git clone --recursive https://github.com/shaoxiang777/arrow.git
cd arrow
```

The `third_part/DAP` submodule uses a GitHub SSH URL. If the recursive clone fails, configure your GitHub SSH key and run:

```bash
git submodule update --init --recursive
```

## 🐍 Create the Conda Environment

Create an environment named `arrow`:

```bash
conda create -n arrow python=3.10 -y
conda activate arrow
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The root `requirements.txt` is based on the local `arrow` environment and currently uses PyTorch 2.5.1 and torchvision 0.20.1.

### Install DAP dependencies

DAP has an additional requirements file:

```bash
pip install -r third_part/DAP/requirements.txt
```

DAP may install a different PyTorch stack. Verify that PyTorch can access the GPU after installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Download the [pretrained DAP model](https://huggingface.co/Insta360-Research/DAP-weights/blob/main/model.pth), then place it at:

```text
third_part/DAP/weights/model.pth
```

## 🐳 Build the Stella VSLAM Docker Image

From the repository root, run:

```bash
docker build \
  -t stella_vslam-desktop \
  -f Dockerfile.desktop \
  --build-arg NUM_THREADS="$(nproc)" \
  .
```

Allow the container to access the host display server:

```bash
xhost +SI:localuser:root
```

## 📐 Camera Calibration

`run_cali_opencv.sh` calibrates a synchronized pair of 360° cameras with an OpenCV checkerboard workflow. The process finds the temporal offset, extracts synchronized equirectangular frames, converts them to perspective views, and estimates the stereo calibration.

### 1. Prepare the checkerboard and videos

Print [`stereo_calibration/calib.io_checker_297x210_7x10_25.pdf`](stereo_calibration/calib.io_checker_297x210_7x10_25.pdf) on A4 paper without automatic page scaling. Attach it to a flat, rigid surface and measure the printed square size with a ruler.

Place the two calibration videos at:

```text
data/ricoh/cali_opencv/
├── cam_0/video.MP4
└── cam_1/video.MP4
```

Activate the project environment:

```bash
conda activate arrow
```

Open `run_cali_opencv.sh` and update the parameters for your recordings:

```bash
CAM0_VIDEO="data/ricoh/cali_opencv/cam_0/video.MP4"
CAM1_VIDEO="data/ricoh/cali_opencv/cam_1/video.MP4"

CAM0_START_FRAME=557
CAM1_START_FRAME=554
TIMESTAMP_FPS=30
PAIR_FRAME_STEP=3
PAIRS=300
NUM_PAIRS=50

CHECKERBOARD_ROWS=9
CHECKERBOARD_COLS=6
CHECKERBOARD_SQUARE_SIZE=0.024
```

- `CAM0_START_FRAME` and `CAM1_START_FRAME`: matching start frames in the two videos.
- `TIMESTAMP_FPS`: actual recording frame rate.
- `PAIR_FRAME_STEP`: interval between sampled frame pairs.
- `PAIRS`: maximum number of synchronized frame pairs to generate.
- `NUM_PAIRS`: number of uniformly selected pairs used for calibration.
- `CHECKERBOARD_ROWS` and `CHECKERBOARD_COLS`: OpenCV checkerboard inner-corner dimensions.
- `CHECKERBOARD_SQUARE_SIZE`: measured checkerboard square size in meters.

### 2. Find the synchronization offset

Set:

```bash
RUN_EXTRACT_sync=1
RUN_EXTRACT_360=0
```

Run:

```bash
bash run_cali_opencv.sh
```

The script decodes both videos into `cam_0/frames` and `cam_1/frames` under `SYNC_ROOT`. Compare the images and identify frame numbers that represent the same moment.

<table>
  <tr>
    <th align="center">cam_0 · frame 000127</th>
    <th align="center">cam_1 · frame 000124</th>
  </tr>
  <tr>
    <td><img src="media/cam_0_frame_000127.jpg" alt="Camera 0 frame 000127" width="100%"></td>
    <td><img src="media/cam_1_frame_000124.jpg" alt="Camera 1 frame 000124" width="100%"></td>
  </tr>
</table>

These images show the same moment at frame `127` in `cam_0` and frame `124` in `cam_1`, revealing a fixed offset of three frames.

The final start frames used by this project are:

```bash
CAM0_START_FRAME=557
CAM1_START_FRAME=554
```

Because `557 - 554 = 3`, the selected start frames preserve the same three-frame offset. Recalculate this offset when using different recordings.

### 3. Generate synchronized equirectangular pairs

Set:

```bash
RUN_EXTRACT_sync=0
RUN_EXTRACT_360=1
```

Run the script again:

```bash
bash run_cali_opencv.sh
```

The synchronized frame pairs are written to:

```text
data/ricoh/cali_opencv/sync/paired_<cam0_start>_<cam1_start>/
├── cam0/
├── cam1/
└── pairs.csv
```

Check that images with matching filenames show the same moment and that the checkerboard is sharp and visible in both cameras.

### 4. Run the OpenCV stereo calibration

Set:

```bash
RUN_EXTRACT_sync=0
RUN_EXTRACT_360=0
```

Run:

```bash
bash run_cali_opencv.sh
```

The script selects `NUM_PAIRS` synchronized pairs, converts them to perspective views, detects the checkerboard, and performs stereo calibration. The default output is:

```text
data/ricoh/cali_opencv/virtual_sync_perspective/
├── cam0/
├── cam1/
├── checkerboard_camchain.yaml
└── visualization/
```

`checkerboard_camchain.yaml` contains the perspective-camera intrinsics, distortion parameters, inter-camera transformation, reprojection errors, and stereo baseline. Inspect `visualization/` for incorrect checkerboard detections or reprojections.

The script prints a result similar to:

```text
baseline: 0.263175 m (26.32 cm)
```

Compare this value with the physically measured distance between the two camera centers.

## 📁 Scene Data Layout

Each scene used by the complete pipeline requires:

```text
data/arrow360/<scene>/
├── config.yaml
├── video_right_crop.mp4
└── video_left_crop.MP4
```

- `video_right_crop.mp4` is the equirectangular video processed by Stella VSLAM.
- `video_left_crop.MP4` is the synchronized video from the other calibrated camera and is used to estimate the metric map scale.
- `config.yaml` is the Stella VSLAM camera configuration.

The stereo calibration must be available at:

```text
data/ricoh/cali_opencv/virtual_sync_perspective/checkerboard_camchain.yaml
```

## ⚡ Fast Run

After installing the dependencies, building the Docker image, calibrating the stereo cameras, and preparing the scene data, run the complete pipeline with:

```bash
conda activate arrow
bash run_slam.sh demo
```

Replace `demo` with another scene directory name when needed. The script automatically downloads the ORB vocabulary if it is missing and reuses completed outputs when possible.

The complete pipeline is:

```text
Stella VSLAM
    ↓
my_saved_map.msg + frame_trajectory.txt
    ↓
estimate_stereo_map_scale.py
    ↓
my_saved_map_metric.msg + frame_trajectory_metric.txt
    ↓
DAP depth inference + object detection and tracking
```

The main generated files are:

```text
data/arrow360/<scene>/
├── frame_trajectory.txt
├── my_saved_map.msg
├── frame_trajectory_metric.txt
├── my_saved_map_metric.msg
├── stereo_map_scale_keyframes.csv
├── stereo_map_scale.yaml
├── stereo_map_scale_visualization/
└── object_tracks.json

output/arrow360/<scene>/
└── video_right_crop_object_detection.mp4
```

`run_slam.sh` expects Conda to be initialized from `~/miniconda3/etc/profile.d/conda.sh`. Update that line if Conda is installed elsewhere.

## 🧭 Manual Pipeline

Use this section when you want to run or debug each stage separately.

### 1. Run Stella VSLAM

Download the ORB vocabulary if it is not already present:

```bash
curl -fL \
  https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow \
  -o orb_vocab.fbow
```

Run Stella VSLAM for the `demo` scene:

```bash
docker run --rm --privileged \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$PWD:/work" \
  --gpus all \
  stella_vslam-desktop \
  -lc 'cd /stella_vslam_examples/build && ./run_video_slam \
    -v /work/orb_vocab.fbow \
    -c /work/data/arrow360/demo/config.yaml \
    -m /work/data/arrow360/demo/video_right_crop.mp4 \
    --frame-skip 2 \
    --no-sleep \
    --eval-log-dir /work/data/arrow360/demo/ \
    --map-db-out /work/data/arrow360/demo/my_saved_map.msg'
```

This stage generates `my_saved_map.msg` and `frame_trajectory.txt`.

### 2. Estimate the metric map scale

```bash
python tools/estimate_stereo_map_scale.py demo \
  --slam-video data/arrow360/demo/video_right_crop.mp4 \
  --slam-camera cam1 \
  --calibration data/ricoh/cali_opencv/virtual_sync_perspective/checkerboard_camchain.yaml \
  --other-video data/arrow360/demo/video_left_crop.MP4
```

This stage generates `my_saved_map_metric.msg`, `frame_trajectory_metric.txt`, and scale diagnostics.

### 3. Run DAP depth inference and object tracking

```bash
conda activate arrow

python tools/equirec2perspec_video.py \
  --video data/arrow360/demo/video_right_crop.mp4 \
  --output output/arrow360/demo \
  --track-output data/arrow360/demo/object_tracks.json \
  --map-path data/arrow360/demo/my_saved_map_metric.msg \
  --frame-traj-path data/arrow360/demo/frame_trajectory_metric.txt \
  --slam-frame-skip 2 \
  --depth-model dap \
  --frame_skip 10 \
  --adaptive-dap \
  --kalman_filter \
  --track-max-speed 3.0 \
  --max_depth 100
```

### 4. Visualize the metric map and object tracks

```bash
python visualize_map_gui.py \
  --map data/arrow360/demo/my_saved_map_metric.msg \
  --fps 30 \
  --tracks-json data/arrow360/demo/object_tracks.json \
  --frame-traj data/arrow360/demo/frame_trajectory_metric.txt \
  --slam-frame-skip 2 \
  --overlay-video output/arrow360/demo/video_right_crop_object_detection.mp4
```

## 🛠️ Troubleshooting

### A submodule cannot be downloaded

Confirm that your GitHub SSH key works, then initialize the submodules again:

```bash
ssh -T git@github.com
git submodule update --init --recursive
```

### Docker cannot access the GPU

Install NVIDIA Container Toolkit and verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

### The GUI cannot open

Allow local root access to the display server and verify `DISPLAY`:

```bash
xhost +SI:localuser:root
echo "$DISPLAY"
```

### Metric scale estimation reports a missing file

Confirm that the scene contains both stereo videos and that calibration completed successfully:

```bash
ls data/arrow360/demo/video_right_crop.mp4
ls data/arrow360/demo/video_left_crop.MP4
ls data/ricoh/cali_opencv/virtual_sync_perspective/checkerboard_camchain.yaml
```

## 🙏 Acknowledgements and License

Arrow builds on [Stella VSLAM](https://github.com/stella-cv/stella_vslam), DAP, Ultralytics YOLO, Open3D, and other open-source projects. See [LICENSE](LICENSE) and the license files in each submodule for details.
