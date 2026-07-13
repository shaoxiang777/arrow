#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-demo}"
GREEN="\033[92m"
RESET="\033[0m"

log_step() {
  printf "${GREEN}%s${RESET}\n" "$1"
}

# python3 tools/crop_video.py \
#   --video-path data/arrow360/${SCENE}/video.mp4 \
#   --start-time 00:17 \
#   --end-time 01:30 \
#   --output-path data/arrow360/${SCENE}/video_crop.mp4



log_step "[1/2] Running Stella VSLAM for scene: ${SCENE}"
if [[ -f "data/arrow360/${SCENE}/my_saved_map.msg" && -f "data/arrow360/${SCENE}/frame_trajectory.txt" ]]; then
  log_step "[1/2] Reusing existing SLAM map and trajectory for scene: ${SCENE}"
else
  docker run --rm --privileged \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
      -v ~/Documents/project/arrow:/work \
      --gpus all \
      stella_vslam-desktop \
      -lc "cd /stella_vslam_examples/build && ./run_video_slam \
        -v /work/orb_vocab.fbow \
        -c /work/data/arrow360/${SCENE}/config.yaml \
        -m /work/data/arrow360/${SCENE}/video_crop.mp4 \
        --frame-skip 2 \
        --no-sleep \
        --eval-log-dir /work/data/arrow360/${SCENE}/ \
        --map-db-out /work/data/arrow360/${SCENE}/my_saved_map.msg"
fi

if [[ ! -f "data/arrow360/${SCENE}/my_saved_map.msg" ]]; then
  echo "SLAM finished but map was not created: data/arrow360/${SCENE}/my_saved_map.msg" >&2
  exit 1
fi
if [[ ! -f "data/arrow360/${SCENE}/frame_trajectory.txt" ]]; then
  echo "SLAM finished but trajectory was not created: data/arrow360/${SCENE}/frame_trajectory.txt" >&2
  exit 1
fi

MAP_PATH="data/arrow360/${SCENE}/my_saved_map.msg"
TRAJ_PATH="data/arrow360/${SCENE}/frame_trajectory.txt"
if [[ -f "data/arrow360/${SCENE}/my_saved_map_metric.msg" && -f "data/arrow360/${SCENE}/frame_trajectory_metric.txt" ]]; then
  MAP_PATH="data/arrow360/${SCENE}/my_saved_map_metric.msg"
  TRAJ_PATH="data/arrow360/${SCENE}/frame_trajectory_metric.txt"
  log_step "[INFO] Using metric map and trajectory for scene: ${SCENE}"
fi


# 运行DAP进行深度推理
log_step "[2/2] Running DAP/object detection for arrow360/${SCENE}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dap

# 进行四个视角的object detection
if [[ -f "output/arrow360/${SCENE}/video_crop_object_detection.mp4" ]] && [[ $(stat -c%s "output/arrow360/${SCENE}/video_crop_object_detection.mp4") -lt 1024 ]]; then
  rm "output/arrow360/${SCENE}/video_crop_object_detection.mp4"
fi
python tools/equirec2perspec_video.py \
  --video data/arrow360/${SCENE}/video_crop.mp4 \
  --output output/arrow360/${SCENE} \
  --track-output data/arrow360/${SCENE}/object_tracks.json \
  --map-path "${MAP_PATH}" \
  --frame-traj-path "${TRAJ_PATH}" \
  --slam-frame-skip 2 \
  --depth-model dap \
  --frame_skip 10 \
  --adaptive-dap \
  --kalman_filter \
  --track-max-speed 3.0 \
  --max_depth 100



# # optional 运行可视化GUI
# python3 visualize_map_gui.py \
#   --map data/arrow360/jia/my_saved_map_metric.msg \
#   --fps 30 \
#   --frame-traj data/arrow360/jia/frame_trajectory_metric.txt \
#   --slam-frame-skip 2 \
#   --overlay-video output/arrow360/jia/video_crop_object_detection.mp4
