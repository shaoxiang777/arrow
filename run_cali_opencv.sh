#!/usr/bin/env bash
set -euo pipefail

# ===== Parameters =====
CAM0_VIDEO="data/ricoh/cali_opencv/cam_1/video.MP4"
CAM1_VIDEO="data/ricoh/cali_opencv/cam_2/video.MP4"
SYNC_ROOT="data/ricoh/cali_opencv/sync"

CAM0_START_FRAME=557
CAM1_START_FRAME=554
TIMESTAMP_FPS=30    # 这里表示的是  录制视频的真实FPS
PAIR_FRAME_STEP=3   # 这个是 采样的帧间隔，表示每隔多少帧采样一次
PAIRS=300
NUM_PAIRS=50

FOV=90
THETA=0
PHI=0
VIEW_WIDTH=1000
VIEW_HEIGHT=1000

CHECKERBOARD_ROWS=9
CHECKERBOARD_COLS=6
CHECKERBOARD_SQUARE_SIZE=0.024

RUN_EXTRACT_sync=0
RUN_EXTRACT_360=0
SHOW_EXTRACTION=1

# ===== Derived paths =====
PAIR_DIR="${SYNC_ROOT}/paired_${CAM0_START_FRAME}_${CAM1_START_FRAME}"
OUTPUT_DIR="data/ricoh/cali_opencv/virtual_sync_${CAM0_START_FRAME}_${CAM1_START_FRAME}"
OPENCV_OUTPUT="${OUTPUT_DIR}/checkerboard_camchain.yaml"
VISUALIZATION_DIR="${OUTPUT_DIR}/visualization"
BAG_PATH="${OUTPUT_DIR}.bag"

# ==== Run the extraction of frames， 这个过程很粗糙，是直接解码的 =====
# ==== 目的是找到合适的开始，以及结束时候的帧对 =====
if [[ "${RUN_EXTRACT_sync}" -eq 1 ]]; then
  python stereo_calibration/richo_syn.py \
    --cam0-video "${CAM0_VIDEO}" \
    --cam1-video "${CAM1_VIDEO}" \
    extract \
    --output-root "${SYNC_ROOT}" \
    --every 1 \
    --jpeg-quality 95 \
    --overwrite
  exit 0
fi

# ===== 真正的提取出来高质量的equiretangular images =====
if [[ "${RUN_EXTRACT_360}" -eq 1 ]]; then
echo "[1/3] Pairing synchronized equirectangular frames..."
python stereo_calibration/richo_syn.py \
  --cam0-video "${CAM0_VIDEO}" \
  --cam1-video "${CAM1_VIDEO}" \
  pair \
  --cam0-start-frame "${CAM0_START_FRAME}" \
  --cam1-start-frame "${CAM1_START_FRAME}" \
  --pairs "${PAIRS}" \
  --frame-step "${PAIR_FRAME_STEP}" \
  --timestamp-fps "${TIMESTAMP_FPS}" \
  --output-dir "${PAIR_DIR}" \
  --overwrite
    exit 0
fi

# ==== 转化equirectangular frames 到 perspective =====
echo "[2/3] Converting equirectangular frames to perspective..."
python stereo_calibration/richo_cal.py \
  --cam0-frames-dir "${PAIR_DIR}/cam0" \
  --cam1-frames-dir "${PAIR_DIR}/cam1" \
  --num_pair "${NUM_PAIRS}" \
  --fov "${FOV}" \
  --theta "${THETA}" \
  --phi "${PHI}" \
  --view-width "${VIEW_WIDTH}" \
  --view-height "${VIEW_HEIGHT}" \
  --output-dir "${OUTPUT_DIR}" \
  --overwrite

# ==== 开始对于perspective images pair的标定 =====
echo "[3/3] Running OpenCV checkerboard stereo calibration..."
python stereo_calibration/calib_offline.py \
  --cam0-dir "${OUTPUT_DIR}/cam0" \
  --cam1-dir "${OUTPUT_DIR}/cam1" \
  --checkerboard-rows "${CHECKERBOARD_ROWS}" \
  --checkerboard-cols "${CHECKERBOARD_COLS}" \
  --square-size "${CHECKERBOARD_SQUARE_SIZE}" \
  --output "${OPENCV_OUTPUT}" \
  --visualization-dir "${VISUALIZATION_DIR}"
