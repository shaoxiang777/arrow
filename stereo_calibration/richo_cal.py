#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.Equirec2Perspec as E2P


DEFAULT_CAM0_VIDEO = REPO_ROOT / "data/ricoh/cali/cam_1/aprilgrid.MP4"
DEFAULT_CAM1_VIDEO = REPO_ROOT / "data/ricoh/cali/cam_2/aprilgrid.MP4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/ricoh/cali/kalibr_virtual_front"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _check_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _check_dir(path, label):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _list_images(path):
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0.0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: fps={fps}, frames={frames}")
    return {"fps": fps, "frames": frames, "width": width, "height": height, "duration": frames / fps}


def _read_frame(cap, frame_idx, path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {path}")
    return frame


def _convert_equirect_to_pinhole(frame, fov, theta, phi, height, width):
    equ = E2P.Equirectangular(frame)
    return equ.GetPerspective(fov, theta, phi, height, width)


def _timestamp_ns(start_ns, sample_idx, sample_fps):
    return int(start_ns + round(sample_idx * 1_000_000_000.0 / sample_fps))


def _evenly_spaced_indices(total, requested):
    if requested is None or requested >= total:
        return list(range(total))
    if requested <= 0:
        raise ValueError(f"num-pairs must be positive, got {requested}")
    return np.linspace(0, total - 1, requested, dtype=int).tolist()


def prepare_virtual_pinhole_dataset(args):
    if args.cam0_frames_dir or args.cam1_frames_dir:
        return prepare_virtual_pinhole_dataset_from_frames(args)

    cam0_video = _check_file(args.cam0_video, "cam0 video")
    cam1_video = _check_file(args.cam1_video, "cam1 video")

    info0 = _video_info(cam0_video)
    info1 = _video_info(cam1_video)
    print(f"cam0: {info0['width']}x{info0['height']} {info0['fps']:.3f} fps, {info0['duration']:.2f}s")
    print(f"cam1: {info1['width']}x{info1['height']} {info1['fps']:.3f} fps, {info1['duration']:.2f}s")

    duration = min(info0["duration"], info1["duration"])
    start_s = max(0.0, float(args.start_s))
    end_s = duration if args.end_s is None else min(duration, float(args.end_s))
    if end_s <= start_s:
        raise ValueError(f"invalid time range: start={start_s}, end={end_s}, duration={duration}")

    input_fps = min(info0["fps"], info1["fps"])
    available_count = int(math.floor((end_s - start_s) * input_fps)) + 1
    if available_count <= 0:
        raise RuntimeError("no samples selected")
    selected_indices = _evenly_spaced_indices(available_count, args.num_pairs)

    output_dir = Path(args.output_dir)
    cam0_out = output_dir / "cam0"
    cam1_out = output_dir / "cam1"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    cam0_out.mkdir(parents=True)
    cam1_out.mkdir(parents=True)

    cap0 = cv2.VideoCapture(str(cam0_video))
    cap1 = cv2.VideoCapture(str(cam1_video))
    try:
        for sample_idx in tqdm(selected_indices, desc="Converting perspective frames", unit="pair"):
            t_s = start_s + sample_idx / input_fps
            idx0 = int(round(t_s * info0["fps"]))
            idx1 = int(round(t_s * info1["fps"]))
            idx0 = min(idx0, info0["frames"] - 1)
            idx1 = min(idx1, info1["frames"] - 1)

            frame0 = _read_frame(cap0, idx0, cam0_video)
            frame1 = _read_frame(cap1, idx1, cam1_video)

            view0 = _convert_equirect_to_pinhole(
                frame0, args.fov, args.theta, args.phi, args.view_height, args.view_width
            )
            view1 = _convert_equirect_to_pinhole(
                frame1, args.fov, args.theta, args.phi, args.view_height, args.view_width
            )

            ts = _timestamp_ns(args.start_ns, sample_idx, input_fps)
            name = f"{ts}.png"
            cv2.imwrite(str(cam0_out / name), view0)
            cv2.imwrite(str(cam1_out / name), view1)
    finally:
        cap0.release()
        cap1.release()

    print(f"wrote Kalibr image dataset: {output_dir}")
    return output_dir


def prepare_virtual_pinhole_dataset_from_frames(args):
    cam0_dir = _check_dir(args.cam0_frames_dir, "cam0 frames dir")
    cam1_dir = _check_dir(args.cam1_frames_dir, "cam1 frames dir")
    cam0_images = _list_images(cam0_dir)
    cam1_images = _list_images(cam1_dir)
    available_count = min(len(cam0_images), len(cam1_images))
    if available_count <= 0:
        raise RuntimeError(f"no image pairs found in {cam0_dir} and {cam1_dir}")
    if len(cam0_images) != len(cam1_images):
        print(
            f"warning: cam0 has {len(cam0_images)} images, cam1 has {len(cam1_images)}; "
            f"using at most {available_count} pairs"
        )
    selected_indices = _evenly_spaced_indices(available_count, args.num_pairs)

    output_dir = Path(args.output_dir)
    cam0_out = output_dir / "cam0"
    cam1_out = output_dir / "cam1"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    cam0_out.mkdir(parents=True)
    cam1_out.mkdir(parents=True)

    print(f"cam0 frames: {cam0_dir} ({len(cam0_images)})")
    print(f"cam1 frames: {cam1_dir} ({len(cam1_images)})")

    for pair_idx in tqdm(selected_indices, desc="Converting perspective frames", unit="pair"):
        cam0_path = cam0_images[pair_idx]
        cam1_path = cam1_images[pair_idx]
        frame0 = cv2.imread(str(cam0_path), cv2.IMREAD_COLOR)
        frame1 = cv2.imread(str(cam1_path), cv2.IMREAD_COLOR)
        if frame0 is None:
            raise RuntimeError(f"failed to read image: {cam0_path}")
        if frame1 is None:
            raise RuntimeError(f"failed to read image: {cam1_path}")

        view0 = _convert_equirect_to_pinhole(
            frame0, args.fov, args.theta, args.phi, args.view_height, args.view_width
        )
        view1 = _convert_equirect_to_pinhole(
            frame1, args.fov, args.theta, args.phi, args.view_height, args.view_width
        )

        if cam0_path.stem != cam1_path.stem:
            raise RuntimeError(
                f"frame timestamps do not match: {cam0_path.name} vs {cam1_path.name}"
            )
        name = f"{cam0_path.stem}.png"
        cv2.imwrite(str(cam0_out / name), view0)
        cv2.imwrite(str(cam1_out / name), view1)

    print(f"wrote Kalibr image dataset: {output_dir}")
    return output_dir


def _run(cmd, cwd):
    print("+ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _repo_relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{path} must be inside repo root for Docker Kalibr: {REPO_ROOT}") from exc


def run_kalibr(args):
    if not args.target:
        raise ValueError("--target is required when --run-kalibr is enabled")

    output_dir = Path(args.output_dir).resolve()
    bag_path = output_dir.with_suffix(".bag")
    target_path = _check_file(args.target, "target yaml").resolve()

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:/data",
        args.docker_image,
        "bash",
        "-lc",
    ]

    rel_output = _repo_relative(output_dir)
    rel_bag = _repo_relative(bag_path)
    rel_target = _repo_relative(target_path)

    bag_cmd = (
        "source /catkin_ws/devel/setup.bash && "
        f"kalibr_bagcreater --folder /data/{rel_output} --output-bag /data/{rel_bag}"
    )
    _run(docker_cmd + [bag_cmd], REPO_ROOT)

    calib_cmd = (
        "source /catkin_ws/devel/setup.bash && "
        f"cd /data/{rel_output.parent} && "
        f"kalibr_calibrate_cameras "
        f"--bag /data/{rel_bag} "
        f"--topics /cam0/image_raw /cam1/image_raw "
        f"--models {args.cam_model} {args.cam_model} "
        f"--target /data/{rel_target} "
        f"{'--show-extraction ' if args.show_extraction else ''}"
    )
    _run(docker_cmd + [calib_cmd], REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Ricoh Theta stereo calibration data: paired videos/frames -> "
            "virtual pinhole images -> optional Kalibr bag/calibration."
        )
    )
    parser.add_argument("--cam0-video", default=str(DEFAULT_CAM0_VIDEO))
    parser.add_argument("--cam1-video", default=str(DEFAULT_CAM1_VIDEO))
    parser.add_argument("--cam0-frames-dir", default=None, help="Optional synchronized cam0 equirectangular frame directory")
    parser.add_argument("--cam1-frames-dir", default=None, help="Optional synchronized cam1 equirectangular frame directory")
    parser.add_argument(
        "--num-pairs",
        "--num_pair",
        dest="num_pairs",
        type=int,
        default=None,
        help="Select this many evenly spaced synchronized pairs; default uses all pairs",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Kalibr target YAML; only required together with --run-kalibr",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--theta", type=float, default=0.0, help="Virtual view yaw angle in degrees")
    parser.add_argument("--phi", type=float, default=0.0, help="Virtual view pitch angle in degrees")
    parser.add_argument("--view-width", type=int, default=1000)
    parser.add_argument("--view-height", type=int, default=1000)
    parser.add_argument("--start-ns", type=int, default=1700000000000000000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-kalibr", action="store_true", help="Also run kalibr_bagcreater and kalibr_calibrate_cameras in Docker")
    parser.add_argument("--docker-image", default="kalibr")
    parser.add_argument("--cam-model", default="pinhole-radtan")
    parser.add_argument("--show-extraction", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    prepare_virtual_pinhole_dataset(args)
    if args.run_kalibr:
        run_kalibr(args)


if __name__ == "__main__":
    main()
