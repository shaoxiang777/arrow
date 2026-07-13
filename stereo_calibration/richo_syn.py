#!/usr/bin/env python3
import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAM0_VIDEO = REPO_ROOT / "data/ricoh/cali/cam_1/8K.MP4"
DEFAULT_CAM1_VIDEO = REPO_ROOT / "data/ricoh/cali/cam_2/8K.MP4"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/ricoh/cali/sync_8k"


def _check_video(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return cap


def _video_info(path):
    cap = _open_video(path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def _prepare_dir(path, overwrite):
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _resize_to_width(image, width):
    if width <= 0 or image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _draw_label(image, text):
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, out.shape[1] / 1200.0)
    thickness = max(1, int(round(scale * 2)))
    margin = max(8, int(round(scale * 12)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (margin - 4, margin - th - 8), (margin + tw + 8, margin + 8), (0, 0, 0), -1)
    cv2.putText(out, text, (margin, margin), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def extract_video_frames(video_path, output_dir, every, max_frames, jpeg_quality):
    video_path = _check_video(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = _open_video(video_path)
    info = _video_info(video_path)
    rows = []
    written = 0
    frame_idx = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % every != 0:
                frame_idx += 1
                continue
            if max_frames is not None and written >= max_frames:
                break

            t_s = frame_idx / info["fps"]
            full_name = f"frame_{frame_idx:06d}_t_{t_s:09.3f}.jpg"
            full_path = output_dir / "frames" / full_name
            full_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full_path), frame, encode_params)

            rows.append([frame_idx, f"{t_s:.6f}", str(full_path.relative_to(output_dir))])
            written += 1
            print(f"{output_dir.name}: wrote frame {frame_idx} ({written})")
            frame_idx += 1
    finally:
        cap.release()

    with open(output_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_s", "frame_path"])
        writer.writerows(rows)
    return written


def command_extract(args):
    root = _prepare_dir(args.output_root, args.overwrite)
    info0 = _video_info(args.cam0_video)
    info1 = _video_info(args.cam1_video)
    print(f"cam0: {info0}")
    print(f"cam1: {info1}")
    extract_video_frames(args.cam0_video, root / "cam_1", args.every, args.max_frames, args.jpeg_quality)
    extract_video_frames(args.cam1_video, root / "cam_2", args.every, args.max_frames, args.jpeg_quality)
    print(f"wrote extracted frames under: {root}")


def _read_frame_at(cap, frame_idx, video_path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def command_contact_sheet(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap0 = _open_video(args.cam0_video)
    cap1 = _open_video(args.cam1_video)
    info0 = _video_info(args.cam0_video)
    info1 = _video_info(args.cam1_video)
    max_frame = min(info0["frames"], info1["frames"])
    tiles = []
    try:
        for frame_idx in range(args.start_frame, max_frame, args.step):
            if len(tiles) >= args.count:
                break
            f0 = _read_frame_at(cap0, frame_idx, args.cam0_video)
            f1_idx = int(np.clip(frame_idx + args.cam1_offset_frames, 0, info1["frames"] - 1))
            f1 = _read_frame_at(cap1, f1_idx, args.cam1_video)
            f0 = _resize_to_width(f0, args.tile_width)
            f1 = _resize_to_width(f1, args.tile_width)
            f0 = _draw_label(f0, f"cam0 frame {frame_idx}")
            f1 = _draw_label(f1, f"cam1 frame {f1_idx}  offset={args.cam1_offset_frames}")
            tile = np.vstack([f0, f1])
            tiles.append(tile)
    finally:
        cap0.release()
        cap1.release()

    if not tiles:
        raise RuntimeError("no tiles generated")

    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)
    padded = []
    for tile in tiles:
        canvas = np.full((max_h, max_w, 3), 24, dtype=np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        padded.append(canvas)

    rows = []
    for i in range(0, len(padded), args.cols):
        row_tiles = padded[i : i + args.cols]
        while len(row_tiles) < args.cols:
            row_tiles.append(np.full((max_h, max_w, 3), 24, dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote contact sheet: {out}")


def command_pair(args):
    output_dir = _prepare_dir(args.output_dir, args.overwrite)
    cam0_out = output_dir / "cam0"
    cam1_out = output_dir / "cam1"
    cam0_out.mkdir()
    cam1_out.mkdir()

    info0 = _video_info(args.cam0_video)
    info1 = _video_info(args.cam1_video)
    fps = args.timestamp_fps or min(info0["fps"], info1["fps"])
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    max_pairs = min(
        math.floor((info0["frames"] - 1 - args.cam0_start_frame) / args.frame_step) + 1,
        math.floor((info1["frames"] - 1 - args.cam1_start_frame) / args.frame_step) + 1,
    )
    if args.pairs is not None:
        max_pairs = min(max_pairs, args.pairs)
    if max_pairs <= 0:
        raise RuntimeError("no valid frame pairs")

    cap0 = _open_video(args.cam0_video)
    cap1 = _open_video(args.cam1_video)
    rows = []
    try:
        for i in tqdm(range(max_pairs), desc="Pairing frames", unit="pair"):
            idx0 = args.cam0_start_frame + i * args.frame_step
            idx1 = args.cam1_start_frame + i * args.frame_step
            f0 = _read_frame_at(cap0, idx0, args.cam0_video)
            f1 = _read_frame_at(cap1, idx1, args.cam1_video)
            ts = args.start_ns + int(round(i * args.frame_step * 1_000_000_000.0 / fps))
            name = f"{ts}.png"
            cv2.imwrite(str(cam0_out / name), f0)
            cv2.imwrite(str(cam1_out / name), f1)
            rows.append([i, idx0, idx1, ts, name])
    finally:
        cap0.release()
        cap1.release()

    with open(output_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_idx", "cam0_frame_idx", "cam1_frame_idx", "timestamp_ns", "filename"])
        writer.writerows(rows)
    print(f"wrote synchronized Kalibr equirectangular frames: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Ricoh stereo video frame extraction and manual synchronization helper.")
    parser.add_argument("--cam0-video", default=str(DEFAULT_CAM0_VIDEO))
    parser.add_argument("--cam1-video", default=str(DEFAULT_CAM1_VIDEO))
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract full-resolution frames into cam_1/cam_2 folders")
    extract.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    extract.add_argument("--every", type=int, default=1, help="Extract every Nth frame")
    extract.add_argument("--max-frames", type=int, default=None)
    extract.add_argument("--jpeg-quality", type=int, default=92)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=command_extract)

    sheet = subparsers.add_parser("contact-sheet", help="Create a side-by-side review sheet for a candidate frame offset")
    sheet.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT / "contact_sheet.jpg"))
    sheet.add_argument("--start-frame", type=int, default=0)
    sheet.add_argument("--step", type=int, default=10)
    sheet.add_argument("--count", type=int, default=12)
    sheet.add_argument("--cols", type=int, default=3)
    sheet.add_argument("--tile-width", type=int, default=640)
    sheet.add_argument("--cam1-offset-frames", type=int, default=0)
    sheet.set_defaults(func=command_contact_sheet)

    pair = subparsers.add_parser("pair", help="Create synchronized Kalibr cam0/cam1 frame folders from chosen start frames")
    pair.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "paired_frames"))
    pair.add_argument("--cam0-start-frame", type=int, required=True)
    pair.add_argument("--cam1-start-frame", type=int, required=True)
    pair.add_argument("--pairs", type=int, default=None)
    pair.add_argument("--frame-step", type=int, default=1, help="Use every Nth synchronized source frame")
    pair.add_argument("--timestamp-fps", type=float, default=None)
    pair.add_argument("--start-ns", type=int, default=1700000000000000000)
    pair.add_argument("--overwrite", action="store_true")
    pair.set_defaults(func=command_pair)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.cam0_video = _check_video(args.cam0_video)
    args.cam1_video = _check_video(args.cam1_video)
    args.func(args)


if __name__ == "__main__":
    main()
