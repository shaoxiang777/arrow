#!/usr/bin/env python3
"""Recover metric scale by aligning Stella landmarks with same-frame stereo depth."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import msgpack
import numpy as np
import yaml

from estimate_stereo_slam_scale import (
    GREEN,
    RESET,
    VirtualStereoRectifier,
    associate_nearest_unique,
    calibration_for_slam_camera,
    load_calibration,
    load_tum_trajectory,
    quaternion_to_rotation,
    read_frame,
    triangulate_correspondences,
    triangulate_stereo,
    video_info,
    write_scaled_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def green(message):
    print(f"{GREEN}{message}{RESET}")


def erp_to_perspective(points, erp_size, fov, theta, phi, output_size):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    erp_width, erp_height = erp_size
    width, height = output_size

    longitude = (points[:, 0] / (erp_width - 1.0) - 0.5) * (2.0 * np.pi)
    latitude = (points[:, 1] / (erp_height - 1.0) - 0.5) * np.pi
    rays_erp = np.column_stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        )
    )

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rotation_yaw, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    rotation_pitch, _ = cv2.Rodrigues(
        (rotation_yaw @ x_axis) * np.radians(phi)
    )
    rotation = rotation_pitch @ rotation_yaw
    rays_view = rays_erp @ rotation

    focal = 0.5 * width / np.tan(0.5 * np.radians(fov))
    pixels = np.column_stack(
        (
            focal * rays_view[:, 0] / rays_view[:, 2] + (width - 1.0) * 0.5,
            focal * rays_view[:, 1] / rays_view[:, 2] + (height - 1.0) * 0.5,
        )
    )
    valid = (
        (rays_view[:, 2] > 0.0)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return pixels, valid


def decode_descriptors(rows):
    if not rows:
        return np.empty((0, 32), dtype=np.uint8)
    words = np.asarray(rows, dtype="<u4").reshape(-1, 8)
    return words.view(np.uint8).reshape(-1, 32)


def load_map(path):
    with Path(path).open("rb") as stream:
        data = msgpack.Unpacker(stream, raw=False).unpack()
    keyframes = []
    for keyframe_id, keyframe in data["keyframes"].items():
        keyframes.append(
            {
                "id": int(keyframe_id),
                "timestamp": float(keyframe["ts"]),
                "rotation_cw": quaternion_to_rotation(keyframe["rot_cw"]),
                "translation_cw": np.asarray(keyframe["trans_cw"], dtype=np.float64).reshape(3),
                "keypoints": np.asarray(
                    [item["pt"] for item in keyframe["undist_keypts"]], dtype=np.float64
                ),
                "descriptors": decode_descriptors(keyframe["descs"]),
                "landmark_ids": np.asarray(keyframe["lm_ids"], dtype=np.int64),
            }
        )
    keyframes.sort(key=lambda item: item["timestamp"])
    return data, keyframes


def get_landmark(landmarks, landmark_id):
    value = landmarks.get(int(landmark_id))
    if value is None:
        value = landmarks.get(str(int(landmark_id)))
    return value


def robust_values(values, minimum_count):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < minimum_count:
        return values[:0]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(3.0 * 1.4826 * mad, 0.10 * median)
    return values[np.abs(values - median) <= threshold]


def triangulate_map_points_lk(left_image, right_image, map_points, rectifier, args):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    start = np.asarray(map_points, dtype=np.float32).reshape(-1, 1, 2)
    parameters = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005),
    }
    right_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, start, None, **parameters
    )
    if right_points is None or forward_status is None:
        return None
    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        right_gray, left_gray, right_points, None, **parameters
    )
    if back_points is None or backward_status is None:
        return None

    left_points = start[:, 0]
    right_points = right_points[:, 0]
    forward_backward_error = np.linalg.norm(back_points[:, 0] - left_points, axis=1)
    height, width = left_gray.shape
    valid = (
        forward_status[:, 0].astype(bool)
        & backward_status[:, 0].astype(bool)
        & (forward_backward_error <= args.max_lk_forward_backward_error)
        & (np.abs(left_points[:, 1] - right_points[:, 1]) <= args.epipolar_threshold)
        & (right_points[:, 0] >= 0.0)
        & (right_points[:, 0] < width)
        & (right_points[:, 1] >= 0.0)
        & (right_points[:, 1] < height)
    )
    candidate_indices = np.flatnonzero(valid)
    points_3d, triangulation_valid = triangulate_correspondences(
        left_points[valid], right_points[valid], rectifier, args
    )
    candidate_indices = candidate_indices[triangulation_valid]
    return {
        "raw_matches": len(map_points),
        "points_3d": points_3d,
        "left_points": left_points[candidate_indices],
        "right_points": right_points[candidate_indices],
        "map_indices": candidate_indices,
        "left_descriptors": np.empty((len(candidate_indices), 32), dtype=np.uint8),
    }


def draw_result(left, right, stereo, map_points, stereo_indices, path, title):
    canvas = np.hstack((left.copy(), right.copy()))
    offset = left.shape[1]
    selected = np.linspace(0, len(stereo_indices) - 1, min(len(stereo_indices), 100), dtype=int)
    for order in selected:
        stereo_index = stereo_indices[order]
        point_left = tuple(np.rint(map_points[order]).astype(int))
        point_right = tuple(
            np.rint(stereo["right_points"][stereo_index] + np.array([offset, 0])).astype(int)
        )
        cv2.circle(canvas, point_left, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, point_right, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, point_left, point_right, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def process_keyframe(
    keyframe,
    frame,
    left_capture,
    right_capture,
    args,
    calibration,
    rectifier,
    landmarks,
):
    result = {
        "keyframe_id": keyframe["id"],
        "frame": frame,
        "stereo_points": 0,
        "visible_map_points": 0,
        "associated_points": 0,
        "descriptor_points": 0,
        "bearing_points": 0,
        "scale_points": 0,
        "scale_m_per_slam_unit": math.nan,
        "relative_iqr": math.nan,
        "status": "rejected",
        "reason": "",
    }
    left_erp = read_frame(left_capture, frame, args.slam_video)
    other_frame = frame + args.other_frame_offset
    right_erp = read_frame(right_capture, other_frame, args.other_video)
    left_rectified = rectifier.left(left_erp)
    right_rectified = rectifier.right(right_erp)
    stereo = None
    if args.stereo_method == "orb":
        stereo = triangulate_stereo(left_rectified, right_rectified, rectifier, args)
        result["stereo_points"] = len(stereo["points_3d"])
        if len(stereo["points_3d"]) < args.min_stereo_points:
            result["reason"] = "too few stereo points"
            return result, None

    valid_landmarks = []
    valid_keypoints = []
    valid_descriptors = []
    for index, landmark_id in enumerate(keyframe["landmark_ids"]):
        if landmark_id < 0:
            continue
        landmark = get_landmark(landmarks, landmark_id)
        if landmark is None:
            continue
        valid_landmarks.append(np.asarray(landmark["pos_w"], dtype=np.float64))
        valid_keypoints.append(keyframe["keypoints"][index])
        valid_descriptors.append(keyframe["descriptors"][index])
    if not valid_landmarks:
        result["reason"] = "keyframe has no mapped landmarks"
        return result, None

    raw_perspective, in_view = erp_to_perspective(
        np.asarray(valid_keypoints),
        (left_erp.shape[1], left_erp.shape[0]),
        args.fov,
        args.theta,
        args.phi,
        (args.view_width, args.view_height),
    )
    map_world = np.asarray(valid_landmarks)[in_view]
    map_descriptors = np.asarray(valid_descriptors, dtype=np.uint8)[in_view]
    raw_perspective = raw_perspective[in_view]
    result["visible_map_points"] = len(raw_perspective)
    if not len(raw_perspective):
        result["reason"] = "no mapped landmarks in calibrated virtual view"
        return result, None

    map_rectified = cv2.undistortPoints(
        raw_perspective.reshape(-1, 1, 2),
        calibration["k0"],
        calibration["d0"],
        R=rectifier.r0,
        P=rectifier.p0,
    ).reshape(-1, 2)
    if args.stereo_method == "lk":
        stereo = triangulate_map_points_lk(
            left_rectified, right_rectified, map_rectified, rectifier, args
        )
        if stereo is None:
            result["reason"] = "same-frame stereo LK failed"
            return result, None
        map_indices = stereo["map_indices"]
        stereo_indices = np.arange(len(stereo["points_3d"]), dtype=np.int64)
    else:
        map_indices, stereo_indices = associate_nearest_unique(
            map_rectified, stereo["left_points"], args.association_radius
        )
    result["stereo_points"] = len(stereo["points_3d"])
    result["associated_points"] = len(map_indices)
    if not len(map_indices):
        result["reason"] = "no spatial associations between Stella and stereo points"
        return result, None

    if args.stereo_method == "orb":
        descriptor_distances = np.asarray(
            [
                cv2.norm(map_descriptors[map_index], stereo["left_descriptors"][stereo_index], cv2.NORM_HAMMING)
                for map_index, stereo_index in zip(map_indices, stereo_indices)
            ]
        )
        descriptor_valid = descriptor_distances <= args.max_descriptor_distance
        map_indices = map_indices[descriptor_valid]
        stereo_indices = stereo_indices[descriptor_valid]
    result["descriptor_points"] = len(map_indices)
    if not len(map_indices):
        result["reason"] = "spatial associations failed descriptor validation"
        return result, None

    slam_camera_points = (
        keyframe["rotation_cw"] @ map_world[map_indices].T
    ).T + keyframe["translation_cw"]
    slam_rectified_points = (rectifier.r0 @ slam_camera_points.T).T
    metric_points = stereo["points_3d"][stereo_indices]
    slam_ranges = np.linalg.norm(slam_rectified_points, axis=1)
    metric_ranges = np.linalg.norm(metric_points, axis=1)
    cosines = np.sum(slam_rectified_points * metric_points, axis=1) / np.maximum(
        slam_ranges * metric_ranges, 1e-12
    )
    bearing_errors = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    bearing_valid = (
        (bearing_errors <= args.max_bearing_error)
        & (slam_rectified_points[:, 2] > 0.0)
        & (slam_ranges > args.min_slam_range)
    )
    map_indices = map_indices[bearing_valid]
    stereo_indices = stereo_indices[bearing_valid]
    slam_ranges = slam_ranges[bearing_valid]
    metric_ranges = metric_ranges[bearing_valid]
    result["bearing_points"] = len(map_indices)

    ratios = metric_ranges / slam_ranges
    ratios = ratios[(ratios >= args.min_scale) & (ratios <= args.max_scale)]
    ratios = robust_values(ratios, args.min_scale_points)
    result["scale_points"] = len(ratios)
    if len(ratios) < args.min_scale_points:
        result["reason"] = "too few consistent landmark/stereo scale points"
        return result, None

    scale = float(np.median(ratios))
    q25, q75 = np.percentile(ratios, (25, 75))
    result["scale_m_per_slam_unit"] = scale
    result["relative_iqr"] = float((q75 - q25) / scale)
    if result["relative_iqr"] > args.max_point_relative_iqr:
        result["reason"] = "within-keyframe scale spread is too large"
        return result, None
    result["status"] = "quality_pass"
    result["reason"] = ""
    visualization = {
        "left": left_rectified,
        "right": right_rectified,
        "stereo": stereo,
        "map_points": map_rectified[map_indices],
        "stereo_indices": stereo_indices,
    }
    return result, visualization


def write_scaled_map(path, map_data, scale):
    for keyframe in map_data["keyframes"].values():
        keyframe["trans_cw"] = [float(value) * scale for value in keyframe["trans_cw"]]
        if keyframe.get("depths"):
            keyframe["depths"] = [float(value) * scale for value in keyframe["depths"]]
    for landmark in map_data["landmarks"].values():
        landmark["pos_w"] = [float(value) * scale for value in landmark["pos_w"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        msgpack.pack(map_data, stream, use_bin_type=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale a Stella map using same-frame stereo landmark ranges."
    )
    parser.add_argument("scene", nargs="?", default="jia")
    parser.add_argument("--slam-video", type=Path, default=None)
    parser.add_argument("--other-video", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "data/ricoh/cali_opencv/kalibr_virtual_sync_557_554_fov90/checkerboard_camchain.yaml")
    parser.add_argument("--slam-camera", choices=("cam0", "cam1"), default="cam1")
    parser.add_argument("--other-frame-offset", type=int, default=0)
    parser.add_argument("--trajectory-first-frame", type=int, default=0)
    parser.add_argument("--slam-frame-skip", type=int, default=2)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--view-width", type=int, default=None)
    parser.add_argument("--view-height", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--stereo-method", choices=("lk", "orb"), default="lk")
    parser.add_argument("--match-ratio", type=float, default=0.75)
    parser.add_argument("--epipolar-threshold", type=float, default=1.5)
    parser.add_argument("--min-disparity", type=float, default=1.5)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=12.0)
    parser.add_argument("--max-stereo-reprojection-error", type=float, default=1.25)
    parser.add_argument("--min-stereo-points", type=int, default=30)
    parser.add_argument("--association-radius", type=float, default=4.0)
    parser.add_argument("--max-lk-forward-backward-error", type=float, default=0.75)
    parser.add_argument("--max-descriptor-distance", type=float, default=100.0)
    parser.add_argument("--max-bearing-error", type=float, default=3.0)
    parser.add_argument("--min-slam-range", type=float, default=0.1)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--min-scale-points", type=int, default=6)
    parser.add_argument("--min-keyframes", type=int, default=5)
    parser.add_argument("--max-point-relative-iqr", type=float, default=0.35)
    parser.add_argument("--max-keyframe-relative-iqr", type=float, default=0.20)
    parser.add_argument("--visualization-limit", type=int, default=12)
    parser.add_argument("--output-trajectory", type=Path, default=None)
    parser.add_argument("--output-map", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--visualization-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args):
    scene = REPO_ROOT / "data" / "arrow360" / args.scene
    defaults = {
        "slam_video": scene / "video_crop.mp4",
        "map": scene / "my_saved_map.msg",
        "trajectory": scene / "frame_trajectory.txt",
        "output_trajectory": scene / "frame_trajectory_metric.txt",
        "output_map": scene / "my_saved_map_metric.msg",
        "report": scene / "stereo_map_scale_keyframes.csv",
        "summary": scene / "stereo_map_scale.yaml",
        "visualization_dir": scene / "stereo_map_scale_visualization",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("slam_video", "other_video", "map", "trajectory", "calibration"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")


def main():
    args = parse_args()
    resolve_paths(args)
    green("[1/4] Loading calibration, synchronized videos, trajectory, and Stella map")
    calibration = calibration_for_slam_camera(load_calibration(args.calibration), args.slam_camera)
    if args.view_width is None:
        args.view_width = calibration["resolution"][0]
    if args.view_height is None:
        args.view_height = calibration["resolution"][1]
    left_info = video_info(args.slam_video)
    right_info = video_info(args.other_video)
    poses = load_tum_trajectory(args.trajectory, left_info["fps"], args.trajectory_first_frame)
    observed_skip = float(np.median(np.diff([pose["frame"] for pose in poses])))
    if not math.isclose(observed_skip, args.slam_frame_skip, abs_tol=0.25):
        raise ValueError(f"observed Stella frame step {observed_skip:.1f}, expected {args.slam_frame_skip}")
    map_data, keyframes = load_map(args.map)
    first_timestamp = poses[0]["timestamp"]
    for keyframe in keyframes:
        keyframe["frame"] = int(round((keyframe["timestamp"] - first_timestamp) * left_info["fps"]))
    rectifier = VirtualStereoRectifier(
        calibration, args.fov, args.theta, args.phi, args.view_width, args.view_height
    )
    print(f"baseline: {calibration['baseline_m']:.6f} m; keyframes: {len(keyframes)}")
    print(f"observed Stella raw-frame step: {observed_skip:.1f}")

    green("[2/4] Aligning same-frame Stella landmarks with metric stereo points")
    left_capture = cv2.VideoCapture(str(args.slam_video))
    right_capture = cv2.VideoCapture(str(args.other_video))
    results = []
    visualization_count = 0
    try:
        for order, keyframe in enumerate(keyframes, start=1):
            frame = keyframe["frame"]
            other_frame = frame + args.other_frame_offset
            if frame < 0 or frame >= left_info["frames"] or other_frame < 0 or other_frame >= right_info["frames"]:
                continue
            result, visualization = process_keyframe(
                keyframe, frame, left_capture, right_capture, args, calibration, rectifier, map_data["landmarks"]
            )
            results.append(result)
            if visualization is not None and visualization_count < args.visualization_limit:
                draw_result(
                    visualization["left"], visualization["right"], visualization["stereo"],
                    visualization["map_points"], visualization["stereo_indices"],
                    args.visualization_dir / f"keyframe_{keyframe['id']:04d}_frame_{frame:06d}.jpg",
                    f"KF {keyframe['id']} frame {frame}: {result['scale_points']} points, scale={result['scale_m_per_slam_unit']:.4f}",
                )
                visualization_count += 1
            detail = (
                f"scale={result['scale_m_per_slam_unit']:.4f}, points={result['scale_points']}"
                if result["status"] == "quality_pass" else result["reason"]
            )
            print(f"[{order:02d}/{len(keyframes):02d}] KF {keyframe['id']} frame {frame}: {detail}")
    finally:
        left_capture.release()
        right_capture.release()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    green("[3/4] Checking scale consistency across keyframes")
    accepted = [row for row in results if row["status"] == "quality_pass"]
    if len(accepted) < args.min_keyframes:
        raise RuntimeError(f"only {len(accepted)} reliable keyframes; need {args.min_keyframes}")
    scales = np.asarray([row["scale_m_per_slam_unit"] for row in accepted], dtype=np.float64)
    scales = robust_values(scales, args.min_keyframes)
    if len(scales) < args.min_keyframes:
        raise RuntimeError(f"only {len(scales)} keyframes remain after robust filtering")
    scale = float(np.median(scales))
    q25, q75 = np.percentile(scales, (25, 75))
    relative_iqr = float((q75 - q25) / scale)
    if relative_iqr > args.max_keyframe_relative_iqr:
        raise RuntimeError(
            f"keyframe scales are inconsistent: relative IQR={relative_iqr:.3f}, "
            f"limit={args.max_keyframe_relative_iqr:.3f}"
        )

    green("[4/4] Writing metric trajectory, metric map, and diagnostics")
    write_scaled_trajectory(args.output_trajectory, poses, scale)
    write_scaled_map(args.output_map, map_data, scale)
    summary = {
        "status": "accepted",
        "scale_m_per_slam_unit": scale,
        "scale_statistics": {
            "reliable_keyframes": len(accepted),
            "robust_keyframes": len(scales),
            "q25": float(q25),
            "q75": float(q75),
            "relative_iqr": relative_iqr,
        },
        "calibration": str(args.calibration.resolve()),
        "baseline_m": calibration["baseline_m"],
        "slam_frame_skip": args.slam_frame_skip,
        "inputs": {
            "slam_video": str(args.slam_video.resolve()),
            "other_video": str(args.other_video.resolve()),
            "map": str(args.map.resolve()),
            "trajectory": str(args.trajectory.resolve()),
            "slam_camera": args.slam_camera,
            "other_frame_offset": args.other_frame_offset,
            "stereo_method": args.stereo_method,
            "max_depth_m": args.max_depth,
        },
        "outputs": {
            "trajectory": str(args.output_trajectory.resolve()),
            "map": str(args.output_map.resolve()),
            "report": str(args.report.resolve()),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)
    green(f"Scale: {scale:.6f} m / SLAM unit")
    green(f"Keyframe IQR: [{q25:.6f}, {q75:.6f}], relative IQR={relative_iqr:.3f}")
    green(f"Metric trajectory: {args.output_trajectory}")
    green(f"Metric map: {args.output_map}")


if __name__ == "__main__":
    main()
