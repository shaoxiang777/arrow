#!/usr/bin/env python3
"""Estimate a metric scale for a monocular Stella VSLAM trajectory.

For each sampled time interval this script:
  1. reconstructs metric 3D points from a synchronized stereo pair,
  2. tracks the left-image points forward with pyramidal LK optical flow,
  3. estimates metric camera motion with PnP RANSAC, and
  4. compares it with Stella's translation over the same timestamps.

The robust median of the per-interval ratios is applied to a copy of the TUM
trajectory. The original trajectory and Stella map are never modified.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.Equirec2Perspec as E2P


GREEN = "\033[92m"
RESET = "\033[0m"


def green(message):
    print(f"{GREEN}{message}{RESET}")


def camera_matrix(intrinsics):
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_calibration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    try:
        cam0 = data["cam0"]
        cam1 = data["cam1"]
        k0 = camera_matrix(cam0["intrinsics"])
        k1 = camera_matrix(cam1["intrinsics"])
        d0 = np.asarray(cam0["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        d1 = np.asarray(cam1["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
        resolution0 = tuple(int(value) for value in cam0["resolution"])
        resolution1 = tuple(int(value) for value in cam1["resolution"])
        transform = np.asarray(cam1["T_cn_cnm1"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkerboard camchain file: {path}") from exc

    if resolution0 != resolution1:
        raise ValueError(f"calibration resolutions differ: cam0={resolution0}, cam1={resolution1}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_cn_cnm1 must be 4x4, got {transform.shape}")

    rotation = transform[:3, :3]
    translation = transform[:3, 3].reshape(3, 1)
    baseline = float(np.linalg.norm(translation))
    if baseline <= 0.0:
        raise ValueError("stereo baseline must be positive")

    return {
        "k0": k0,
        "d0": d0,
        "k1": k1,
        "d1": d1,
        "resolution": resolution0,
        "rotation": rotation,
        "translation": translation,
        "baseline_m": baseline,
    }


def calibration_for_slam_camera(calibration, slam_camera):
    if slam_camera == "cam0":
        return calibration
    if slam_camera != "cam1":
        raise ValueError(f"unsupported Stella camera: {slam_camera}")

    rotation_10 = calibration["rotation"].T
    translation_10 = -rotation_10 @ calibration["translation"]
    return {
        "k0": calibration["k1"],
        "d0": calibration["d1"],
        "k1": calibration["k0"],
        "d1": calibration["d0"],
        "resolution": calibration["resolution"],
        "rotation": rotation_10,
        "translation": translation_10,
        "baseline_m": calibration["baseline_m"],
    }


def load_tum_trajectory(path, fps, first_frame):
    poses = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TUM fields, got {len(fields)}")
            values = [float(value) for value in fields]
            poses.append(
                {
                    "timestamp": values[0],
                    "position": np.asarray(values[1:4], dtype=np.float64),
                    "quaternion": np.asarray(values[4:8], dtype=np.float64),
                }
            )

    if len(poses) < 2:
        raise ValueError(f"trajectory needs at least two poses: {path}")

    first_timestamp = poses[0]["timestamp"]
    previous_frame = None
    for pose in poses:
        pose["frame"] = int(first_frame + round((pose["timestamp"] - first_timestamp) * fps))
        if previous_frame is not None and pose["frame"] <= previous_frame:
            raise ValueError("trajectory timestamps do not map to strictly increasing video frames")
        previous_frame = pose["frame"]
    return poses


def quaternion_to_rotation(quaternion_xyzw):
    x, y, z, w = [float(value) for value in quaternion_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero-length trajectory quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        info = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if info["fps"] <= 0.0 or info["frames"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError(f"invalid video metadata: {path}: {info}")
    return info


def read_frame(capture, frame_index, path):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {path}")
    return frame


def read_next_frame(capture, frame_index, path):
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {path}")
    return frame


class VirtualStereoRectifier:
    def __init__(self, calibration, fov, theta, phi, width, height):
        self.fov = float(fov)
        self.theta = float(theta)
        self.phi = float(phi)
        self.width = int(width)
        self.height = int(height)
        image_size = (self.width, self.height)

        if image_size != calibration["resolution"]:
            raise ValueError(
                "virtual view resolution must match calibration: "
                f"requested={image_size}, calibrated={calibration['resolution']}"
            )

        rectification = cv2.stereoRectify(
            calibration["k0"],
            calibration["d0"],
            calibration["k1"],
            calibration["d1"],
            image_size,
            calibration["rotation"],
            calibration["translation"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        self.r0, self.r1, self.p0, self.p1 = rectification[:4]
        self.map0 = cv2.initUndistortRectifyMap(
            calibration["k0"], calibration["d0"], self.r0, self.p0, image_size, cv2.CV_32FC1
        )
        self.map1 = cv2.initUndistortRectifyMap(
            calibration["k1"], calibration["d1"], self.r1, self.p1, image_size, cv2.CV_32FC1
        )
        self.rectified_k = self.p0[:, :3].copy()

    def perspective(self, erp_frame):
        equ = E2P.Equirectangular(erp_frame)
        return equ.GetPerspective(self.fov, self.theta, self.phi, self.height, self.width)

    def left(self, erp_frame):
        perspective = self.perspective(erp_frame)
        return cv2.remap(perspective, self.map0[0], self.map0[1], cv2.INTER_LINEAR)

    def right(self, erp_frame):
        perspective = self.perspective(erp_frame)
        return cv2.remap(perspective, self.map1[0], self.map1[1], cv2.INTER_LINEAR)


def ratio_matches(descriptors0, descriptors1, ratio):
    if descriptors0 is None or descriptors1 is None or len(descriptors0) < 2 or len(descriptors1) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward_knn = matcher.knnMatch(descriptors0, descriptors1, k=2)
    reverse_knn = matcher.knnMatch(descriptors1, descriptors0, k=2)

    reverse = {}
    for candidates in reverse_knn:
        if len(candidates) == 2 and candidates[0].distance < ratio * candidates[1].distance:
            reverse[candidates[0].queryIdx] = candidates[0].trainIdx

    matches = []
    for candidates in forward_knn:
        if len(candidates) != 2:
            continue
        best, second = candidates
        if best.distance < ratio * second.distance and reverse.get(best.trainIdx) == best.queryIdx:
            matches.append(best)
    return matches


def project_points(projection, points_4d):
    homogeneous = projection @ points_4d
    valid = np.abs(homogeneous[2]) > 1e-12
    pixels = np.full((points_4d.shape[1], 2), np.nan, dtype=np.float64)
    pixels[valid] = (homogeneous[:2, valid] / homogeneous[2:3, valid]).T
    return pixels, homogeneous[2], valid


def triangulate_correspondences(left_points, right_points, rectifier, args):
    left_points = np.asarray(left_points, dtype=np.float64).reshape(-1, 2)
    right_points = np.asarray(right_points, dtype=np.float64).reshape(-1, 2)
    if len(left_points) == 0 or len(left_points) != len(right_points):
        return np.empty((0, 3), dtype=np.float64), np.zeros(len(left_points), dtype=bool)

    points_4d = cv2.triangulatePoints(rectifier.p0, rectifier.p1, left_points.T, right_points.T)
    finite_w = np.abs(points_4d[3]) > 1e-12
    points_4d[:, finite_w] /= points_4d[3:4, finite_w]

    projected0, depth0, valid_projection0 = project_points(rectifier.p0, points_4d)
    projected1, depth1, valid_projection1 = project_points(rectifier.p1, points_4d)
    reprojection0 = np.linalg.norm(projected0 - left_points, axis=1)
    reprojection1 = np.linalg.norm(projected1 - right_points, axis=1)
    disparity = np.abs(left_points[:, 0] - right_points[:, 0])
    points_3d = points_4d[:3].T

    valid = (
        finite_w
        & valid_projection0
        & valid_projection1
        & np.all(np.isfinite(points_3d), axis=1)
        & (np.abs(left_points[:, 1] - right_points[:, 1]) <= args.epipolar_threshold)
        & (depth0 > 0.0)
        & (depth1 > 0.0)
        & (points_3d[:, 2] >= args.min_depth)
        & (points_3d[:, 2] <= args.max_depth)
        & (disparity >= args.min_disparity)
        & (reprojection0 <= args.max_stereo_reprojection_error)
        & (reprojection1 <= args.max_stereo_reprojection_error)
    )
    return points_3d[valid].astype(np.float64), valid


def triangulate_stereo(left_image, right_image, rectifier, args):
    gray0 = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=args.max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        fastThreshold=12,
    )
    keypoints0, descriptors0 = orb.detectAndCompute(gray0, None)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    matches = ratio_matches(descriptors0, descriptors1, args.match_ratio)

    epipolar_matches = []
    for match in matches:
        point0 = keypoints0[match.queryIdx].pt
        point1 = keypoints1[match.trainIdx].pt
        if abs(point0[1] - point1[1]) <= args.epipolar_threshold:
            epipolar_matches.append(match)

    if not epipolar_matches:
        return {
            "raw_matches": len(matches),
            "points_3d": np.empty((0, 3), dtype=np.float64),
            "left_points": np.empty((0, 2), dtype=np.float32),
            "right_points": np.empty((0, 2), dtype=np.float32),
            "left_descriptors": np.empty((0, 32), dtype=np.uint8),
        }

    left_points = np.asarray([keypoints0[m.queryIdx].pt for m in epipolar_matches], dtype=np.float64)
    right_points = np.asarray([keypoints1[m.trainIdx].pt for m in epipolar_matches], dtype=np.float64)
    left_descriptors = np.asarray(
        [descriptors0[m.queryIdx] for m in epipolar_matches], dtype=np.uint8
    )
    points_3d, valid = triangulate_correspondences(left_points, right_points, rectifier, args)

    return {
        "raw_matches": len(matches),
        "epipolar_matches": len(epipolar_matches),
        "points_3d": points_3d,
        "left_points": left_points[valid].astype(np.float32),
        "right_points": right_points[valid].astype(np.float32),
        "left_descriptors": left_descriptors[valid],
    }


def track_points(capture, left_video, rectifier, anchor_frame, target_frame, anchor_image, points_3d, points_2d, args):
    previous_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    current_points = points_2d.reshape(-1, 1, 2).astype(np.float32)
    original_points = points_2d.copy()
    tracked_3d = points_3d.copy()
    target_image = anchor_image

    lk_parameters = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    for frame_index in range(anchor_frame + 1, target_frame + 1):
        erp_frame = read_next_frame(capture, frame_index, left_video)
        target_image = rectifier.left(erp_frame)
        current_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, current_points, None, **lk_parameters
        )
        if next_points is None or status_forward is None:
            return target_image, tracked_3d[:0], original_points[:0], original_points[:0]

        back_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_points, None, **lk_parameters
        )
        if back_points is None or status_backward is None:
            return target_image, tracked_3d[:0], original_points[:0], original_points[:0]

        forward_backward_error = np.linalg.norm(current_points[:, 0] - back_points[:, 0], axis=1)
        coordinates = next_points[:, 0]
        h, w = current_gray.shape
        valid = (
            status_forward[:, 0].astype(bool)
            & status_backward[:, 0].astype(bool)
            & np.all(np.isfinite(coordinates), axis=1)
            & (forward_backward_error <= args.max_lk_forward_backward_error)
            & (coordinates[:, 0] >= 1.0)
            & (coordinates[:, 0] < w - 1.0)
            & (coordinates[:, 1] >= 1.0)
            & (coordinates[:, 1] < h - 1.0)
        )

        current_points = next_points[valid].reshape(-1, 1, 2)
        tracked_3d = tracked_3d[valid]
        original_points = original_points[valid]
        previous_gray = current_gray
        if len(current_points) < args.min_tracked_points:
            return target_image, tracked_3d[:0], original_points[:0], original_points[:0]

    return target_image, tracked_3d, original_points, current_points[:, 0]


def associate_nearest_unique(query_points, reference_points, maximum_distance):
    query_points = np.asarray(query_points, dtype=np.float64).reshape(-1, 2)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    if len(query_points) == 0 or len(reference_points) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    squared_distances = np.sum(
        np.square(query_points[:, None, :] - reference_points[None, :, :]), axis=2
    )
    nearest_reference = np.argmin(squared_distances, axis=1)
    nearest_distance = np.sqrt(squared_distances[np.arange(len(query_points)), nearest_reference])
    candidates = np.flatnonzero(nearest_distance <= maximum_distance)
    candidates = candidates[np.argsort(nearest_distance[candidates])]

    used_references = set()
    accepted_queries = []
    accepted_references = []
    for query_index in candidates:
        reference_index = int(nearest_reference[query_index])
        if reference_index in used_references:
            continue
        used_references.add(reference_index)
        accepted_queries.append(int(query_index))
        accepted_references.append(reference_index)
    return np.asarray(accepted_queries, dtype=np.int64), np.asarray(accepted_references, dtype=np.int64)


def lk_track_step(previous_gray, current_gray, current_points, args):
    lk_parameters = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, current_points, None, **lk_parameters
    )
    if next_points is None or status_forward is None:
        return None, np.zeros(len(current_points), dtype=bool)
    back_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, next_points, None, **lk_parameters
    )
    if back_points is None or status_backward is None:
        return None, np.zeros(len(current_points), dtype=bool)

    coordinates = next_points[:, 0]
    forward_backward_error = np.linalg.norm(current_points[:, 0] - back_points[:, 0], axis=1)
    height, width = current_gray.shape
    valid = (
        status_forward[:, 0].astype(bool)
        & status_backward[:, 0].astype(bool)
        & np.all(np.isfinite(coordinates), axis=1)
        & (forward_backward_error <= args.max_lk_forward_backward_error)
        & (coordinates[:, 0] >= 1.0)
        & (coordinates[:, 0] < width - 1.0)
        & (coordinates[:, 1] >= 1.0)
        & (coordinates[:, 1] < height - 1.0)
    )
    return next_points, valid


def track_stereo_points(
    left_capture,
    right_capture,
    left_video,
    right_video,
    rectifier,
    anchor_frame,
    target_frame,
    left_anchor_image,
    right_anchor_image,
    points_3d,
    left_points,
    right_points,
    args,
):
    previous_left_gray = cv2.cvtColor(left_anchor_image, cv2.COLOR_BGR2GRAY)
    previous_right_gray = cv2.cvtColor(right_anchor_image, cv2.COLOR_BGR2GRAY)
    current_left_points = left_points.reshape(-1, 1, 2).astype(np.float32)
    current_right_points = right_points.reshape(-1, 1, 2).astype(np.float32)
    anchor_left_points = left_points.copy()
    anchor_right_points = right_points.copy()
    anchor_points_3d = points_3d.copy()
    target_left_image = left_anchor_image
    target_right_image = right_anchor_image

    for frame_index in range(anchor_frame + 1, target_frame + 1):
        left_erp = read_next_frame(left_capture, frame_index, left_video)
        right_erp = read_next_frame(right_capture, frame_index, right_video)
        target_left_image = rectifier.left(left_erp)
        target_right_image = rectifier.right(right_erp)
        current_left_gray = cv2.cvtColor(target_left_image, cv2.COLOR_BGR2GRAY)
        current_right_gray = cv2.cvtColor(target_right_image, cv2.COLOR_BGR2GRAY)

        next_left_points, valid_left = lk_track_step(
            previous_left_gray, current_left_gray, current_left_points, args
        )
        next_right_points, valid_right = lk_track_step(
            previous_right_gray, current_right_gray, current_right_points, args
        )
        if next_left_points is None or next_right_points is None:
            valid = np.zeros(len(anchor_points_3d), dtype=bool)
        else:
            valid = valid_left & valid_right

        current_left_points = next_left_points[valid].reshape(-1, 1, 2) if next_left_points is not None else current_left_points[:0]
        current_right_points = next_right_points[valid].reshape(-1, 1, 2) if next_right_points is not None else current_right_points[:0]
        anchor_points_3d = anchor_points_3d[valid]
        anchor_left_points = anchor_left_points[valid]
        anchor_right_points = anchor_right_points[valid]
        previous_left_gray = current_left_gray
        previous_right_gray = current_right_gray
        if len(anchor_points_3d) < args.min_tracked_points:
            return {
                "left_image": target_left_image,
                "right_image": target_right_image,
                "anchor_points_3d": anchor_points_3d[:0],
                "anchor_left_points": anchor_left_points[:0],
                "anchor_right_points": anchor_right_points[:0],
                "target_left_points": anchor_left_points[:0],
                "target_right_points": anchor_right_points[:0],
            }

    return {
        "left_image": target_left_image,
        "right_image": target_right_image,
        "anchor_points_3d": anchor_points_3d,
        "anchor_left_points": anchor_left_points,
        "anchor_right_points": anchor_right_points,
        "target_left_points": current_left_points[:, 0],
        "target_right_points": current_right_points[:, 0],
    }


def estimate_pnp(points_3d, points_2d, camera_matrix_rectified, args):
    if len(points_3d) < args.min_tracked_points:
        return None

    ok, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera_matrix_rectified,
        None,
        iterationsCount=300,
        reprojectionError=args.pnp_reprojection_threshold,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < args.min_pnp_inliers:
        return None

    inlier_indices = inliers[:, 0]
    ok, rotation_vector, translation = cv2.solvePnP(
        points_3d[inlier_indices],
        points_2d[inlier_indices],
        camera_matrix_rectified,
        None,
        rotation_vector,
        translation,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(
        points_3d[inlier_indices], rotation_vector, translation, camera_matrix_rectified, None
    )
    reprojection_errors = np.linalg.norm(projected[:, 0] - points_2d[inlier_indices], axis=1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return {
        "rotation": rotation,
        "translation": translation.reshape(3),
        "inliers": inlier_indices,
        "reprojection_rms": float(np.sqrt(np.mean(np.square(reprojection_errors)))),
    }


def estimate_scale_from_stella_motion(
    points_3d,
    points_2d,
    pnp_inliers,
    pnp_rotation,
    anchor_pose,
    target_pose,
    rectifier,
    args,
):
    rotation0_wc = quaternion_to_rotation(anchor_pose["quaternion"])
    rotation1_wc = quaternion_to_rotation(target_pose["quaternion"])
    rotation_cam = rotation1_wc.T @ rotation0_wc
    translation_cam = rotation1_wc.T @ (anchor_pose["position"] - target_pose["position"])

    rotation_rectified = rectifier.r0 @ rotation_cam @ rectifier.r0.T
    translation_rectified = rectifier.r0 @ translation_cam
    rotation_error_deg = rotation_angle_deg(pnp_rotation @ rotation_rectified.T)
    if rotation_error_deg > args.max_rotation_error:
        return None

    indices = np.asarray(pnp_inliers, dtype=np.int64).reshape(-1)
    source_points = points_3d[indices]
    image_points = points_2d[indices]
    rotated_points = (rotation_rectified @ source_points.T).T

    fx = rectifier.rectified_k[0, 0]
    fy = rectifier.rectified_k[1, 1]
    cx = rectifier.rectified_k[0, 2]
    cy = rectifier.rectified_k[1, 2]
    normalized_x = (image_points[:, 0] - cx) / fx
    normalized_y = (image_points[:, 1] - cy) / fy
    bx, by, bz = translation_rectified

    coefficient_x = normalized_x * bz - bx
    coefficient_y = normalized_y * bz - by
    target_x = rotated_points[:, 0] - normalized_x * rotated_points[:, 2]
    target_y = rotated_points[:, 1] - normalized_y * rotated_points[:, 2]
    denominator = np.square(coefficient_x) + np.square(coefficient_y)
    scale_values = (coefficient_x * target_x + coefficient_y * target_y) / np.maximum(
        denominator, 1e-12
    )

    valid = (
        np.isfinite(scale_values)
        & (denominator > args.min_scale_observability)
        & (scale_values >= args.min_scale)
        & (scale_values <= args.max_scale)
    )
    scale_values = scale_values[valid]
    if len(scale_values) < args.min_scale_point_inliers:
        return None

    median = float(np.median(scale_values))
    mad = float(np.median(np.abs(scale_values - median)))
    threshold = max(3.0 * 1.4826 * mad, 0.15 * median)
    scale_values = scale_values[np.abs(scale_values - median) <= threshold]
    if len(scale_values) < args.min_scale_point_inliers:
        return None

    scale = float(np.median(scale_values))
    transformed = rotated_points + scale * translation_rectified
    projected = np.column_stack(
        (
            fx * transformed[:, 0] / transformed[:, 2] + cx,
            fy * transformed[:, 1] / transformed[:, 2] + cy,
        )
    )
    reprojection_errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "scale": scale,
        "point_count": len(scale_values),
        "point_mad": float(np.median(np.abs(scale_values - scale))),
        "rotation_error_deg": rotation_error_deg,
        "reprojection_rms_px": float(np.sqrt(np.mean(np.square(reprojection_errors)))),
        "rotation": rotation_rectified,
    }


def fit_rigid_transform(source_points, target_points):
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def estimate_rigid_transform_ransac(source_points, target_points, args):
    if len(source_points) < args.min_rigid_inliers:
        return None

    rng = np.random.default_rng(20260713)
    best_inliers = None
    best_median_error = float("inf")
    for _ in range(args.rigid_ransac_iterations):
        sample = rng.choice(len(source_points), size=3, replace=False)
        if np.linalg.matrix_rank(source_points[sample] - source_points[sample].mean(axis=0)) < 2:
            continue
        rotation, translation = fit_rigid_transform(source_points[sample], target_points[sample])
        transformed = (rotation @ source_points.T).T + translation
        errors = np.linalg.norm(transformed - target_points, axis=1)
        inliers = np.flatnonzero(errors <= args.rigid_ransac_threshold)
        if len(inliers) == 0:
            continue
        median_error = float(np.median(errors[inliers]))
        if best_inliers is None or len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and median_error < best_median_error
        ):
            best_inliers = inliers
            best_median_error = median_error

    if best_inliers is None or len(best_inliers) < args.min_rigid_inliers:
        return None
    if len(best_inliers) / len(source_points) < args.min_rigid_inlier_ratio:
        return None

    rotation, translation = fit_rigid_transform(source_points[best_inliers], target_points[best_inliers])
    transformed = (rotation @ source_points.T).T + translation
    errors = np.linalg.norm(transformed - target_points, axis=1)
    inliers = np.flatnonzero(errors <= args.rigid_ransac_threshold)
    if len(inliers) < args.min_rigid_inliers:
        return None
    rotation, translation = fit_rigid_transform(source_points[inliers], target_points[inliers])
    transformed_inliers = (rotation @ source_points[inliers].T).T + translation
    residuals = np.linalg.norm(transformed_inliers - target_points[inliers], axis=1)
    return {
        "rotation": rotation,
        "translation": translation,
        "inliers": inliers,
        "residual_rms_m": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def select_segments(poses, start_frame, end_frame, temporal_gap, sample_step, max_target_delay):
    frames = np.asarray([pose["frame"] for pose in poses], dtype=np.int64)
    selected = []
    next_anchor = int(start_frame)
    for anchor_index, anchor_pose in enumerate(poses):
        anchor_frame = anchor_pose["frame"]
        if anchor_frame < next_anchor or anchor_frame < start_frame:
            continue
        if anchor_frame + temporal_gap > end_frame:
            break

        target_index = int(np.searchsorted(frames, anchor_frame + temporal_gap, side="left"))
        if target_index >= len(poses):
            break
        target_frame = poses[target_index]["frame"]
        if target_frame > end_frame or target_frame - anchor_frame > temporal_gap + max_target_delay:
            next_anchor = anchor_frame + sample_step
            continue

        selected.append((anchor_index, target_index))
        next_anchor = anchor_frame + sample_step
    return selected


def base_result(anchor_pose, target_pose, right_frame):
    return {
        "anchor_frame": anchor_pose["frame"],
        "target_frame": target_pose["frame"],
        "right_frame": right_frame,
        "status": "failed",
        "reason": "",
        "orb_matches": 0,
        "epipolar_matches": 0,
        "triangulated_points": 0,
        "tracked_points": 0,
        "target_triangulated_points": 0,
        "pnp_inliers": 0,
        "pnp_inlier_ratio": float("nan"),
        "pnp_reprojection_rms_px": float("nan"),
        "pnp_metric_distance_m": float("nan"),
        "rigid_inliers": 0,
        "rigid_inlier_ratio": float("nan"),
        "rigid_residual_rms_m": float("nan"),
        "scale_point_count": 0,
        "scale_point_mad": float("nan"),
        "rotation_matrix_error_deg": float("nan"),
        "stella_scale_reprojection_rms_px": float("nan"),
        "metric_distance_m": float("nan"),
        "slam_distance_units": float("nan"),
        "scale_m_per_slam_unit": float("nan"),
        "pnp_rotation_deg": float("nan"),
        "rigid_rotation_deg": float("nan"),
        "slam_rotation_deg": float("nan"),
        "scale_inlier": False,
    }


def process_segment(left_capture, right_capture, left_video, right_video, rectifier, anchor_pose, target_pose, args):
    anchor_frame = anchor_pose["frame"]
    target_frame = target_pose["frame"]
    right_frame = anchor_frame + args.right_frame_offset
    result = base_result(anchor_pose, target_pose, right_frame)
    visualization = None

    if right_frame < 0:
        result["reason"] = "right frame is negative; check --right-frame-offset"
        return result, visualization

    left_erp = read_frame(left_capture, anchor_frame, left_video)
    right_erp = read_frame(right_capture, right_frame, right_video)
    left_rectified = rectifier.left(left_erp)
    right_rectified = rectifier.right(right_erp)

    stereo = triangulate_stereo(left_rectified, right_rectified, rectifier, args)
    result["orb_matches"] = stereo["raw_matches"]
    result["epipolar_matches"] = stereo.get("epipolar_matches", 0)
    result["triangulated_points"] = len(stereo["points_3d"])
    if len(stereo["points_3d"]) < args.min_stereo_points:
        result["reason"] = "too few valid stereo points"
        return result, visualization

    target_image, tracked_3d, start_points, target_points = track_points(
        left_capture,
        left_video,
        rectifier,
        anchor_frame,
        target_frame,
        left_rectified,
        stereo["points_3d"],
        stereo["left_points"],
        args,
    )
    result["tracked_points"] = len(tracked_3d)
    if len(tracked_3d) < args.min_tracked_points:
        result["reason"] = "too few temporally tracked points"
        return result, visualization

    pnp = estimate_pnp(tracked_3d, target_points, rectifier.rectified_k, args)
    if pnp is None:
        result["reason"] = "PnP RANSAC failed or has too few inliers"
        return result, visualization

    inlier_count = len(pnp["inliers"])
    inlier_ratio = inlier_count / max(len(tracked_3d), 1)
    result["pnp_inliers"] = inlier_count
    result["pnp_inlier_ratio"] = inlier_ratio
    result["pnp_reprojection_rms_px"] = pnp["reprojection_rms"]
    if inlier_ratio < args.min_pnp_inlier_ratio:
        result["reason"] = "PnP inlier ratio is too low"
        return result, visualization
    if pnp["reprojection_rms"] > args.max_pnp_reprojection_rms:
        result["reason"] = "PnP reprojection RMS is too high"
        return result, visualization

    scale_estimate = estimate_scale_from_stella_motion(
        tracked_3d,
        target_points,
        pnp["inliers"],
        pnp["rotation"],
        anchor_pose,
        target_pose,
        rectifier,
        args,
    )
    if scale_estimate is None:
        result["reason"] = "too few consistent point-wise Stella scale constraints"
        return result, visualization
    result["pnp_metric_distance_m"] = float(np.linalg.norm(pnp["translation"]))
    result["scale_point_count"] = scale_estimate["point_count"]
    result["scale_point_mad"] = scale_estimate["point_mad"]
    result["rotation_matrix_error_deg"] = scale_estimate["rotation_error_deg"]
    result["stella_scale_reprojection_rms_px"] = scale_estimate["reprojection_rms_px"]

    slam_delta = target_pose["position"] - anchor_pose["position"]
    slam_distance = float(np.linalg.norm(slam_delta))
    metric_distance = scale_estimate["scale"] * slam_distance
    result["metric_distance_m"] = metric_distance
    result["slam_distance_units"] = slam_distance
    if metric_distance < args.min_metric_motion:
        result["reason"] = "metric motion is too small"
        return result, visualization
    if metric_distance > args.max_metric_motion:
        result["reason"] = "metric motion is implausibly large"
        return result, visualization
    if slam_distance < args.min_slam_motion:
        result["reason"] = "Stella motion is too small"
        return result, visualization

    rotation0_wc = quaternion_to_rotation(anchor_pose["quaternion"])
    rotation1_wc = quaternion_to_rotation(target_pose["quaternion"])
    slam_relative_rotation = rotation1_wc.T @ rotation0_wc
    result["pnp_rotation_deg"] = rotation_angle_deg(pnp["rotation"])
    result["rigid_rotation_deg"] = rotation_angle_deg(scale_estimate["rotation"])
    result["slam_rotation_deg"] = rotation_angle_deg(slam_relative_rotation)
    result["scale_m_per_slam_unit"] = scale_estimate["scale"]
    result["status"] = "quality_pass"
    result["reason"] = ""

    visualization = {
        "left": left_rectified,
        "right": right_rectified,
        "target": target_image,
        "stereo_left": stereo["left_points"],
        "stereo_right": stereo["right_points"],
        "start_points": start_points,
        "target_points": target_points,
        "temporal_inliers": pnp["inliers"],
    }
    return result, visualization


def fit_robust_scale(results, minimum_segments, max_relative_iqr):
    quality_indices = [
        index
        for index, result in enumerate(results)
        if result["status"] == "quality_pass" and np.isfinite(result["scale_m_per_slam_unit"])
    ]
    if len(quality_indices) < minimum_segments:
        raise RuntimeError(
            f"only {len(quality_indices)} quality scale segments; need at least {minimum_segments}. "
            "Inspect the CSV/visualizations and synchronization offset."
        )

    ratios = np.asarray([results[index]["scale_m_per_slam_unit"] for index in quality_indices], dtype=np.float64)
    initial_median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - initial_median)))
    robust_sigma = 1.4826 * mad
    threshold = max(3.0 * robust_sigma, 0.10 * initial_median)
    keep = np.abs(ratios - initial_median) <= threshold

    kept_indices = [index for index, accepted in zip(quality_indices, keep) if accepted]
    if len(kept_indices) < minimum_segments:
        raise RuntimeError(
            f"only {len(kept_indices)} robust scale segments remain; need at least {minimum_segments}"
        )
    for index in kept_indices:
        results[index]["scale_inlier"] = True
        results[index]["status"] = "scale_inlier"
    for index in quality_indices:
        if not results[index]["scale_inlier"]:
            results[index]["status"] = "scale_outlier"
            results[index]["reason"] = "rejected by MAD scale filter"

    final_ratios = np.asarray(
        [results[index]["scale_m_per_slam_unit"] for index in kept_indices], dtype=np.float64
    )
    final_scale = float(np.median(final_ratios))
    q25 = float(np.percentile(final_ratios, 25))
    q75 = float(np.percentile(final_ratios, 75))
    relative_iqr = (q75 - q25) / max(final_scale, 1e-12)
    if relative_iqr > max_relative_iqr:
        raise RuntimeError(
            f"scale estimates are inconsistent: relative IQR={relative_iqr:.3f}, "
            f"limit={max_relative_iqr:.3f}; refusing to write a metric trajectory"
        )
    return {
        "scale": final_scale,
        "quality_segments": len(quality_indices),
        "scale_inlier_segments": len(kept_indices),
        "median_before_filter": initial_median,
        "mad_before_filter": mad,
        "q25": q25,
        "q75": q75,
        "relative_iqr": relative_iqr,
        "minimum": float(np.min(final_ratios)),
        "maximum": float(np.max(final_ratios)),
    }


def draw_correspondences(image0, image1, points0, points1, title, limit=120):
    canvas = np.hstack([image0.copy(), image1.copy()])
    offset = image0.shape[1]
    count = min(len(points0), int(limit))
    if count > 0:
        indices = np.linspace(0, len(points0) - 1, count, dtype=int)
        for order, index in enumerate(indices):
            color = tuple(int(value) for value in cv2.applyColorMap(
                np.array([[[(order * 17) % 256]]], dtype=np.uint8), cv2.COLORMAP_TURBO
            )[0, 0])
            p0 = tuple(int(round(value)) for value in points0[index])
            p1 = (int(round(points1[index][0])) + offset, int(round(points1[index][1])))
            cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_visualization(path, visualization, result):
    path = Path(path)
    stereo_canvas = draw_correspondences(
        visualization["left"],
        visualization["right"],
        visualization["stereo_left"],
        visualization["stereo_right"],
        f"Stereo frame {result['anchor_frame']}: {result['triangulated_points']} metric 3D points",
    )
    temporal_indices = visualization["temporal_inliers"]
    temporal_canvas = draw_correspondences(
        visualization["left"],
        visualization["target"],
        visualization["start_points"][temporal_indices],
        visualization["target_points"][temporal_indices],
        (
            f"Temporal {result['anchor_frame']} -> {result['target_frame']}: "
            f"{result['scale_point_count']} scale points, scale={result['scale_m_per_slam_unit']:.4f}"
        ),
    )
    output = np.vstack([stereo_canvas, temporal_canvas])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_csv(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(base_result({"frame": 0}, {"frame": 0}, 0).keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_scaled_trajectory(path, poses, scale):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for pose in poses:
            position = pose["position"] * scale
            quaternion = pose["quaternion"]
            stream.write(
                f"{pose['timestamp']:.15g} "
                f"{position[0]:.9g} {position[1]:.9g} {position[2]:.9g} "
                f"{quaternion[0]:.9g} {quaternion[1]:.9g} {quaternion[2]:.9g} {quaternion[3]:.9g}\n"
            )


def write_summary(path, args, calibration, left_info, right_info, scale_fit, output_trajectory, report_path):
    path = Path(path)
    data = {
        "scale_m_per_slam_unit": scale_fit["scale"],
        "scale_statistics": {key: value for key, value in scale_fit.items() if key != "scale"},
        "calibration": {
            "path": str(Path(args.calibration).resolve()),
            "baseline_m": calibration["baseline_m"],
            "virtual_view": {
                "fov_deg": args.fov,
                "theta_deg": args.theta,
                "phi_deg": args.phi,
                "width": args.view_width,
                "height": args.view_height,
            },
        },
        "inputs": {
            "slam_camera": args.slam_camera,
            "slam_video": str(Path(args.left_video).resolve()),
            "other_camera_video": str(Path(args.right_video).resolve()),
            "other_camera_frame_offset": args.right_frame_offset,
            "trajectory": str(Path(args.trajectory).resolve()),
            "trajectory_first_frame": args.trajectory_first_frame,
            "fps": left_info["fps"],
            "left_frames": left_info["frames"],
            "right_frames": right_info["frames"],
        },
        "outputs": {
            "metric_trajectory": str(Path(output_trajectory).resolve()),
            "segments_csv": str(Path(report_path).resolve()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def resolve_paths(args):
    scene_dir = REPO_ROOT / "data" / "arrow360" / args.scene
    if args.left_video is None:
        args.left_video = scene_dir / "video_crop.mp4"
    if args.trajectory is None:
        args.trajectory = scene_dir / "frame_trajectory.txt"
    if args.output_trajectory is None:
        args.output_trajectory = scene_dir / "frame_trajectory_metric.txt"
    if args.report is None:
        args.report = scene_dir / "stereo_scale_segments.csv"
    if args.summary is None:
        args.summary = scene_dir / "stereo_scale.yaml"
    if args.visualization_dir is None:
        args.visualization_dir = scene_dir / "stereo_scale_visualization"

    for name in ("left_video", "right_video", "calibration", "trajectory"):
        value = getattr(args, name)
        if value is None or not Path(value).is_file():
            if name == "right_video":
                raise FileNotFoundError(
                    "the other stereo camera video is required; pass --other-video with the other Theta recording"
                )
            raise FileNotFoundError(f"{name.replace('_', ' ')} not found: {value}")

    if Path(args.left_video).resolve() == Path(args.right_video).resolve():
        raise ValueError("left and right videos must be different camera recordings")
    if args.temporal_gap <= 0 or args.sample_step <= 0 or args.slam_frame_skip <= 0:
        raise ValueError("--temporal-gap, --sample-step, and --slam-frame-skip must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover metric scale for a monocular Stella trajectory using synchronized stereo Theta video."
    )
    parser.add_argument("scene", nargs="?", default="jia", help="Scene name under data/arrow360 (default: jia)")
    parser.add_argument(
        "--slam-video",
        "--left-video",
        dest="left_video",
        type=Path,
        default=None,
        help="ERP video used by Stella; defaults to scene/video_crop.mp4",
    )
    parser.add_argument(
        "--other-video",
        "--right-video",
        dest="right_video",
        type=Path,
        required=True,
        help="Synchronized ERP video from the other calibrated Theta",
    )
    parser.add_argument(
        "--slam-camera",
        choices=("cam0", "cam1"),
        default="cam1",
        help="Camera key in checkerboard_camchain.yaml used by Stella (jia physical cam2 is cam1)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "data/ricoh/cali_opencv/kalibr_virtual_sync_557_554_fov90/checkerboard_camchain.yaml",
    )
    parser.add_argument("--trajectory", type=Path, default=None, help="Stella TUM trajectory; defaults to scene/frame_trajectory.txt")
    parser.add_argument("--output-trajectory", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="Per-segment CSV output")
    parser.add_argument("--summary", type=Path, default=None, help="Scale YAML output")
    parser.add_argument("--visualization-dir", type=Path, default=None)
    parser.add_argument(
        "--other-frame-offset",
        "--right-frame-offset",
        dest="right_frame_offset",
        type=int,
        default=0,
        help="Read other-camera frame = Stella-video frame + offset; determine it for each recording.",
    )
    parser.add_argument("--trajectory-first-frame", type=int, default=0, help="Video frame corresponding to the first TUM timestamp")
    parser.add_argument(
        "--slam-frame-skip",
        type=int,
        default=2,
        help="Expected raw-video frame step between Stella poses (must match run_slam.sh)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--temporal-gap", type=int, default=10, help="Desired frame gap for each metric-motion estimate")
    parser.add_argument("--sample-step", type=int, default=30, help="Spacing between sampled anchor frames")
    parser.add_argument("--max-target-delay", type=int, default=4, help="Allowed extra frames when Stella has no pose at the exact target")
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--view-width", type=int, default=None)
    parser.add_argument("--view-height", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=3500)
    parser.add_argument("--match-ratio", type=float, default=0.78)
    parser.add_argument("--temporal-match-ratio", type=float, default=0.80)
    parser.add_argument("--epipolar-threshold", type=float, default=2.0)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-stereo-reprojection-error", type=float, default=1.5)
    parser.add_argument("--min-stereo-points", type=int, default=30)
    parser.add_argument("--max-lk-forward-backward-error", type=float, default=1.0)
    parser.add_argument("--min-tracked-points", type=int, default=20)
    parser.add_argument("--pnp-reprojection-threshold", type=float, default=3.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=15)
    parser.add_argument("--min-pnp-inlier-ratio", type=float, default=0.45)
    parser.add_argument("--max-pnp-reprojection-rms", type=float, default=2.5)
    parser.add_argument("--rigid-ransac-iterations", type=int, default=500)
    parser.add_argument("--rigid-ransac-threshold", type=float, default=0.25, help="3D-3D inlier threshold in meters")
    parser.add_argument("--min-rigid-inliers", type=int, default=20)
    parser.add_argument("--min-rigid-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--target-association-radius", type=float, default=4.0, help="LK-to-target-ORB radius in pixels")
    parser.add_argument("--max-rotation-error", type=float, default=2.0, help="PnP-vs-Stella rotation error in degrees")
    parser.add_argument("--min-scale-observability", type=float, default=1e-6)
    parser.add_argument("--min-scale-point-inliers", type=int, default=20)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--min-metric-motion", type=float, default=0.03)
    parser.add_argument("--max-metric-motion", type=float, default=5.0)
    parser.add_argument("--min-slam-motion", type=float, default=0.005)
    parser.add_argument("--min-scale-segments", type=int, default=3)
    parser.add_argument("--max-relative-iqr", type=float, default=0.25)
    parser.add_argument("--visualization-limit", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_paths(args)

    green("[1/4] Loading stereo calibration, videos, and Stella trajectory")
    raw_calibration = load_calibration(args.calibration)
    calibration = calibration_for_slam_camera(raw_calibration, args.slam_camera)
    if args.view_width is None:
        args.view_width = calibration["resolution"][0]
    if args.view_height is None:
        args.view_height = calibration["resolution"][1]
    left_info = video_info(args.left_video)
    right_info = video_info(args.right_video)
    if abs(left_info["fps"] - right_info["fps"]) > 0.05:
        raise ValueError(f"video FPS mismatch: left={left_info['fps']}, right={right_info['fps']}")
    if (left_info["width"], left_info["height"]) != (right_info["width"], right_info["height"]):
        raise ValueError(
            f"ERP video resolution mismatch: left={left_info['width']}x{left_info['height']}, "
            f"right={right_info['width']}x{right_info['height']}"
        )
    poses = load_tum_trajectory(args.trajectory, left_info["fps"], args.trajectory_first_frame)
    observed_frame_step = float(
        np.median(np.diff([pose["frame"] for pose in poses]))
    )
    if not math.isclose(observed_frame_step, float(args.slam_frame_skip), abs_tol=0.25):
        raise ValueError(
            "trajectory/video frame mapping mismatch: "
            f"observed median step={observed_frame_step:.1f}, "
            f"expected --slam-frame-skip={args.slam_frame_skip}"
        )
    rectifier = VirtualStereoRectifier(
        calibration, args.fov, args.theta, args.phi, args.view_width, args.view_height
    )
    print(f"baseline: {calibration['baseline_m']:.6f} m; Stella camera: {args.slam_camera}")
    print(
        f"left: {left_info['width']}x{left_info['height']}, {left_info['frames']} frames @ {left_info['fps']:.6f} fps"
    )
    print(f"other camera: {right_info['frames']} frames, frame offset: {args.right_frame_offset:+d}")
    print(f"trajectory: {len(poses)} poses, mapped frames {poses[0]['frame']}..{poses[-1]['frame']}")
    print(f"observed Stella raw-frame step: {observed_frame_step:.1f}")

    maximum_right_aligned_left_frame = right_info["frames"] - 1 - args.right_frame_offset
    default_end = min(left_info["frames"] - 1, maximum_right_aligned_left_frame, poses[-1]["frame"])
    end_frame = default_end if args.end_frame is None else min(args.end_frame, default_end)
    if end_frame <= args.start_frame:
        raise ValueError(f"empty usable frame range: {args.start_frame}..{end_frame}")
    segments = select_segments(
        poses,
        args.start_frame,
        end_frame,
        args.temporal_gap,
        args.sample_step,
        args.max_target_delay,
    )
    if not segments:
        raise RuntimeError("no trajectory segments selected; check frame range and trajectory timestamps")

    green(f"[2/4] Estimating metric motion for {len(segments)} synchronized frame segments")
    left_capture = cv2.VideoCapture(str(args.left_video))
    right_capture = cv2.VideoCapture(str(args.right_video))
    if not left_capture.isOpened() or not right_capture.isOpened():
        left_capture.release()
        right_capture.release()
        raise RuntimeError("failed to open stereo videos")

    results = []
    visualization_count = 0
    try:
        for segment_number, (anchor_index, target_index) in enumerate(segments, start=1):
            anchor_pose = poses[anchor_index]
            target_pose = poses[target_index]
            try:
                result, visualization = process_segment(
                    left_capture,
                    right_capture,
                    args.left_video,
                    args.right_video,
                    rectifier,
                    anchor_pose,
                    target_pose,
                    args,
                )
            except (cv2.error, RuntimeError, ValueError) as exc:
                result = base_result(
                    anchor_pose, target_pose, anchor_pose["frame"] + args.right_frame_offset
                )
                result["reason"] = str(exc)
                visualization = None
            results.append(result)

            if visualization is not None and visualization_count < args.visualization_limit:
                filename = (
                    f"segment_{segment_number:03d}_"
                    f"{anchor_pose['frame']:06d}_{target_pose['frame']:06d}.jpg"
                )
                save_visualization(Path(args.visualization_dir) / filename, visualization, result)
                visualization_count += 1

            if result["status"] == "quality_pass":
                detail = (
                    f"scale={result['scale_m_per_slam_unit']:.4f}, "
                    f"metric={result['metric_distance_m']:.3f} m, "
                    f"slam={result['slam_distance_units']:.4f}, "
                    f"scale/pnp points={result['scale_point_count']}/{result['pnp_inliers']}"
                )
            else:
                detail = result["reason"]
            print(
                f"[{segment_number:02d}/{len(segments):02d}] "
                f"frames {anchor_pose['frame']}->{target_pose['frame']}: {detail}"
            )
    finally:
        left_capture.release()
        right_capture.release()

    green("[3/4] Rejecting inconsistent segments and estimating one global scale")
    try:
        scale_fit = fit_robust_scale(
            results, args.min_scale_segments, args.max_relative_iqr
        )
    except RuntimeError:
        write_csv(Path(args.report), results)
        raise

    green("[4/4] Writing metric trajectory and diagnostics")
    write_csv(Path(args.report), results)
    write_scaled_trajectory(Path(args.output_trajectory), poses, scale_fit["scale"])
    write_summary(
        Path(args.summary),
        args,
        calibration,
        left_info,
        right_info,
        scale_fit,
        args.output_trajectory,
        args.report,
    )

    green(f"Scale: {scale_fit['scale']:.6f} m / SLAM unit")
    green(
        f"Robust segments: {scale_fit['scale_inlier_segments']}/{scale_fit['quality_segments']}; "
        f"IQR [{scale_fit['q25']:.6f}, {scale_fit['q75']:.6f}]"
    )
    green(f"Metric trajectory: {args.output_trajectory}")
    print(f"segment report: {args.report}")
    print(f"summary: {args.summary}")
    print(f"visualizations: {args.visualization_dir}")


if __name__ == "__main__":
    main()
