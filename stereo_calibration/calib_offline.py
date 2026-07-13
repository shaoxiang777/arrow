#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GREEN = "\033[92m"
RESET = "\033[0m"


def list_images(path):
    images = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"no images found in {path}")
    return images


def checkerboard_object_points(rows, cols, square_size):
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    return points * float(square_size)


def detect_checkerboard(image, rows, cols, legacy_fallback=False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    pattern_size = (rows, cols)

    if hasattr(cv, "findChessboardCornersSB"):
        ok, corners = cv.findChessboardCornersSB(gray, pattern_size)
        if ok:
            return corners.astype(np.float32)
        if not legacy_fallback:
            return None

    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def draw_detection(image, rows, cols, corners, label):
    out = image.copy()
    ok = corners is not None
    if ok:
        cv.drawChessboardCorners(out, (rows, cols), corners, True)
        first = tuple(corners.reshape(-1, 2)[0].astype(int))
        cv.circle(out, first, 8, (0, 0, 255), -1)
    color = (40, 220, 40) if ok else (0, 0, 255)
    text = f"{label}: {'OK' if ok else 'NO CHECKERBOARD'}"
    cv.rectangle(out, (10, 10), (420, 52), (0, 0, 0), -1)
    cv.putText(out, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
    return out


def resize_to_width(image, width):
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def draw_axis(image, origin, axis_points):
    origin_i = tuple(np.round(origin.reshape(2)).astype(int))
    axis_points = axis_points.reshape(3, 2)
    # OpenCV uses BGR colors; these correspond to X=red, Y=green, Z=blue.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]
    for point, color, axis_label in zip(axis_points, colors, labels):
        point_i = tuple(np.round(point).astype(int))
        cv.line(image, origin_i, point_i, color, 3, cv.LINE_AA)
        cv.circle(image, point_i, 5, color, -1, cv.LINE_AA)
        cv.putText(image, axis_label, point_i, cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
    cv.circle(image, origin_i, 6, (255, 255, 255), -1, cv.LINE_AA)
    return image


def draw_reprojection_error(image, observed, projected, label, rms_px, axis_origin=None, axis_points=None):
    out = image.copy()
    observed = observed.reshape(-1, 2)
    projected = projected.reshape(-1, 2)
    for obs, proj in zip(observed, projected):
        obs_i = tuple(np.round(obs).astype(int))
        proj_i = tuple(np.round(proj).astype(int))
        cv.line(out, obs_i, proj_i, (0, 180, 255), 1, cv.LINE_AA)
        cv.circle(out, obs_i, 3, (0, 220, 0), -1, cv.LINE_AA)
        cv.drawMarker(out, proj_i, (255, 0, 255), cv.MARKER_CROSS, 8, 1, cv.LINE_AA)

    if axis_origin is not None and axis_points is not None:
        out = draw_axis(out, axis_origin, axis_points)

    cv.rectangle(out, (10, 10), (760, 110), (0, 0, 0), -1)
    cv.putText(out, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(
        out,
        f"green=detected  magenta=projected  rms={rms_px:.4f}px",
        (20, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        out,
        "axis: X=red Y=green Z=blue, origin=reprojected board corner",
        (20, 98),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    return out


def write_reprojection_visualizations(
    image_paths,
    objpoints,
    imgpoints,
    camera_matrix,
    dist,
    rvecs,
    tvecs,
    output_dir,
    label,
    square_size,
    limit=80,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(3, 1, 3) * float(square_size)
    center = np.float32([[0, 0, 0]])

    for idx, (image_path, objp, detected, rvec, tvec) in enumerate(
        zip(image_paths, objpoints, imgpoints, rvecs, tvecs)
    ):
        projected, _ = cv.projectPoints(objp, rvec, tvec, camera_matrix, dist)
        centerpts, _ = cv.projectPoints(center, rvec, tvec, camera_matrix, dist)
        axispts, _ = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist)
        detected_2d = detected.reshape(-1, 2)
        projected_2d = projected.reshape(-1, 2)
        errors = np.linalg.norm(detected_2d - projected_2d, axis=1)
        rms_px = float(np.sqrt(np.mean(errors * errors)))
        mean_px = float(np.mean(errors))
        max_px = float(np.max(errors))
        rows.append([image_path.name, f"{rms_px:.6f}", f"{mean_px:.6f}", f"{max_px:.6f}"])

        if idx < limit:
            image = cv.imread(str(image_path), cv.IMREAD_COLOR)
            if image is None:
                continue
            vis = draw_reprojection_error(
                image,
                detected,
                projected,
                f"{label} {image_path.name}",
                rms_px,
                axis_origin=centerpts,
                axis_points=axispts,
            )
            cv.imwrite(str(output_dir / f"{label}_{idx:03d}_{image_path.stem}.jpg"), vis)

    csv_path = output_dir / f"{label}_reprojection_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "rms_px", "mean_px", "max_px"])
        writer.writerows(rows)
    print(f"wrote {label} reprojection visualizations: {output_dir}")


def write_stitched_reprojection_visualizations(
    cam0_data,
    cam1_data,
    output_dir,
    square_size,
    limit=80,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    common_names = sorted(set(cam0_data) & set(cam1_data))

    for idx, name in enumerate(common_names):
        data0 = cam0_data[name]
        data1 = cam1_data[name]
        vis0, rms0, mean0, max0 = make_reprojection_visualization(data0, "cam0", square_size)
        vis1, rms1, mean1, max1 = make_reprojection_visualization(data1, "cam1", square_size)
        rows.append(
            [
                name,
                f"{rms0:.6f}",
                f"{mean0:.6f}",
                f"{max0:.6f}",
                f"{rms1:.6f}",
                f"{mean1:.6f}",
                f"{max1:.6f}",
            ]
        )

        if idx < limit and vis0 is not None and vis1 is not None:
            tile = np.hstack([resize_to_width(vis0, 500), resize_to_width(vis1, 500)])
            cv.imwrite(str(output_dir / f"reprojection_pair_{idx:03d}_{Path(name).stem}.jpg"), tile)

    csv_path = output_dir / "stitched_reprojection_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "cam0_rms_px",
                "cam0_mean_px",
                "cam0_max_px",
                "cam1_rms_px",
                "cam1_mean_px",
                "cam1_max_px",
            ]
        )
        writer.writerows(rows)
    print(f"wrote stitched reprojection visualizations: {output_dir}")


def make_reprojection_visualization(data, label, square_size):
    image_path = data["image_path"]
    objp = data["objpoints"]
    detected = data["imgpoints"]
    camera_matrix = data["camera_matrix"]
    dist = data["dist"]
    rvec = data["rvec"]
    tvec = data["tvec"]

    projected, _ = cv.projectPoints(objp, rvec, tvec, camera_matrix, dist)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(3, 1, 3) * float(square_size)
    center = np.float32([[0, 0, 0]])
    centerpts, _ = cv.projectPoints(center, rvec, tvec, camera_matrix, dist)
    axispts, _ = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist)

    detected_2d = detected.reshape(-1, 2)
    projected_2d = projected.reshape(-1, 2)
    errors = np.linalg.norm(detected_2d - projected_2d, axis=1)
    rms_px = float(np.sqrt(np.mean(errors * errors)))
    mean_px = float(np.mean(errors))
    max_px = float(np.max(errors))

    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        return None, rms_px, mean_px, max_px
    vis = draw_reprojection_error(
        image,
        detected,
        projected,
        f"{label} {image_path.name}",
        rms_px,
        axis_origin=centerpts,
        axis_points=axispts,
    )
    return vis, rms_px, mean_px, max_px


def build_reprojection_data(image_paths, objpoints, imgpoints, camera, label):
    _, camera_matrix, dist, rvecs, tvecs = camera
    data = {}
    for image_path, objp, imgp, rvec, tvec in zip(image_paths, objpoints, imgpoints, rvecs, tvecs):
        if image_path.name in data:
            raise RuntimeError(f"{label}: duplicate accepted image name: {image_path.name}")
        data[image_path.name] = {
            "image_path": image_path,
            "objpoints": objp,
            "imgpoints": imgp,
            "camera_matrix": camera_matrix,
            "dist": dist,
            "rvec": rvec,
            "tvec": tvec,
        }
    return data


def load_detections(images, rows, cols, square_size, label, legacy_fallback=False):
    objp = checkerboard_object_points(rows, cols, square_size)
    objpoints = []
    imgpoints = []
    image_size = None
    accepted = []

    for image_path in images:
        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if image is None:
            print(f"{label}: skip unreadable image {image_path}")
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif image_size != size:
            raise RuntimeError(f"{label}: image size changed: {image_path} is {size}, expected {image_size}")

        corners = detect_checkerboard(image, rows, cols, legacy_fallback=legacy_fallback)
        if corners is None:
            print(f"{label}: no checkerboard in {image_path.name}")
            continue

        objpoints.append(objp)
        imgpoints.append(corners)
        accepted.append(image_path)

    if len(objpoints) < 5:
        raise RuntimeError(f"{label}: only {len(objpoints)} valid checkerboard images; need at least 5")

    return objpoints, imgpoints, image_size, accepted


def collect_stereo_detections(
    cam0_images,
    cam1_images,
    rows,
    cols,
    square_size,
    legacy_fallback=False,
    visualization_dir=None,
    visualization_limit=80,
):
    objp = checkerboard_object_points(rows, cols, square_size)
    objpoints = []
    imgpoints0 = []
    imgpoints1 = []
    image_size = None
    vis_dir = Path(visualization_dir) if visualization_dir else None

    pairs = list(zip(cam0_images, cam1_images))
    for pair_idx, (cam0_path, cam1_path) in enumerate(pairs):
        if cam0_path.stem != cam1_path.stem:
            print(f"warning: pairing by sort order: {cam0_path.name} <-> {cam1_path.name}")

        image0 = cv.imread(str(cam0_path), cv.IMREAD_COLOR)
        image1 = cv.imread(str(cam1_path), cv.IMREAD_COLOR)
        if image0 is None or image1 is None:
            print(f"stereo: skip unreadable pair {cam0_path.name}, {cam1_path.name}")
            continue
        size0 = (image0.shape[1], image0.shape[0])
        size1 = (image1.shape[1], image1.shape[0])
        if size0 != size1:
            raise RuntimeError(f"stereo: image sizes differ in pair {cam0_path.name}, {cam1_path.name}: {size0} vs {size1}")
        if image_size is None:
            image_size = size0
        elif image_size != size0:
            raise RuntimeError(f"stereo: image size changed: {cam0_path.name} is {size0}, expected {image_size}")

        corners0 = detect_checkerboard(image0, rows, cols, legacy_fallback=legacy_fallback)
        corners1 = detect_checkerboard(image1, rows, cols, legacy_fallback=legacy_fallback)

        if vis_dir and pair_idx < visualization_limit:
            vis0 = draw_detection(image0, rows, cols, corners0, f"cam0 {cam0_path.name}")
            vis1 = draw_detection(image1, rows, cols, corners1, f"cam1 {cam1_path.name}")
            tile = np.hstack([resize_to_width(vis0, 500), resize_to_width(vis1, 500)])
            vis_dir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(vis_dir / f"pair_{pair_idx:03d}_{cam0_path.stem}.jpg"), tile)

        if corners0 is None or corners1 is None:
            print(f"stereo: no checkerboard in pair {cam0_path.name}, {cam1_path.name}")
            continue

        objpoints.append(objp)
        imgpoints0.append(corners0)
        imgpoints1.append(corners1)

    if vis_dir:
        print(f"wrote visualizations: {vis_dir}")

    if len(objpoints) < 5:
        raise RuntimeError(f"stereo: only {len(objpoints)} valid pairs; need at least 5")

    return objpoints, imgpoints0, imgpoints1, image_size


def calibrate_intrinsics(objpoints, imgpoints, image_size, label):
    rms, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f"{label}: intrinsic rms={rms:.6f}")
    print(f"{label}: camera_matrix=\n{camera_matrix}")
    print(f"{label}: distortion={dist.ravel()}")
    return rms, camera_matrix, dist, rvecs, tvecs


def write_outputs(output_path, image_size, camera0, camera1, stereo):
    rms0, k0, d0 = camera0[:3]
    rms1, k1, d1 = camera1[:3]
    stereo_rms, rotation, translation = stereo
    width, height = image_size
    baseline_m = float(np.linalg.norm(translation.reshape(3)))

    data = {
        "cam0": {
            "camera_model": "pinhole",
            "distortion_model": "radtan",
            "intrinsics": [float(k0[0, 0]), float(k0[1, 1]), float(k0[0, 2]), float(k0[1, 2])],
            "distortion_coeffs": [float(x) for x in d0.ravel()[:4]],
            "resolution": [int(width), int(height)],
            "reprojection_error": float(rms0),
        },
        "cam1": {
            "camera_model": "pinhole",
            "distortion_model": "radtan",
            "intrinsics": [float(k1[0, 0]), float(k1[1, 1]), float(k1[0, 2]), float(k1[1, 2])],
            "distortion_coeffs": [float(x) for x in d1.ravel()[:4]],
            "resolution": [int(width), int(height)],
            "reprojection_error": float(rms1),
            "T_cn_cnm1": [
                [float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), float(translation[0, 0])],
                [float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), float(translation[1, 0])],
                [float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), float(translation[2, 0])],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "stereo_reprojection_error": float(stereo_rms),
            "baseline_m": baseline_m,
            "baseline_cm": baseline_m * 100.0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"wrote {output_path}")
    print(f"{GREEN}baseline: {baseline_m:.6f} m ({baseline_m * 100.0:.2f} cm){RESET}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline OpenCV stereo calibration from checkerboard image pairs.")
    parser.add_argument("--cam0-dir", required=True, help="Directory containing synchronized cam0 checkerboard images")
    parser.add_argument("--cam1-dir", required=True, help="Directory containing synchronized cam1 checkerboard images")
    parser.add_argument("--checkerboard-rows", type=int, required=True, help="Inner corners along checkerboard rows")
    parser.add_argument("--checkerboard-cols", type=int, required=True, help="Inner corners along checkerboard columns")
    parser.add_argument("--square-size", type=float, required=True, help="Checkerboard square size in meters")
    parser.add_argument("--output", default="camera_parameters/opencv_checkerboard_camchain.yaml")
    parser.add_argument(
        "--legacy-fallback",
        action="store_true",
        help="Fall back to cv.findChessboardCorners when findChessboardCornersSB fails.",
    )
    parser.add_argument("--visualization-dir", default=None, help="Directory for checkerboard overlay images.")
    parser.add_argument(
        "--visualization-limit",
        type=int,
        default=80,
        help="Maximum number of stereo pairs to save as visualization overlays.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cam0_images = list_images(args.cam0_dir)
    cam1_images = list_images(args.cam1_dir)
    pair_count = min(len(cam0_images), len(cam1_images))
    if len(cam0_images) != len(cam1_images):
        print(f"warning: cam0 has {len(cam0_images)} images, cam1 has {len(cam1_images)}; using {pair_count} sorted pairs")
    cam0_images = cam0_images[:pair_count]
    cam1_images = cam1_images[:pair_count]

    rows = args.checkerboard_rows
    cols = args.checkerboard_cols
    square_size = args.square_size

    obj0, img0, size0, accepted0 = load_detections(
        cam0_images, rows, cols, square_size, "cam0", legacy_fallback=args.legacy_fallback
    )
    obj1, img1, size1, accepted1 = load_detections(
        cam1_images, rows, cols, square_size, "cam1", legacy_fallback=args.legacy_fallback
    )
    if size0 != size1:
        raise RuntimeError(f"cam0 and cam1 image sizes differ: {size0} vs {size1}")

    camera0 = calibrate_intrinsics(obj0, img0, size0, "cam0")
    camera1 = calibrate_intrinsics(obj1, img1, size1, "cam1")
    if args.visualization_dir:
        vis_root = Path(args.visualization_dir)
        write_stitched_reprojection_visualizations(
            build_reprojection_data(accepted0, obj0, img0, camera0, "cam0"),
            build_reprojection_data(accepted1, obj1, img1, camera1, "cam1"),
            vis_root / "reprojection_stitched",
            square_size,
            limit=args.visualization_limit,
        )

    obj_stereo, img_stereo0, img_stereo1, stereo_size = collect_stereo_detections(
        cam0_images,
        cam1_images,
        rows,
        cols,
        square_size,
        legacy_fallback=args.legacy_fallback,
        visualization_dir=args.visualization_dir,
        visualization_limit=args.visualization_limit,
    )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    stereo_rms, _, _, _, _, rotation, translation, _, _ = cv.stereoCalibrate(
        obj_stereo,
        img_stereo0,
        img_stereo1,
        camera0[1],
        camera0[2],
        camera1[1],
        camera1[2],
        stereo_size,
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC,
    )
    print(f"stereo: rms={stereo_rms:.6f}")
    print(f"stereo: R=\n{rotation}")
    print(f"stereo: T={translation.ravel()}")
    print(f"cam0 accepted images: {len(accepted0)}")
    print(f"cam1 accepted images: {len(accepted1)}")
    print(f"stereo accepted pairs: {len(obj_stereo)}")

    write_outputs(Path(args.output), stereo_size, camera0, camera1, (stereo_rms, rotation, translation))


if __name__ == "__main__":
    main()
