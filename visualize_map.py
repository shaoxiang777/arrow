import argparse
import time

import msgpack
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_map(msg_path):
    with open(msg_path, "rb") as f:
        msg = msgpack.Unpacker(f, raw=False).unpack()

    keyframes = []
    for kf_id_str, kf in msg["keyframes"].items():
        kf_id = int(kf_id_str)
        rot_cw = R.from_quat(kf["rot_cw"]).as_matrix()
        trans_cw = np.array(kf["trans_cw"], dtype=np.float64).reshape(3, 1)

        rot_wc = rot_cw.T
        pos_w = (-rot_wc @ trans_cw).reshape(3)
        keyframes.append({
            "id": kf_id,
            "ts": float(kf["ts"]),
            "rot_wc": rot_wc,
            "pos_w": pos_w,
        })

    keyframes.sort(key=lambda x: x["ts"])

    # 按路标首次关键帧分组，便于逐帧累积显示点云
    lm_groups = {}
    for lm in msg["landmarks"].values():
        first_kf_id = int(lm.get("1st_keyfrm", -1))
        pos = np.array(lm["pos_w"], dtype=np.float64)
        lm_groups.setdefault(first_kf_id, []).append(pos)

    lm_first_ids = sorted(lm_groups.keys())
    return keyframes, lm_groups, lm_first_ids


def make_frustum(pos_w, rot_wc, scale=0.35):
    z = scale
    hx = 0.7 * scale
    hy = 0.45 * scale
    points_c = np.array([
        [0.0, 0.0, 0.0],      # 相机中心
        [hx, hy, z],
        [hx, -hy, z],
        [-hx, -hy, z],
        [-hx, hy, z],
    ], dtype=np.float64)

    points_w = (rot_wc @ points_c.T).T + pos_w.reshape(1, 3)
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ], dtype=np.int32)
    colors = np.tile(np.array([[0.1, 0.6, 1.0]], dtype=np.float64), (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_w)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def _update_follow_view(vis, pos_w, rot_wc, look_ahead=2.0, up_bias=0.4, zoom=0.45):
    """
    让可视化视角跟随关键帧向前移动。
    """
    ctr = vis.get_view_control()

    # 相机坐标系前向 z 轴，转到世界系
    forward = rot_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = rot_wc @ np.array([0.0, -1.0, 0.0], dtype=np.float64)

    n_f = np.linalg.norm(forward)
    n_u = np.linalg.norm(up)
    if n_f < 1e-6 or n_u < 1e-6:
        return
    forward = forward / n_f
    up = up / n_u

    lookat = pos_w + look_ahead * forward + up_bias * up

    # Open3D 的 front 定义为从 lookat 指向相机方向，因此取负向
    ctr.set_lookat(lookat.tolist())
    ctr.set_front((-forward).tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(float(zoom))


def _wait_with_events(vis, wait_ms, tick_ms=10):
    """
    等待 wait_ms，同时持续处理 GUI 事件，保证鼠标交互流畅。
    返回 False 表示窗口已关闭。
    """
    remaining = max(0, int(wait_ms))
    while remaining > 0:
        if not vis.poll_events():
            return False
        vis.update_renderer()
        dt = min(tick_ms, remaining)
        time.sleep(dt / 1000.0)
        remaining -= dt
    return True


def _filter_bounds_points(points, min_points_for_filter=32, quantile=0.02):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < min_points_for_filter:
        return pts

    lower = np.quantile(pts, quantile, axis=0)
    upper = np.quantile(pts, 1.0 - quantile, axis=0)
    mask = np.all((pts >= lower) & (pts <= upper), axis=1)

    filtered = pts[mask]
    if len(filtered) < max(8, len(pts) // 4):
        return pts
    return filtered


def _filter_points_by_radius(points, center, radius):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or radius is None or radius <= 0:
        return pts

    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    dists = np.linalg.norm(pts - center, axis=1)
    filtered = pts[dists <= radius]
    if len(filtered) == 0:
        return pts
    return filtered


def _compute_scene_bounds(
    traj_points,
    lm_points,
    frustum,
    current_pos_w=None,
    bounds_outlier_quantile=0.02,
    bounds_focus_radius=None,
):
    all_points = []

    traj_for_bounds = (
        _filter_points_by_radius(traj_points, current_pos_w, bounds_focus_radius)
        if traj_points else np.empty((0, 3), dtype=np.float64)
    )
    if len(traj_for_bounds) > 0:
        all_points.extend(traj_for_bounds)

    frustum_points = np.asarray(frustum.points)
    if len(frustum_points) > 0:
        all_points.extend(frustum_points)

    lm_for_bounds = (
        _filter_points_by_radius(lm_points, current_pos_w, bounds_focus_radius)
        if lm_points else np.empty((0, 3), dtype=np.float64)
    )
    filtered_lm_points = (
        _filter_bounds_points(lm_for_bounds, quantile=bounds_outlier_quantile)
        if lm_points else np.empty((0, 3), dtype=np.float64)
    )
    if len(filtered_lm_points) > 0:
        all_points.extend(filtered_lm_points)

    if not all_points:
        return None

    pts = np.asarray(all_points, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def _bounds_expanded(bounds, refreshed_bounds, margin_ratio=0.05):
    if bounds is None:
        return False
    if refreshed_bounds is None:
        return True

    cur_min, cur_max = bounds
    ref_min, ref_max = refreshed_bounds
    ref_extent = np.maximum(ref_max - ref_min, 1e-6)
    margin = ref_extent * margin_ratio

    return np.any(cur_min < (ref_min - margin)) or np.any(cur_max > (ref_max + margin))


def _refresh_bounding_box(vis, bounds):
    if bounds is None:
        return

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    bbox_min, bbox_max = bounds
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ], dtype=np.float64)

    bbox_pcd = o3d.geometry.PointCloud()
    bbox_pcd.points = o3d.utility.Vector3dVector(corners)
    vis.add_geometry(bbox_pcd, reset_bounding_box=True)
    vis.remove_geometry(bbox_pcd, reset_bounding_box=False)

    try:
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(cam_params)


def animate_map(
    msg_path,
    frame_ms=50,
    frustum_scale=0.35,
    follow_view=True,
    bounds_outlier_quantile=0.02,
    bounds_focus_radius=None,
):
    keyframes, lm_groups, lm_first_ids = load_map(msg_path)
    if not keyframes:
        raise RuntimeError("No keyframes found in map.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Stella VSLAM Animated Map", width=1280, height=720)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    traj = o3d.geometry.LineSet()
    frustum = make_frustum(keyframes[0]["pos_w"], keyframes[0]["rot_wc"], scale=frustum_scale)

    vis.add_geometry(coord)
    vis.add_geometry(frustum)
    pcd_added = False
    traj_added = False

    traj_points = []
    traj_lines = []
    traj_colors = []
    lm_points = []
    lm_cursor = 0
    refreshed_bounds = _compute_scene_bounds(
        [],
        [],
        frustum,
        current_pos_w=keyframes[0]["pos_w"],
        bounds_outlier_quantile=bounds_outlier_quantile,
        bounds_focus_radius=bounds_focus_radius,
    )

    print(f"Total Keyframes: {len(keyframes)}")
    print(f"Total Landmark Groups: {len(lm_first_ids)}")

    for i, kf in enumerate(keyframes):
        traj_points.append(kf["pos_w"])
        if i > 0:
            traj_lines.append([i - 1, i])
            traj_colors.append([0.0, 1.0, 0.0])

        while lm_cursor < len(lm_first_ids) and lm_first_ids[lm_cursor] <= kf["id"]:
            first_id = lm_first_ids[lm_cursor]
            lm_points.extend(lm_groups[first_id])
            lm_cursor += 1

        if traj_points:
            traj_arr = np.array(traj_points, dtype=np.float64)
            traj.points = o3d.utility.Vector3dVector(traj_arr)
        if traj_lines:
            traj.lines = o3d.utility.Vector2iVector(np.array(traj_lines, dtype=np.int32))
            traj.colors = o3d.utility.Vector3dVector(np.array(traj_colors, dtype=np.float64))
            if not traj_added:
                vis.add_geometry(traj)
                traj_added = True

        if lm_points:
            lm_arr = np.array(lm_points, dtype=np.float64)
            pcd.points = o3d.utility.Vector3dVector(lm_arr)
            pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(lm_arr), 1)))
            if not pcd_added:
                vis.add_geometry(pcd)
                pcd_added = True

        frustum_new = make_frustum(kf["pos_w"], kf["rot_wc"], scale=frustum_scale)
        frustum.points = frustum_new.points
        frustum.lines = frustum_new.lines
        frustum.colors = frustum_new.colors

        current_bounds = _compute_scene_bounds(
            traj_points,
            lm_points,
            frustum,
            current_pos_w=kf["pos_w"],
            bounds_outlier_quantile=bounds_outlier_quantile,
            bounds_focus_radius=bounds_focus_radius,
        )
        if _bounds_expanded(current_bounds, refreshed_bounds):
            _refresh_bounding_box(vis, current_bounds)
            refreshed_bounds = current_bounds

        if follow_view:
            _update_follow_view(vis, kf["pos_w"], kf["rot_wc"])

        if pcd_added:
            vis.update_geometry(pcd)
        if traj_added:
            vis.update_geometry(traj)
        vis.update_geometry(frustum)
        if not vis.poll_events():
            break
        vis.update_renderer()
        if not _wait_with_events(vis, frame_ms):
            break

    # 动画结束后保留最终画面供交互查看，直到用户关闭窗口
    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.02)

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate stella_vslam map with frustum and incremental points")
    parser.add_argument("--map", required=True, help="Path to map msgpack, e.g. /work/data/arrow360/VIDEO_0665/my_saved_map.msg")
    parser.add_argument("--frame-ms", type=int, default=50, help="Animation step interval in milliseconds")
    parser.add_argument("--frustum-scale", type=float, default=0.35, help="Frustum size scale")
    parser.add_argument("--no-follow-view", action="store_true", help="Disable moving camera viewpoint")
    parser.add_argument(
        "--bounds-outlier-quantile",
        type=float,
        default=0.02,
        help="Per-axis quantile used to ignore extreme landmark outliers when refreshing bounding box",
    )
    parser.add_argument(
        "--bounds-focus-radius",
        type=float,
        default=None,
        help="Only use points within this radius of the current camera position when refreshing bounding box",
    )
    args = parser.parse_args()

    animate_map(
        args.map,
        frame_ms=args.frame_ms,
        frustum_scale=args.frustum_scale,
        follow_view=not args.no_follow_view,
        bounds_outlier_quantile=args.bounds_outlier_quantile,
        bounds_focus_radius=args.bounds_focus_radius,
    )
