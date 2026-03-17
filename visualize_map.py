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


def animate_map(msg_path, frame_ms=50, frustum_scale=0.35, follow_view=True):
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
    args = parser.parse_args()

    animate_map(
        args.map,
        frame_ms=args.frame_ms,
        frustum_scale=args.frustum_scale,
        follow_view=not args.no_follow_view
    )
