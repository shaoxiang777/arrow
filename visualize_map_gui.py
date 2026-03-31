import argparse
import json
import os
import threading
import time

import msgpack
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from scipy.spatial.transform import Rotation as R, Slerp


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

    kf_order = {kf["id"]: idx for idx, kf in enumerate(keyframes)}
    lm_groups = {}
    for lm in msg["landmarks"].values():
        first_kf_id = int(lm.get("1st_keyfrm", -1))
        pos = np.array(lm["pos_w"], dtype=np.float64)
        first_kf_order = kf_order.get(first_kf_id, -1)
        lm_groups.setdefault(first_kf_order, []).append(pos)

    lm_first_orders = sorted(lm_groups.keys())
    return keyframes, lm_groups, lm_first_orders


def make_frustum(pos_w, rot_wc, scale=0.35):
    z = scale
    hx = 0.7 * scale
    hy = 0.45 * scale
    points_c = np.array([
        [0.0, 0.0, 0.0],
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

def build_trajectory_line(points):
    traj = o3d.geometry.LineSet()
    if len(points) == 0:
        return traj

    pts = np.asarray(points, dtype=np.float64)
    traj.points = o3d.utility.Vector3dVector(pts)
    if len(pts) > 1:
        lines = np.column_stack([
            np.arange(0, len(pts) - 1, dtype=np.int32),
            np.arange(1, len(pts), dtype=np.int32),
        ])
        colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(lines), 1))
        traj.lines = o3d.utility.Vector2iVector(lines)
        traj.colors = o3d.utility.Vector3dVector(colors)
    return traj


def build_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    if len(points) == 0:
        return pcd

    pts = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.1, 0.1], (len(pts), 1)))
    return pcd


def has_point_cloud_points(pcd):
    return len(np.asarray(pcd.points)) > 0


def has_lineset_lines(lineset):
    return len(np.asarray(lineset.lines)) > 0 and len(np.asarray(lineset.points)) > 0


def build_colored_lineset(points, lines, colors):
    lineset = o3d.geometry.LineSet()
    if len(points) == 0 or len(lines) == 0:
        return lineset
    lineset.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    lineset.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return lineset


def _color_from_track_id(track_id):
    tid = int(track_id)
    return np.array([
        (37 * tid + 80) % 256,
        (17 * tid + 140) % 256,
        (29 * tid + 200) % 256,
    ], dtype=np.float64) / 255.0


def load_track_export(track_path):
    if track_path is None or not os.path.isfile(track_path):
        return None
    with open(track_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tracks = {}
    for sid_str, items in data.get("tracks", {}).items():
        sid = int(sid_str)
        tracks[sid] = sorted(items, key=lambda x: x["frame_idx"])
    return {
        "total_frames": int(data.get("total_frames", 0)),
        "frame_width": int(data.get("frame_width", 0)),
        "frame_height": int(data.get("frame_height", 0)),
        "tracks": tracks,
    }


def default_track_json_from_map(map_path):
    return os.path.join(os.path.dirname(os.path.abspath(map_path)), "object_tracks.json")


def default_frame_traj_from_map(map_path):
    return os.path.join(os.path.dirname(os.path.abspath(map_path)), "frame_trajectory.txt")


def load_frame_trajectory(path):
    if path is None or not os.path.isfile(path):
        return []

    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            pos_w = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            quat_xyzw = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            rot_wc = R.from_quat(quat_xyzw).as_matrix()
            poses.append({
                "ts": ts,
                "pos_w": pos_w,
                "rot_wc": rot_wc,
            })
    return poses


def estimate_global_track_plane(keyframes):
    positions = np.asarray([kf["pos_w"] for kf in keyframes], dtype=np.float64)
    if len(positions) < 3:
        return positions[0], np.array([0.0, 1.0, 0.0], dtype=np.float64)

    center = positions.mean(axis=0)
    centered = positions - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-6)

    # Keep the normal orientation stable with the first camera up vector.
    first_up = keyframes[0]["rot_wc"] @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if float(np.dot(normal, first_up)) < 0.0:
        normal = -normal

    return center, normal


def estimate_pose_frame_step(frame_poses, track_data, override_step):
    if override_step is not None and override_step > 0:
        return float(override_step)
    if not frame_poses or track_data is None:
        return 1.0
    fps = float(track_data.get("fps", 0.0))
    if fps <= 0.0:
        return 1.0
    timestamps = np.asarray([pose["ts"] for pose in frame_poses], dtype=np.float64)
    if len(timestamps) < 2:
        return 1.0
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-9]
    if len(deltas) == 0:
        return 1.0
    median_dt = float(np.median(deltas))
    estimated = median_dt * fps
    return max(1.0, estimated)

class MapGuiApp:
    def __init__(
        self,
        msg_path,
        frame_ms=50,
        frustum_scale=0.35,
        follow_view=False,
        point_size=4.0,
        pan_sensitivity=4.0,
        track_path=None,
        track_radius=6.0,
        frame_traj_path=None,
        slam_frame_skip=None,
    ):
        self.keyframes, self.lm_groups, self.lm_first_orders = load_map(msg_path)
        if not self.keyframes:
            raise RuntimeError("No keyframes found in map.")

        self.frame_ms = int(frame_ms)
        self.frustum_scale = float(frustum_scale)
        self.follow_view = bool(follow_view)
        self.point_size = float(point_size)
        self.pan_sensitivity = float(pan_sensitivity)
        self.track_radius = float(track_radius)
        self.frame_poses = load_frame_trajectory(frame_traj_path)
        self.track_data = load_track_export(track_path)
        self.track_plane_point, self.track_plane_normal = estimate_global_track_plane(self.keyframes)
        self.pose_frame_step = estimate_pose_frame_step(
            self.frame_poses,
            self.track_data,
            slam_frame_skip,
        )
        self.show_object_tracks = self.track_data is not None
        self.selected_track_ids = set()
        if self.track_data is not None:
            self.selected_track_ids = set(sorted(self.track_data["tracks"].keys()))
        self._track_checkboxes = {}

        self.cur_index = 0
        self.playing = True
        self._slider_updating = False
        self._closed = False
        self._animation_thread = None
        self._last_middle_down = None
        self._middle_dragging = False
        self._drag_threshold = 4.0
        self._lookat = np.asarray(self.keyframes[0]["pos_w"], dtype=np.float64)

        self._landmark_prefix = []
        total = 0
        self._ordered_landmarks = []
        for first_order in self.lm_first_orders:
            group = self.lm_groups[first_order]
            self._ordered_landmarks.extend(group)
            total += len(group)
            self._landmark_prefix.append((first_order, total))

        self._all_points_for_bbox = self._compute_full_bbox_points()
        bbox_min = self._all_points_for_bbox.min(axis=0)
        bbox_max = self._all_points_for_bbox.max(axis=0)
        self._scene_extent = float(np.linalg.norm(bbox_max - bbox_min))

        app = gui.Application.instance
        app.initialize()

        self.window = app.create_window("Map GUI Viewer", 1600, 960)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.98, 0.98, 0.99, 1.0])
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene_widget.set_on_mouse(self._on_mouse)

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))

        self.play_btn = gui.Button("Pause")
        self.play_btn.set_on_clicked(self._toggle_play)
        self.panel.add_child(self.play_btn)

        self.follow_checkbox = gui.Checkbox("Follow Current Frame")
        self.follow_checkbox.checked = self.follow_view
        self.follow_checkbox.set_on_checked(self._on_follow_checked)
        self.panel.add_child(self.follow_checkbox)

        self.track_checkbox = gui.Checkbox("Show Object Tracks")
        self.track_checkbox.checked = self.show_object_tracks
        self.track_checkbox.set_on_checked(self._on_track_visibility_checked)
        self.panel.add_child(self.track_checkbox)

        self.track_select_all_btn = gui.Button("Select All Tracks")
        self.track_select_all_btn.set_on_clicked(self._select_all_tracks)
        self.panel.add_child(self.track_select_all_btn)

        self.track_clear_all_btn = gui.Button("Clear All Tracks")
        self.track_clear_all_btn.set_on_clicked(self._clear_all_tracks)
        self.panel.add_child(self.track_clear_all_btn)

        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, len(self.keyframes) - 1)
        self.frame_slider.int_value = 0
        self.frame_slider.set_on_value_changed(self._on_slider_changed)
        self.panel.add_child(gui.Label("Frame"))
        self.panel.add_child(self.frame_slider)

        self.info_label = gui.Label("")
        self.panel.add_child(self.info_label)

        self.help_label = gui.Label(
            "Mouse:\n"
            "Left drag: rotate\n"
            "Wheel: zoom\n"
            "Middle drag: pan\n"
            "Right click: focus clicked point"
        )
        self.panel.add_child(self.help_label)

        self.panel.add_child(gui.Label("Visible Objects"))
        self.track_list = gui.Vert(0.2 * em, gui.Margins(0, 0, 0, 0))
        self.panel.add_child(self.track_list)
        self._build_track_selector_ui()

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        self._setup_scene()
        self._apply_frame(0, reset_camera=True)
        self._start_animation()

    def _compute_full_bbox_points(self):
        pts = [kf["pos_w"] for kf in self.keyframes]
        pts.extend(self._ordered_landmarks)
        return np.asarray(pts, dtype=np.float64) if pts else np.zeros((1, 3), dtype=np.float64)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(18 * layout_context.theme.font_size)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    def _on_close(self):
        self._closed = True
        self.playing = False
        return True

    def _setup_scene(self):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coord_mat = rendering.MaterialRecord()
        coord_mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("coord", coord, coord_mat)

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            self._all_points_for_bbox.min(axis=0),
            self._all_points_for_bbox.max(axis=0),
        )
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.text = "Pause" if self.playing else "Play"

    def _on_follow_checked(self, checked):
        self.follow_view = checked

    def _on_track_visibility_checked(self, checked):
        self.show_object_tracks = bool(checked)
        self._apply_frame(self.cur_index, reset_camera=False)

    def _select_all_tracks(self):
        if self.track_data is None:
            return
        self.selected_track_ids = set(self.track_data["tracks"].keys())
        for sid, checkbox in self._track_checkboxes.items():
            checkbox.checked = sid in self.selected_track_ids
        self._apply_frame(self.cur_index, reset_camera=False)

    def _clear_all_tracks(self):
        self.selected_track_ids = set()
        for checkbox in self._track_checkboxes.values():
            checkbox.checked = False
        self._apply_frame(self.cur_index, reset_camera=False)

    def _on_track_selected(self, sid, checked):
        if checked:
            self.selected_track_ids.add(sid)
        else:
            self.selected_track_ids.discard(sid)
        self._apply_frame(self.cur_index, reset_camera=False)

    def _build_track_selector_ui(self):
        if self.track_data is None:
            self.track_list.add_child(gui.Label("No object tracks loaded"))
            return

        for sid in sorted(self.track_data["tracks"].keys()):
            checkbox = gui.Checkbox(f"Object {sid}")
            checkbox.checked = sid in self.selected_track_ids
            checkbox.set_on_checked(lambda checked, sid=sid: self._on_track_selected(sid, checked))
            self._track_checkboxes[sid] = checkbox
            self.track_list.add_child(checkbox)

    def _on_slider_changed(self, value):
        if self._slider_updating:
            return
        self.cur_index = int(value)
        self._apply_frame(self.cur_index, reset_camera=False)

    def _start_animation(self):
        def worker():
            while not self._closed:
                if not self.playing:
                    time.sleep(0.05)
                    continue
                next_index = min(self.cur_index + 1, len(self.keyframes) - 1)
                if next_index == self.cur_index:
                    self.playing = False
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: setattr(self.play_btn, "text", "Play"),
                    )
                    continue

                self.cur_index = next_index
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda idx=next_index: self._apply_frame(idx, reset_camera=False),
                )
                time.sleep(self.frame_ms / 1000.0)

        self._animation_thread = threading.Thread(target=worker, daemon=True)
        self._animation_thread.start()

    def _num_landmarks_until(self, keyframe_order):
        total = 0
        for first_order, count in self._landmark_prefix:
            if first_order <= keyframe_order:
                total = count
            else:
                break
        return total

    def _set_camera_from_keyframe(self, kf):
        forward = kf["rot_wc"] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = kf["rot_wc"] @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        up = up / max(np.linalg.norm(up), 1e-6)

        lookat = kf["pos_w"] + 1.8 * forward + 0.3 * up
        eye = lookat - 3.0 * forward
        self._lookat = np.asarray(lookat, dtype=np.float64)
        self.scene_widget.look_at(lookat, eye, up)

    def _frame_pose_for_video_frame(self, video_frame_idx):
        if self.frame_poses:
            pose_index = float(video_frame_idx) / max(self.pose_frame_step, 1e-6)
            pose_index = float(np.clip(pose_index, 0.0, len(self.frame_poses) - 1))
            low = int(np.floor(pose_index))
            high = int(np.ceil(pose_index))
            if low == high:
                return self.frame_poses[low]

            alpha = float(pose_index - low)
            pose_a = self.frame_poses[low]
            pose_b = self.frame_poses[high]
            pos_w = (1.0 - alpha) * pose_a["pos_w"] + alpha * pose_b["pos_w"]
            slerp = Slerp(
                [0.0, 1.0],
                R.from_matrix([pose_a["rot_wc"], pose_b["rot_wc"]]),
            )
            rot_wc = slerp([alpha]).as_matrix()[0]
            ts = (1.0 - alpha) * pose_a["ts"] + alpha * pose_b["ts"]
            return {"ts": ts, "pos_w": pos_w, "rot_wc": rot_wc}
        if self.track_data is None or len(self.keyframes) <= 1:
            return {
                "ts": self.keyframes[0]["ts"],
                "pos_w": self.keyframes[0]["pos_w"],
                "rot_wc": self.keyframes[0]["rot_wc"],
            }
        total_video_frames = max(self.track_data.get("total_frames", 0), 1)
        if total_video_frames <= 1:
            return {
                "ts": self.keyframes[0]["ts"],
                "pos_w": self.keyframes[0]["pos_w"],
                "rot_wc": self.keyframes[0]["rot_wc"],
            }
        progress = float(video_frame_idx) / float(total_video_frames - 1)
        kf_idx = int(np.clip(round(progress * (len(self.keyframes) - 1)), 0, len(self.keyframes) - 1))
        kf = self.keyframes[kf_idx]
        return {"ts": kf["ts"], "pos_w": kf["pos_w"], "rot_wc": kf["rot_wc"]}

    def _track_world_point(self, pose, bbox_xyxy, frame_width, frame_height):
        if frame_width <= 0 or frame_height <= 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        u = 0.5 * (x1 + x2)
        v = y2

        panel_w = frame_width / 4.0
        panel_h = float(frame_height)
        if panel_w <= 1.0 or panel_h <= 1.0:
            return None

        view_idx = int(np.clip(np.floor(u / panel_w), 0, 3))
        u_local = u - view_idx * panel_w
        u_local = float(np.clip(u_local, 0.0, panel_w - 1.0))
        v_local = float(np.clip(v, 0.0, panel_h - 1.0))

        f = 0.5 * panel_w / np.tan(np.deg2rad(90.0) * 0.5)
        cx = 0.5 * panel_w
        cy = 0.5 * panel_h

        x_cam = (u_local - cx) / max(f, 1e-6)
        y_cam = (v_local - cy) / max(f, 1e-6)
        dir_panel = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
        dir_panel = dir_panel / max(np.linalg.norm(dir_panel), 1e-6)

        yaw_centers_deg = [0.0, 90.0, 180.0, 270.0]
        yaw = np.deg2rad(yaw_centers_deg[view_idx])
        rot_y = np.array([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ], dtype=np.float64)
        dir_cam = rot_y @ dir_panel
        dir_cam = dir_cam / max(np.linalg.norm(dir_cam), 1e-6)

        origin = np.asarray(pose["pos_w"], dtype=np.float64)
        dir_world = pose["rot_wc"] @ dir_cam
        dir_world = dir_world / max(np.linalg.norm(dir_world), 1e-6)

        plane_normal = self.track_plane_normal
        plane_point = self.track_plane_point
        denom = float(np.dot(plane_normal, dir_world))
        if abs(denom) > 1e-6:
            t = float(np.dot(plane_normal, plane_point - origin) / denom)
            if t > 0:
                return origin + t * dir_world

        fallback = origin + self.track_radius * dir_world
        signed_dist = float(np.dot(plane_normal, fallback - plane_point))
        return fallback - signed_dist * plane_normal

    def _build_track_lineset(self, current_keyframe_idx):
        if self.track_data is None or not self.show_object_tracks:
            return o3d.geometry.LineSet()

        current_ts = self.keyframes[current_keyframe_idx]["ts"]
        points = []
        lines = []
        colors = []
        point_offset = 0

        frame_width = int(self.track_data.get("frame_width", 0))
        frame_height = int(self.track_data.get("frame_height", 0))

        for sid, items in self.track_data["tracks"].items():
            if sid not in self.selected_track_ids:
                continue
            traj = []
            for item in items:
                pose = self._frame_pose_for_video_frame(item["frame_idx"])
                if pose["ts"] > current_ts:
                    break
                if item.get("ghost", False):
                    continue
                if "world_point" in item:
                    world = np.asarray(item["world_point"], dtype=np.float64)
                else:
                    world = self._track_world_point(
                        pose,
                        item["bbox_xyxy"],
                        frame_width,
                        frame_height,
                    )
                if world is None:
                    continue
                traj.append(world)

            if len(traj) < 2:
                continue

            color = _color_from_track_id(sid)
            points.extend(traj)
            for i in range(len(traj) - 1):
                lines.append([point_offset + i, point_offset + i + 1])
                colors.append(color)
            point_offset += len(traj)

        return build_colored_lineset(points, lines, colors)

    def _apply_frame(self, idx, reset_camera=False):
        idx = max(0, min(idx, len(self.keyframes) - 1))
        kf = self.keyframes[idx]

        traj_points = [x["pos_w"] for x in self.keyframes[:idx + 1]]
        lm_count = self._num_landmarks_until(idx)
        lm_points = self._ordered_landmarks[:lm_count] if lm_count > 0 else np.empty((0, 3), dtype=np.float64)

        traj = build_trajectory_line(traj_points)
        pcd = build_point_cloud(lm_points)
        frustum = make_frustum(kf["pos_w"], kf["rot_wc"], scale=self.frustum_scale)
        obj_tracks = self._build_track_lineset(idx)

        pcd_mat = rendering.MaterialRecord()
        pcd_mat.shader = "defaultUnlit"
        pcd_mat.point_size = self.point_size

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0

        for name in ("pcd", "traj", "frustum", "obj_tracks"):
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)

        if has_point_cloud_points(pcd):
            self.scene_widget.scene.add_geometry("pcd", pcd, pcd_mat)
        if has_lineset_lines(traj):
            self.scene_widget.scene.add_geometry("traj", traj, line_mat)
        if has_lineset_lines(frustum):
            self.scene_widget.scene.add_geometry("frustum", frustum, line_mat)
        if has_lineset_lines(obj_tracks):
            self.scene_widget.scene.add_geometry("obj_tracks", obj_tracks, line_mat)

        if reset_camera or self.follow_view:
            self._set_camera_from_keyframe(kf)

        self._slider_updating = True
        self.frame_slider.int_value = idx
        self._slider_updating = False
        self.info_label.text = (
            f"Frame: {idx + 1}/{len(self.keyframes)}\n"
            f"Keyframe ID: {kf['id']}\n"
            f"Visible landmarks: {len(lm_points)}\n"
            f"Object tracks: {0 if self.track_data is None else len(self.track_data['tracks'])}\n"
            f"Selected tracks: {len(self.selected_track_ids) if self.track_data is not None else 0}"
        )

    def _camera_eye_up(self):
        cam = self.scene_widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        view_inv = np.linalg.inv(view)
        eye = view_inv[:3, 3]
        up = view_inv[:3, 1]
        up = up / max(np.linalg.norm(up), 1e-6)
        return eye, up

    def _camera_eye_right_up(self):
        cam = self.scene_widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        view_inv = np.linalg.inv(view)
        eye = view_inv[:3, 3]
        right = view_inv[:3, 0]
        up = view_inv[:3, 1]
        right = right / max(np.linalg.norm(right), 1e-6)
        up = up / max(np.linalg.norm(up), 1e-6)
        return eye, right, up

    def _focus_on_world_point(self, world):
        eye, up = self._camera_eye_up()
        self._lookat = np.asarray(world, dtype=np.float64)
        self.scene_widget.look_at(self._lookat, eye, up)

    def _pan_camera(self, dx, dy):
        eye, right, up = self._camera_eye_right_up()
        lookat = np.asarray(self._lookat, dtype=np.float64)
        dist = max(np.linalg.norm(eye - lookat), 1e-3)
        reference_dist = max(dist, 0.03 * max(self._scene_extent, 1e-3))
        fov_deg = float(self.scene_widget.scene.camera.get_field_of_view())
        viewport_w = max(float(self.scene_widget.frame.width), 1.0)
        viewport_h = max(float(self.scene_widget.frame.height), 1.0)
        world_per_pixel_y = 2.0 * reference_dist * np.tan(np.deg2rad(fov_deg) * 0.5) / viewport_h
        world_per_pixel_x = world_per_pixel_y * (viewport_w / viewport_h)

        delta = (
            -dx * right * world_per_pixel_x +
            dy * up * world_per_pixel_y
        ) * self.pan_sensitivity
        new_eye = eye + delta
        new_lookat = lookat + delta
        self._lookat = new_lookat
        self.scene_widget.look_at(new_lookat, new_eye, up)

    def _on_depth(self, local_x, local_y, depth_image):
        depth = np.asarray(depth_image)
        h, w = depth.shape[:2]
        x = int(np.clip(local_x, 0, w - 1))
        y = int(np.clip(local_y, 0, h - 1))
        z = float(depth[y, x])
        if not np.isfinite(z) or z >= 1.0:
            return

        world = self.scene_widget.scene.camera.unproject(
            float(x),
            float(y),
            z,
            float(w),
            float(h),
        )
        gui.Application.instance.post_to_main_thread(
            self.window,
            lambda: self._focus_on_world_point(world),
        )

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.MIDDLE):
            self._last_middle_down = (float(event.x), float(event.y))
            self._middle_dragging = False
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.DRAG and event.is_button_down(gui.MouseButton.MIDDLE):
            if self._last_middle_down is None:
                self._last_middle_down = (float(event.x), float(event.y))
                return gui.Widget.EventCallbackResult.HANDLED

            prev_x, prev_y = self._last_middle_down
            dx = float(event.x) - prev_x
            dy = float(event.y) - prev_y
            if abs(dx) + abs(dy) > self._drag_threshold:
                self._middle_dragging = True
            self._pan_camera(dx, dy)
            self._last_middle_down = (float(event.x), float(event.y))
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and self._last_middle_down is not None:
            self._last_middle_down = None
            self._middle_dragging = False
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            local_x = event.x - self.scene_widget.frame.x
            local_y = event.y - self.scene_widget.frame.y
            if local_x < 0 or local_y < 0:
                return gui.Widget.EventCallbackResult.IGNORED
            self.scene_widget.scene.scene.render_to_depth_image(
                lambda depth: self._on_depth(local_x, local_y, depth)
            )
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        gui.Application.instance.run()


def main():
    parser = argparse.ArgumentParser(description="GUI map viewer with mouse-focus navigation")
    parser.add_argument("--map", required=True, help="Path to map msgpack")
    parser.add_argument("--frame-ms", type=int, default=50, help="Animation step interval in milliseconds")
    parser.add_argument("--frustum-scale", type=float, default=0.35, help="Frustum size scale")
    parser.add_argument("--follow-view", action="store_true", help="Follow current keyframe during animation")
    parser.add_argument("--point-size", type=float, default=4.0, help="Rendered point size")
    parser.add_argument("--pan-sensitivity", type=float, default=4.0, help="Mouse middle-drag pan sensitivity")
    parser.add_argument("--tracks-json", default=None, help="Optional track json path; defaults to sibling object_tracks.json")
    parser.add_argument("--track-radius", type=float, default=6.0, help="Fallback radius used to project 2D tracks into 3D")
    parser.add_argument("--frame-traj", default=None, help="Optional frame trajectory txt path; defaults to sibling frame_trajectory.txt")
    parser.add_argument("--slam-frame-skip", type=float, default=None, help="Raw-video frame step used when exporting frame_trajectory.txt, e.g. 2 for --frame-skip 2")
    args = parser.parse_args()

    track_json_path = args.tracks_json or default_track_json_from_map(args.map)
    frame_traj_path = args.frame_traj or default_frame_traj_from_map(args.map)

    app = MapGuiApp(
        msg_path=args.map,
        frame_ms=args.frame_ms,
        frustum_scale=args.frustum_scale,
        follow_view=args.follow_view,
        point_size=args.point_size,
        pan_sensitivity=args.pan_sensitivity,
        track_path=track_json_path,
        track_radius=args.track_radius,
        frame_traj_path=frame_traj_path,
        slam_frame_skip=args.slam_frame_skip,
    )
    app.run()


if __name__ == "__main__":
    main()
