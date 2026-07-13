import argparse
from collections import OrderedDict
import json
import os
import threading
import time

import cv2
import msgpack
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from scipy.spatial.transform import Rotation as R, Slerp

import tools.Equirec2Perspec as E2P


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


def _rgb_color_from_track_id(track_id):
    return _color_from_track_id(track_id)[::-1]


def load_track_export(track_path):
    if track_path is None or not os.path.isfile(track_path):
        return None
    with open(track_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tracks = {}
    for sid_str, items in data.get("tracks", {}).items():
        sid = int(sid_str)
        tracks[sid] = sorted(items, key=lambda x: x["frame_idx"])
    depth_backend = data.get("depth_backend")
    requires_explicit_world_point = (
        "stereo" in str(depth_backend or "").lower()
        or any(
            "stereo_valid" in item
            for items in tracks.values()
            for item in items
        )
    )
    return {
        "track_path": os.path.abspath(track_path),
        "video_path": data.get("video_path"),
        "total_frames": int(data.get("total_frames", 0)),
        "fps": float(data.get("fps", 0.0)),
        "frame_width": int(data.get("frame_width", 0)),
        "frame_height": int(data.get("frame_height", 0)),
        "depth_backend": depth_backend,
        "coordinate_frame": data.get("coordinate_frame"),
        "requires_explicit_world_point": requires_explicit_world_point,
        "slam_pose_frame_step": float(data.get("slam_pose_frame_step", 0.0)),
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
    timestamps = np.asarray([pose["ts"] for pose in frame_poses], dtype=np.float64)
    if len(timestamps) < 2:
        return 1.0
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-9]
    if len(deltas) == 0:
        return 1.0
    if fps > 0.0:
        observed_step = max(1.0, float(np.median(deltas)) * fps)
    else:
        observed_step = 1.0

    stored_step = float(track_data.get("slam_pose_frame_step", 0.0))
    if stored_step > 0.0:
        # Some exported json files can carry stale step metadata that no longer
        # matches the accompanying frame_trajectory.txt. Prefer the value inferred
        # from timestamps unless the stored value agrees closely.
        if abs(stored_step - observed_step) <= 0.25 * max(1.0, observed_step):
            return stored_step

    return observed_step


def _make_placeholder_thumbnail(text, width=160, height=90):
    image = np.full((height, width, 3), 36, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (92, 92, 92), 1)
    cv2.putText(
        image,
        text,
        (8, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return image


def _fit_thumbnail(image, width=160, height=90):
    if image is None or image.size == 0:
        return _make_placeholder_thumbnail("No Preview", width=width, height=height)

    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x0 = (width - dst_w) // 2
    y0 = (height - dst_h) // 2
    canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized
    return canvas


def _make_b_object_preview_image(distance_text, thumbnail, width, height):
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.95
    thickness = 2
    text_size, _ = cv2.getTextSize(distance_text, font, font_scale, thickness)
    text_x = max(8, (width - text_size[0]) // 2)
    cv2.putText(
        image,
        distance_text,
        (text_x, 44),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    thumb = _fit_thumbnail(thumbnail, width=width, height=max(1, height - 62))
    image[62:62 + thumb.shape[0], :thumb.shape[1]] = thumb
    return image


def _resolve_track_video_path(track_data):
    if track_data is None:
        return None
    video_path = track_data.get("video_path")
    if not video_path:
        return None
    if os.path.isabs(video_path) and os.path.isfile(video_path):
        return video_path

    track_dir = os.path.dirname(track_data.get("track_path", ""))
    candidate = os.path.abspath(os.path.join(track_dir, video_path))
    if os.path.isfile(candidate):
        return candidate

    repo_candidate = os.path.abspath(video_path)
    if os.path.isfile(repo_candidate):
        return repo_candidate
    return None


def _representative_track_item(items):
    for item in items:
        if not item.get("ghost", False):
            return item
    return items[0] if items else None


def build_track_thumbnails(track_data, thumb_width=160, thumb_height=90):
    if track_data is None:
        return {}

    video_path = _resolve_track_video_path(track_data)
    if not video_path:
        return {
            sid: _make_placeholder_thumbnail(f"Object {sid}", width=thumb_width, height=thumb_height)
            for sid in track_data["tracks"].keys()
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            sid: _make_placeholder_thumbnail(f"Object {sid}", width=thumb_width, height=thumb_height)
            for sid in track_data["tracks"].keys()
        }

    thumbnails = {}
    frame_width = int(track_data.get("frame_width", 0))
    frame_height = int(track_data.get("frame_height", 0))
    panel_height = max(1, frame_height)
    panel_width = max(1, frame_width // 4) if frame_width > 0 else panel_height
    angles = [0, 90, 180, 270]
    try:
        for sid, items in track_data["tracks"].items():
            item = _representative_track_item(items)
            if item is None:
                thumbnails[sid] = _make_placeholder_thumbnail(
                    f"Object {sid}",
                    width=thumb_width,
                    height=thumb_height,
                )
                continue

            frame_idx = int(item.get("frame_idx", 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                thumbnails[sid] = _make_placeholder_thumbnail(
                    f"Object {sid}",
                    width=thumb_width,
                    height=thumb_height,
                )
                continue

            equ = E2P.Equirectangular(frame)
            combined_frame = np.hstack([
                equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=panel_height, width=panel_width)
                for theta in angles
            ])

            x1, y1, x2, y2 = [int(round(v)) for v in item.get("bbox_xyxy", [0, 0, 0, 0])]
            h, w = combined_frame.shape[:2]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_x = max(8, int(round(0.2 * bw)))
            pad_y = max(8, int(round(0.2 * bh)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = combined_frame[y1:y2, x1:x2]

            if crop.size == 0:
                thumb = _make_placeholder_thumbnail(f"Object {sid}", width=thumb_width, height=thumb_height)
            else:
                color = (_color_from_track_id(sid) * 255.0).astype(np.uint8)
                cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), color.tolist(), 2)
                cv2.putText(
                    crop,
                    f"Object {sid}",
                    (8, min(crop.shape[0] - 10, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                thumb = _fit_thumbnail(crop, width=thumb_width, height=thumb_height)

            thumbnails[sid] = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

    return thumbnails


def load_video_metadata(video_path):
    if video_path is None or not os.path.isfile(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        return None
    return {
        "path": os.path.abspath(video_path),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "aspect": float(width) / float(height),
    }

class MapGuiApp:
    def __init__(
        self,
        msg_path,
        fps=30.0,
        frustum_scale=0.35,
        follow_view=False,
        point_size=4.0,
        pan_sensitivity=4.0,
        track_path=None,
        overlay_video_path=None,
        track_radius=6.0,
        frame_traj_path=None,
        slam_frame_skip=None,
        overlay_rgb_only=False,
    ):
        self.keyframes, self.lm_groups, self.lm_first_orders = load_map(msg_path)
        if not self.keyframes:
            raise RuntimeError("No keyframes found in map.")

        self.fps = max(float(fps), 1e-6)
        self.frame_dt = 1.0 / self.fps
        self.frustum_scale = float(frustum_scale)
        self.follow_view = bool(follow_view)
        self.point_size = float(point_size)
        self.pan_sensitivity = float(pan_sensitivity)
        self.track_radius = float(track_radius)
        self.overlay_video_info = load_video_metadata(overlay_video_path)
        self.overlay_rgb_only = bool(overlay_rgb_only)
        self.overlay_video_cap = None
        self._overlay_last_frame_idx = None
        self._overlay_capture_next_idx = None
        self._overlay_frame_cache = OrderedDict()
        self._overlay_frame_cache_limit = 48
        self.frame_poses = load_frame_trajectory(frame_traj_path)
        self.track_data = load_track_export(track_path)
        self.overlay_display_aspect = None
        if self.overlay_video_info is not None:
            display_height = int(self.overlay_video_info["height"])
            if self.overlay_rgb_only:
                tracked_height = int((self.track_data or {}).get("frame_height", 0))
                display_height = tracked_height if tracked_height > 0 else max(1, display_height // 2)
            self.overlay_display_aspect = (
                float(self.overlay_video_info["width"]) / max(display_height, 1)
            )
        trajectory_fps = self.fps
        if self.overlay_video_info is not None and self.overlay_video_info.get("fps", 0.0) > 0.0:
            trajectory_fps = float(self.overlay_video_info["fps"])
        elif self.track_data is not None and self.track_data.get("fps", 0.0) > 0.0:
            trajectory_fps = float(self.track_data["fps"])
        if self.frame_poses:
            first_pose_ts = float(self.frame_poses[0]["ts"])
            self._pose_video_frames = np.rint(
                [(float(pose["ts"]) - first_pose_ts) * trajectory_fps for pose in self.frame_poses]
            ).astype(np.int64)
        else:
            self._pose_video_frames = np.empty(0, dtype=np.int64)
        self.track_plane_point, self.track_plane_normal = estimate_global_track_plane(self.keyframes)
        self.pose_frame_step = estimate_pose_frame_step(
            self.frame_poses,
            self.track_data,
            slam_frame_skip,
        )
        if self.overlay_video_info is not None and int(self.overlay_video_info.get("frame_count", 0)) > 0:
            self.total_frames = int(self.overlay_video_info["frame_count"])
        elif self.track_data is not None and int(self.track_data.get("total_frames", 0)) > 0:
            self.total_frames = int(self.track_data["total_frames"])
        elif self.frame_poses:
            self.total_frames = int(round((len(self.frame_poses) - 1) * max(self.pose_frame_step, 1.0))) + 1
        else:
            self.total_frames = len(self.keyframes)
        self.show_object_tracks = self.track_data is not None
        self.selected_track_ids = set()
        if self.track_data is not None:
            self.selected_track_ids = set(sorted(self.track_data["tracks"].keys()))
        self.track_thumbnails = build_track_thumbnails(self.track_data)
        self._track_checkboxes = {}
        self._track_distance_labels = {}
        self._ego_scale = 1.0
        self._distance_object_a = 0
        self._distance_object_b = 1

        self.cur_index = 0
        self.playing = True
        self._slider_updating = False
        self._slider_seek_timer = None
        self._slider_seek_generation = 0
        self._closed = False
        self._animation_thread = None
        self._frame_update_pending = False
        self._last_middle_down = None
        self._middle_dragging = False
        self._drag_threshold = 4.0
        self._last_b_preview_key = None
        self._rendered_landmark_key = None
        self._object_frustum_names = {}
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

        window_width = 1600
        window_height = 960
        if self.overlay_video_info is not None:
            window_height += int(round((window_width * 0.5) / self.overlay_display_aspect))
            self.overlay_video_cap = cv2.VideoCapture(self.overlay_video_info["path"])

        self.window = app.create_window("Map GUI Viewer", window_width, window_height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.video_widget = None
        self.b_object_panel = None
        self.b_object_image = None
        self._b_object_preview_size = (360, 450)
        if self.overlay_video_info is not None:
            placeholder_height = int(self.overlay_video_info["height"])
            if self.overlay_rgb_only:
                placeholder_height = int((self.track_data or {}).get("frame_height", 0)) or max(
                    1, placeholder_height // 2
                )
            placeholder_width = min(int(self.overlay_video_info["width"]), 1280)
            placeholder_height = max(
                1,
                int(round(placeholder_height * placeholder_width / self.overlay_video_info["width"])),
            )
            placeholder = np.zeros(
                (placeholder_height, placeholder_width, 3),
                dtype=np.uint8,
            )
            self.video_widget = gui.ImageWidget(o3d.geometry.Image(np.ascontiguousarray(placeholder)))
            self.window.add_child(self.video_widget)
            self.b_object_panel = gui.Vert(0.25 * self.window.theme.font_size, gui.Margins(0, 0, 0, 0))
            preview = _make_b_object_preview_image(
                "Distance to Ego: N/A",
                _make_placeholder_thumbnail("Object B"),
                *self._b_object_preview_size,
            )
            self.b_object_image = gui.ImageWidget(
                o3d.geometry.Image(np.ascontiguousarray(preview))
            )
            self.b_object_panel.add_child(self.b_object_image)
            self.window.add_child(self.b_object_panel)

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
        self.frame_slider.set_limits(0, self.total_frames - 1)
        self.frame_slider.int_value = 0
        self.frame_slider.set_on_value_changed(self._on_slider_changed)
        self.panel.add_child(gui.Label("Video Frame"))
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

        self.panel.add_child(gui.Label("Ego Scale"))
        self.ego_scale_input = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ego_scale_input.double_value = self._ego_scale
        self.ego_scale_input.set_on_value_changed(self._on_ego_scale_changed)
        self.panel.add_child(self.ego_scale_input)

        self.panel.add_child(gui.Label("Objects Distance"))
        object_distance_row_a = gui.Horiz(0.4 * em, gui.Margins(0, 0, 0, 0))
        object_distance_row_a.add_child(gui.Label("A"))
        self.distance_object_a_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.distance_object_a_input.int_value = self._distance_object_a
        self.distance_object_a_input.set_on_value_changed(self._on_distance_object_a_changed)
        object_distance_row_a.add_child(self.distance_object_a_input)
        self.panel.add_child(object_distance_row_a)

        object_distance_row_b = gui.Horiz(0.4 * em, gui.Margins(0, 0, 0, 0))
        object_distance_row_b.add_child(gui.Label("B"))
        self.distance_object_b_input = gui.NumberEdit(gui.NumberEdit.INT)
        self.distance_object_b_input.int_value = self._distance_object_b
        self.distance_object_b_input.set_on_value_changed(self._on_distance_object_b_changed)
        object_distance_row_b.add_child(self.distance_object_b_input)
        self.panel.add_child(object_distance_row_b)
        self.object_distance_hint_label = gui.Label("0 = ego, 1..N = object id")
        self.panel.add_child(self.object_distance_hint_label)
        self.object_distance_label = gui.Label("Distance: N/A")
        self.panel.add_child(self.object_distance_label)

        self.panel.add_child(gui.Label("Visible Objects"))
        self.track_list_scroll = gui.ScrollableVert(0.2 * em, gui.Margins(0, 0, 0, 0))
        self.track_list = gui.Vert(0.2 * em, gui.Margins(0, 0, 0, 0))
        self.track_list_scroll.add_child(self.track_list)
        self.panel.add_child(self.track_list_scroll)
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
        video_height = 0
        if self.video_widget is not None and self.overlay_video_info is not None:
            video_width = int(round(r.width * 0.5))
            video_height = int(round(video_width / self.overlay_display_aspect))
            video_x = r.x + (r.width - video_width) // 2
            self.video_widget.frame = gui.Rect(video_x, r.y, video_width, video_height)
            if self.b_object_panel is not None:
                margin = int(0.5 * layout_context.theme.font_size)
                preview_width = max(160, video_x - r.x - 2 * margin)
                preview_height = max(1, video_height - 2 * margin)
                old_preview_size = self._b_object_preview_size
                self._b_object_preview_size = (preview_width, preview_height)
                self.b_object_panel.frame = gui.Rect(
                    r.x + margin,
                    r.y + margin,
                    preview_width,
                    preview_height,
                )
                if old_preview_size != self._b_object_preview_size:
                    pose = self._frame_pose_for_video_frame(self.cur_index)
                    self._update_b_object_preview(self.cur_index, np.asarray(pose["pos_w"], dtype=np.float64))
        bottom_y = r.y + video_height
        bottom_h = max(1, r.height - video_height)
        self.scene_widget.frame = gui.Rect(r.x, bottom_y, r.width - panel_width, bottom_h)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, bottom_y, panel_width, bottom_h)

    def _on_close(self):
        self._closed = True
        self.playing = False
        if self.overlay_video_cap is not None:
            self.overlay_video_cap.release()
            self.overlay_video_cap = None
        if self._slider_seek_timer is not None:
            self._slider_seek_timer.cancel()
            self._slider_seek_timer = None
        return True

    def _setup_scene(self):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        coord_mat = rendering.MaterialRecord()
        coord_mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry("coord", coord, coord_mat)

        frustum_mat = rendering.MaterialRecord()
        frustum_mat.shader = "unlitLine"
        frustum_mat.line_width = 4.0
        ego_frustum = make_frustum(
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            scale=self.frustum_scale,
        )
        self.scene_widget.scene.add_geometry("frustum", ego_frustum, frustum_mat)

        if self.track_data is not None:
            for sid in sorted(self.track_data["tracks"]):
                name = f"obj_frustum_{sid}"
                object_frustum = make_frustum(
                    np.zeros(3, dtype=np.float64),
                    np.eye(3, dtype=np.float64),
                    scale=self.frustum_scale,
                )
                color = np.array([1.0, 0.45, 0.0], dtype=np.float64)
                object_frustum.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(object_frustum.lines), 1))
                )
                self.scene_widget.scene.add_geometry(name, object_frustum, frustum_mat)
                self.scene_widget.scene.show_geometry(name, False)
                self._object_frustum_names[sid] = name
        self._reset_camera_to_scene_bounds()

    def _reset_camera_to_scene_bounds(self):
        bbox_points = self._scale_map_points(self._all_points_for_bbox)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            bbox_points.min(axis=0),
            bbox_points.max(axis=0),
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

    def _on_ego_scale_changed(self, value):
        self._ego_scale = max(float(value), 1e-6)
        self._rendered_landmark_key = None
        self._reset_camera_to_scene_bounds()
        self._apply_frame(self.cur_index, reset_camera=True)

    def _on_distance_object_a_changed(self, value):
        self._distance_object_a = int(value)
        self._apply_frame(self.cur_index, reset_camera=False)

    def _on_distance_object_b_changed(self, value):
        self._distance_object_b = int(value)
        self._apply_frame(self.cur_index, reset_camera=False)

    def _scale_map_point(self, point):
        return self._ego_scale * np.asarray(point, dtype=np.float64)

    def _scale_map_points(self, points):
        return self._ego_scale * np.asarray(points, dtype=np.float64)

    def _scaled_ego_position(self, pose):
        return self._scale_map_point(pose["pos_w"])

    def _scaled_object_world_point_for_item(self, item):
        if "world_point" in item and item.get(
            "world_point_method"
        ) != "median_depth_backprojection_from_perspective":
            return self._scale_map_point(item["world_point"])

        pose = self._frame_pose_for_video_frame(item["frame_idx"])
        scaled_origin = self._scaled_ego_position(pose)

        if "point_cam" in item:
            point_cam = np.asarray(item["point_cam"], dtype=np.float64)
            offset_world = pose["rot_wc"] @ point_cam
            return scaled_origin + offset_world

        if "world_point" in item:
            world_point = np.asarray(item["world_point"], dtype=np.float64)
            if item.get("world_point_method") == "median_depth_backprojection_from_perspective":
                offset_world = world_point - np.asarray(pose["pos_w"], dtype=np.float64)
                return scaled_origin + offset_world
            return self._scale_map_point(world_point)

        if self.track_data.get("requires_explicit_world_point", False):
            return None

        frame_width = int(self.track_data.get("frame_width", 0))
        frame_height = int(self.track_data.get("frame_height", 0))
        world = self._track_world_point(
            pose,
            item["bbox_xyxy"],
            frame_width,
            frame_height,
        )
        if world is None:
            return None
        return self._scale_map_point(world)

    def _build_track_selector_ui(self):
        if self.track_data is None:
            self.track_list.add_child(gui.Label("No object tracks loaded"))
            return

        for sid in sorted(self.track_data["tracks"].keys()):
            item = gui.Vert(0.15 * self.window.theme.font_size, gui.Margins(0, 0, 0, 0))
            thumbnail = self.track_thumbnails.get(sid)
            if thumbnail is not None:
                image_widget = gui.ImageWidget(o3d.geometry.Image(np.ascontiguousarray(thumbnail)))
                item.add_child(image_widget)

            checkbox = gui.Checkbox(f"Object {sid}")
            checkbox.checked = sid in self.selected_track_ids
            checkbox.set_on_checked(lambda checked, sid=sid: self._on_track_selected(sid, checked))
            self._track_checkboxes[sid] = checkbox
            item.add_child(checkbox)
            item.add_child(gui.Label(f"Frames: {len(self.track_data['tracks'][sid])}"))
            distance_label = gui.Label("Distance to ego: N/A")
            self._track_distance_labels[sid] = distance_label
            item.add_child(distance_label)
            self.track_list.add_child(item)

    def _track_world_point_for_item(self, item):
        return self._scaled_object_world_point_for_item(item)

    def _current_track_distance_map(self, current_video_frame_idx, ego_pos_w):
        if self.track_data is None:
            return {}

        ego_pos_w = self._scale_map_point(ego_pos_w)
        distances = {}
        for sid in self.track_data["tracks"].keys():
            world = self._current_track_world_point(current_video_frame_idx, sid)
            if world is None:
                continue
            distances[sid] = float(np.linalg.norm(world - ego_pos_w))
        return distances

    def _current_track_world_point(self, current_video_frame_idx, sid):
        if self.track_data is None:
            return None

        items = self.track_data["tracks"].get(int(sid))
        if not items:
            return None

        explicit_depth_tracks = self.track_data.get("requires_explicit_world_point", False)

        if explicit_depth_tracks:
            for item in items:
                item_frame_idx = int(item["frame_idx"])
                if item_frame_idx > int(current_video_frame_idx):
                    break
                if item_frame_idx != int(current_video_frame_idx):
                    continue
                if item.get("ghost", False) or item.get("measurement_rejected", False):
                    return None
                return self._track_world_point_for_item(item)
            return None

        latest_item = None
        latest_world = None
        for item in items:
            if int(item["frame_idx"]) > int(current_video_frame_idx):
                break
            if item.get("ghost", False):
                continue
            if explicit_depth_tracks and item.get("measurement_rejected", False):
                continue
            world = self._track_world_point_for_item(item)
            if world is None:
                continue
            latest_item = item
            latest_world = world
        if latest_item is None:
            return None

        return latest_world

    def _update_track_distance_labels(self, current_video_frame_idx, ego_pos_w):
        distance_map = self._current_track_distance_map(current_video_frame_idx, ego_pos_w)
        for sid, label in self._track_distance_labels.items():
            distance = distance_map.get(sid)
            if distance is None:
                label.text = "Distance to ego: N/A"
            else:
                label.text = f"Distance to ego: {distance:.2f} m"

    def _distance_object_world_point(self, current_video_frame_idx, sid, ego_pos_w):
        sid = int(sid)
        if sid == 0:
            return self._scale_map_point(ego_pos_w)
        return self._current_track_world_point(current_video_frame_idx, sid)

    def _update_object_distance_label(self, current_video_frame_idx, ego_pos_w):
        sid_a = int(self._distance_object_a)
        sid_b = int(self._distance_object_b)
        world_a = self._distance_object_world_point(current_video_frame_idx, sid_a, ego_pos_w)
        world_b = self._distance_object_world_point(current_video_frame_idx, sid_b, ego_pos_w)
        if world_a is None or world_b is None:
            self.object_distance_label.text = f"Distance {sid_a} <-> {sid_b}: N/A"
            return

        distance = float(np.linalg.norm(world_a - world_b))
        self.object_distance_label.text = f"Distance {sid_a} <-> {sid_b}: {distance:.2f} m"

    def _update_b_object_preview(self, current_video_frame_idx, ego_pos_w):
        if self.b_object_panel is None:
            return

        sid_b = int(self._distance_object_b)
        if sid_b == 0:
            distance = 0.0
            image = _make_placeholder_thumbnail("Ego")
        else:
            distance = self._current_track_distance_map(current_video_frame_idx, ego_pos_w).get(sid_b)
            image = self.track_thumbnails.get(sid_b)
            if image is None:
                image = _make_placeholder_thumbnail(f"Object {sid_b}")

        if distance is None:
            distance_text = "Distance to Ego: N/A"
            distance_bucket = None
        else:
            distance_text = f"Distance to Ego: {distance:.2f} m"
            distance_bucket = round(float(distance) * 20.0) / 20.0
        preview_key = (
            sid_b,
            distance_bucket,
            self._b_object_preview_size,
        )
        if preview_key == self._last_b_preview_key:
            return
        preview = _make_b_object_preview_image(distance_text, image, *self._b_object_preview_size)
        self.b_object_image.update_image(o3d.geometry.Image(np.ascontiguousarray(preview)))
        self._last_b_preview_key = preview_key

    def _on_slider_changed(self, value):
        if self._slider_updating:
            return
        self.playing = False
        self.play_btn.text = "Play"
        self.cur_index = int(value)
        self._slider_seek_generation += 1
        generation = self._slider_seek_generation
        if self._slider_seek_timer is not None:
            self._slider_seek_timer.cancel()

        def request_latest_frame():
            if self._closed or generation != self._slider_seek_generation:
                return

            def apply_latest_frame():
                if self._closed or generation != self._slider_seek_generation:
                    return
                self._apply_frame(self.cur_index, reset_camera=False)

            gui.Application.instance.post_to_main_thread(self.window, apply_latest_frame)

        self._slider_seek_timer = threading.Timer(0.06, request_latest_frame)
        self._slider_seek_timer.daemon = True
        self._slider_seek_timer.start()

    def _start_animation(self):
        def worker():
            while not self._closed:
                if not self.playing:
                    time.sleep(0.05)
                    continue
                if self._frame_update_pending:
                    time.sleep(0.001)
                    continue
                next_index = min(self.cur_index + 1, self.total_frames - 1)
                if next_index == self.cur_index:
                    self.playing = False
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: setattr(self.play_btn, "text", "Play"),
                    )
                    continue

                self.cur_index = next_index
                self._frame_update_pending = True

                def apply_frame(idx=next_index):
                    try:
                        self._apply_frame(idx, reset_camera=False)
                    finally:
                        self._frame_update_pending = False

                gui.Application.instance.post_to_main_thread(
                    self.window,
                    apply_frame,
                )
                time.sleep(self.frame_dt)

        self._animation_thread = threading.Thread(target=worker, daemon=True)
        self._animation_thread.start()

    def _update_overlay_video_frame(self, idx):
        if self.video_widget is None or self.overlay_video_cap is None or self.overlay_video_info is None:
            return
        frame_count = int(self.overlay_video_info.get("frame_count", 0))
        if frame_count > 0:
            idx = int(np.clip(idx, 0, frame_count - 1))
        else:
            idx = max(0, int(idx))
        if self._overlay_last_frame_idx == idx:
            return

        cached = self._overlay_frame_cache.get(idx)
        if cached is not None:
            self._overlay_frame_cache.move_to_end(idx)
            self.video_widget.update_image(o3d.geometry.Image(np.ascontiguousarray(cached)))
            self._overlay_last_frame_idx = idx
            return

        if self._overlay_capture_next_idx != idx:
            self.overlay_video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.overlay_video_cap.read()
        if not ret or frame is None:
            return
        self._overlay_capture_next_idx = idx + 1

        if self.overlay_rgb_only:
            tracked_height = int((self.track_data or {}).get("frame_height", 0))
            rgb_height = tracked_height if tracked_height > 0 else max(1, frame.shape[0] // 2)
            frame = frame[:min(rgb_height, frame.shape[0])]

        max_overlay_width = 1280
        if frame.shape[1] > max_overlay_width:
            scale = max_overlay_width / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (max_overlay_width, max(1, int(round(frame.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._overlay_frame_cache[idx] = rgb
        self._overlay_frame_cache.move_to_end(idx)
        while len(self._overlay_frame_cache) > self._overlay_frame_cache_limit:
            self._overlay_frame_cache.popitem(last=False)
        self.video_widget.update_image(o3d.geometry.Image(np.ascontiguousarray(rgb)))
        self._overlay_last_frame_idx = idx

    def _num_landmarks_until(self, keyframe_order):
        total = 0
        for first_order, count in self._landmark_prefix:
            if first_order <= keyframe_order:
                total = count
            else:
                break
        return total

    def _set_camera_from_pose(self, pos_w, rot_wc):
        pos_w = self._scale_map_point(pos_w)
        forward = rot_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = rot_wc @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        up = up / max(np.linalg.norm(up), 1e-6)

        lookat = pos_w + 1.8 * forward + 0.3 * up
        eye = lookat - 3.0 * forward
        self._lookat = np.asarray(lookat, dtype=np.float64)
        self.scene_widget.look_at(lookat, eye, up)

    def _video_frame_to_keyframe_index(self, video_frame_idx):
        if self.frame_poses:
            pose = self._frame_pose_for_video_frame(video_frame_idx)
            pose_ts = float(pose["ts"])
            keyframe_ts = np.asarray([kf["ts"] for kf in self.keyframes], dtype=np.float64)
            keyframe_idx = int(np.searchsorted(keyframe_ts, pose_ts, side="right") - 1)
            return int(np.clip(keyframe_idx, 0, len(self.keyframes) - 1))
        if self.total_frames <= 1:
            return 0
        progress = float(video_frame_idx) / float(self.total_frames - 1)
        return int(np.clip(round(progress * (len(self.keyframes) - 1)), 0, len(self.keyframes) - 1))

    def _frame_pose_for_video_frame(self, video_frame_idx):
        if self.frame_poses:
            high = int(np.searchsorted(self._pose_video_frames, video_frame_idx, side="left"))
            if high <= 0:
                return self.frame_poses[0]
            if high >= len(self.frame_poses):
                return self.frame_poses[-1]
            if self._pose_video_frames[high] == video_frame_idx:
                return self.frame_poses[high]

            low = high - 1
            if low == high:
                return self.frame_poses[low]

            low_frame = int(self._pose_video_frames[low])
            high_frame = int(self._pose_video_frames[high])
            alpha = float(video_frame_idx - low_frame) / max(high_frame - low_frame, 1)
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

    def _build_track_lineset(self, current_video_frame_idx):
        if self.track_data is None or not self.show_object_tracks:
            return o3d.geometry.LineSet()

        points = []
        lines = []
        colors = []
        point_offset = 0

        frame_width = int(self.track_data.get("frame_width", 0))
        frame_height = int(self.track_data.get("frame_height", 0))

        for sid, items in self.track_data["tracks"].items():
            if sid not in self.selected_track_ids:
                continue
            explicit_depth_tracks = self.track_data.get("requires_explicit_world_point", False)
            traj = []
            for item in items:
                if int(item["frame_idx"]) > int(current_video_frame_idx):
                    break
                if item.get("ghost", False):
                    continue
                if explicit_depth_tracks and item.get("measurement_rejected", False):
                    continue
                world = self._track_world_point_for_item(item)
                if world is None:
                    continue
                traj.append((int(item["frame_idx"]), world))

            if len(traj) < 2:
                continue

            color = _rgb_color_from_track_id(sid)
            points.extend(world for _, world in traj)
            max_gap_frames = max(1, int(round(float(self.fps))))
            for i in range(len(traj) - 1):
                if not explicit_depth_tracks or traj[i + 1][0] - traj[i][0] <= max_gap_frames:
                    lines.append([point_offset + i, point_offset + i + 1])
                    colors.append(color)
            point_offset += len(traj)

        return build_colored_lineset(points, lines, colors)

    def _visual_track_world_point(self, current_video_frame_idx, sid):
        world = self._current_track_world_point(current_video_frame_idx, sid)
        if world is not None:
            return world
        if self.track_data is None or not self.track_data.get(
            "requires_explicit_world_point",
            False,
        ):
            return None

        items = self.track_data["tracks"].get(int(sid), [])
        previous = None
        following = None
        for item in items:
            frame_idx = int(item["frame_idx"])
            if item.get("ghost", False) or item.get("measurement_rejected", False):
                continue
            item_world = self._track_world_point_for_item(item)
            if item_world is None:
                continue
            if frame_idx < int(current_video_frame_idx):
                previous = (frame_idx, item_world)
                continue
            if frame_idx > int(current_video_frame_idx):
                following = (frame_idx, item_world)
                break

        if previous is None or following is None:
            return None
        max_gap_frames = max(1, int(round(0.5 * float(self.fps))))
        frame_gap = following[0] - previous[0]
        if frame_gap > max_gap_frames:
            return None
        alpha = float(current_video_frame_idx - previous[0]) / max(frame_gap, 1)
        return (1.0 - alpha) * previous[1] + alpha * following[1]

    def _frustum_transform(self, world, rot_wc, world_is_scaled=False):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = float(self._ego_scale) * np.asarray(rot_wc, dtype=np.float64)
        transform[:3, 3] = (
            np.asarray(world, dtype=np.float64)
            if world_is_scaled
            else self._scale_map_point(world)
        )
        return transform

    def _update_frustum_transforms(self, current_video_frame_idx, pose):
        self.scene_widget.scene.set_geometry_transform(
            "frustum",
            self._frustum_transform(pose["pos_w"], pose["rot_wc"]),
        )
        for sid, name in self._object_frustum_names.items():
            visible = self.show_object_tracks and sid in self.selected_track_ids
            world = self._visual_track_world_point(current_video_frame_idx, sid) if visible else None
            if world is None:
                self.scene_widget.scene.show_geometry(name, False)
                continue
            self.scene_widget.scene.set_geometry_transform(
                name,
                self._frustum_transform(world, pose["rot_wc"], world_is_scaled=True),
            )
            self.scene_widget.scene.show_geometry(name, True)

    def _apply_frame(self, idx, reset_camera=False):
        idx = max(0, min(idx, self.total_frames - 1))
        self._update_overlay_video_frame(idx)
        pose = self._frame_pose_for_video_frame(idx)
        ego_pos_w = np.asarray(pose["pos_w"], dtype=np.float64)
        self._update_track_distance_labels(idx, ego_pos_w)
        self._update_object_distance_label(idx, ego_pos_w)
        self._update_b_object_preview(idx, ego_pos_w)
        keyframe_idx = self._video_frame_to_keyframe_index(idx)
        kf = self.keyframes[keyframe_idx]

        if self.frame_poses:
            pose_index = int(np.searchsorted(self._pose_video_frames, idx, side="right") - 1)
            pose_index = int(np.clip(pose_index, 0, len(self.frame_poses) - 1))
            traj_points = [self._scale_map_point(x["pos_w"]) for x in self.frame_poses[:pose_index + 1]]
        else:
            traj_points = [self._scale_map_point(x["pos_w"]) for x in self.keyframes[:keyframe_idx + 1]]
        lm_count = self._num_landmarks_until(keyframe_idx)
        landmark_key = (keyframe_idx, lm_count, float(self._ego_scale), float(self.point_size))
        update_landmarks = landmark_key != self._rendered_landmark_key

        traj = build_trajectory_line(traj_points)
        pcd = None
        if update_landmarks:
            lm_points = (
                self._scale_map_points(self._ordered_landmarks[:lm_count])
                if lm_count > 0
                else np.empty((0, 3), dtype=np.float64)
            )
            pcd = build_point_cloud(lm_points)
        obj_tracks = self._build_track_lineset(idx)
        self._update_frustum_transforms(idx, pose)

        pcd_mat = rendering.MaterialRecord()
        pcd_mat.shader = "defaultUnlit"
        pcd_mat.point_size = self.point_size

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 4.0

        for name in ("traj", "obj_tracks"):
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)

        if update_landmarks:
            if self.scene_widget.scene.has_geometry("pcd"):
                self.scene_widget.scene.remove_geometry("pcd")
            if has_point_cloud_points(pcd):
                self.scene_widget.scene.add_geometry("pcd", pcd, pcd_mat)
            self._rendered_landmark_key = landmark_key
        if has_lineset_lines(traj):
            self.scene_widget.scene.add_geometry("traj", traj, line_mat)
        if has_lineset_lines(obj_tracks):
            self.scene_widget.scene.add_geometry("obj_tracks", obj_tracks, line_mat)

        if reset_camera or self.follow_view:
            self._set_camera_from_pose(pose["pos_w"], pose["rot_wc"])

        self._slider_updating = True
        self.frame_slider.int_value = idx
        self._slider_updating = False
        self.info_label.text = (
            f"Video Frame: {idx + 1}/{self.total_frames}\n"
            f"Timestamp: {pose['ts']:.3f}\n"
            f"Matched Keyframe: {keyframe_idx + 1}/{len(self.keyframes)} (ID {kf['id']})\n"
            f"Visible landmarks: {lm_count}\n"
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
        reference_dist = max(dist, 0.03 * max(self._scene_extent * self._ego_scale, 1e-3))
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
    parser.add_argument("--fps", type=float, default=30.0, help="Playback speed in frames per second")
    parser.add_argument("--frustum-scale", type=float, default=0.35, help="Frustum size scale")
    parser.add_argument("--follow-view", action="store_true", help="Follow current keyframe during animation")
    parser.add_argument("--point-size", type=float, default=4.0, help="Rendered point size")
    parser.add_argument("--pan-sensitivity", type=float, default=4.0, help="Mouse middle-drag pan sensitivity")
    parser.add_argument("--tracks-json", default=None, help="Optional track json path; defaults to sibling object_tracks.json")
    parser.add_argument("--overlay-video", default=None, help="Optional synced video shown above the 3D map")
    parser.add_argument(
        "--overlay-rgb-only",
        action="store_true",
        help="Show only the RGB row from a two-row RGB/depth overlay for faster playback.",
    )
    parser.add_argument("--track-radius", type=float, default=6.0, help="Fallback radius used to project 2D tracks into 3D")
    parser.add_argument("--frame-traj", default=None, help="Optional frame trajectory txt path; defaults to sibling frame_trajectory.txt")
    parser.add_argument("--slam-frame-skip", type=float, default=None, help="Raw-video frame step used when exporting frame_trajectory.txt, e.g. 2 for --frame-skip 2")
    args = parser.parse_args()

    track_json_path = args.tracks_json or default_track_json_from_map(args.map)
    frame_traj_path = args.frame_traj or default_frame_traj_from_map(args.map)

    app = MapGuiApp(
        msg_path=args.map,
        fps=args.fps,
        frustum_scale=args.frustum_scale,
        follow_view=args.follow_view,
        point_size=args.point_size,
        pan_sensitivity=args.pan_sensitivity,
        track_path=track_json_path,
        overlay_video_path=args.overlay_video,
        track_radius=args.track_radius,
        frame_traj_path=frame_traj_path,
        slam_frame_skip=args.slam_frame_skip,
        overlay_rgb_only=args.overlay_rgb_only,
    )
    app.run()


if __name__ == "__main__":
    main()
