import argparse
import os

import cv2
import json
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import tools.Equirec2Perspec as E2P

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def _has_intersection(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    return inter_w > 0 and inter_h > 0


def _color_from_track_id(track_id):
    """
    基于 track id 生成稳定颜色；无 id 时使用灰色。
    """
    if track_id is None:
        return (180, 180, 180)
    tid = int(track_id)
    return (
        (37 * tid + 80) % 256,
        (17 * tid + 140) % 256,
        (29 * tid + 200) % 256
    )


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _extract_hsv_feature(image, box, bins_h=24, bins_s=16):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins_h, bins_s], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _circular_x_distance(cx1, cx2, width):
    if width <= 0:
        return abs(cx1 - cx2)
    d = abs(cx1 - cx2)
    return min(d, width - d)


def _appearance_similarity(feat_a, feat_b):
    if feat_a is None or feat_b is None:
        return 0.0
    # Bhattacharyya 距离: 0(相同) -> 1(不同)，转成相似度
    dist = cv2.compareHist(feat_a.astype(np.float32), feat_b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - dist))


class StablePersonTracker:
    """
    对 detector/tracker 输出做稳定化：
    1) 短时丢检保留 (anti-flicker)
    2) IoU 重关联 (减少 ID 跳变)
    3) bbox EMA 平滑
    """
    def __init__(self, frame_width, hold_frames=5, reid_iou_thresh=0.3, smooth_alpha=0.65,
                 max_reid_frames=90, appearance_thresh=0.55):
        self.frame_width = frame_width
        self.hold_frames = hold_frames
        self.reid_iou_thresh = reid_iou_thresh
        self.smooth_alpha = smooth_alpha
        self.max_reid_frames = max_reid_frames
        self.appearance_thresh = appearance_thresh
        self.next_stable_id = 1
        self.tracks = {}  # stable_id -> {"box": np.array(4), "last_seen": int, "last_conf": float, "feat": np.array}
        self.raw_to_stable = {}  # raw person track id -> stable_id

    def _new_track(self, box, frame_idx, conf, feat):
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        self.tracks[stable_id] = {
            "box": np.array(box, dtype=np.float32),
            "last_seen": frame_idx,
            "last_conf": conf,
            "feat": feat.copy() if feat is not None else None
        }
        return stable_id

    def update(self, persons, frame_idx):
        used_stable_ids = set()
        visible_items = []

        for person in persons:
            box = np.array(person["box"], dtype=np.float32)
            raw_id = person.get("raw_id")
            conf = float(person.get("conf", 0.0))
            feat = person.get("feat")

            stable_id = None

            if raw_id is not None and raw_id in self.raw_to_stable:
                sid = self.raw_to_stable[raw_id]
                if sid in self.tracks and sid not in used_stable_ids:
                    stable_id = sid

            if stable_id is None:
                best_iou = 0.0
                best_sid = None
                for sid, track in self.tracks.items():
                    if sid in used_stable_ids:
                        continue
                    if frame_idx - track["last_seen"] > self.hold_frames:
                        continue
                    score = _iou(box, track["box"])
                    if score > best_iou:
                        best_iou = score
                        best_sid = sid
                if best_sid is not None and best_iou >= self.reid_iou_thresh:
                    stable_id = best_sid

            # IoU 关联失败后，使用外观+环形位置做 ReID
            if stable_id is None:
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                best_score = -1.0
                best_sid = None
                for sid, track in self.tracks.items():
                    if sid in used_stable_ids:
                        continue
                    age = frame_idx - track["last_seen"]
                    if age <= 0 or age > self.max_reid_frames:
                        continue

                    sim = _appearance_similarity(feat, track.get("feat"))
                    if sim < self.appearance_thresh:
                        continue

                    tbox = track["box"]
                    tcx = 0.5 * (tbox[0] + tbox[2])
                    tcy = 0.5 * (tbox[1] + tbox[3])
                    dx = _circular_x_distance(cx, tcx, self.frame_width) / max(1.0, self.frame_width * 0.5)
                    dy = abs(cy - tcy) / 480.0
                    pos_score = max(0.0, 1.0 - (0.7 * dx + 0.3 * dy))

                    # 时间越久，分数小幅衰减
                    age_penalty = min(0.3, 0.003 * age)
                    score = 0.75 * sim + 0.25 * pos_score - age_penalty
                    if score > best_score:
                        best_score = score
                        best_sid = sid

                if best_sid is not None:
                    stable_id = best_sid

            if stable_id is None:
                stable_id = self._new_track(box, frame_idx, conf, feat)
            else:
                old_box = self.tracks[stable_id]["box"]
                smoothed_box = self.smooth_alpha * box + (1.0 - self.smooth_alpha) * old_box
                self.tracks[stable_id]["box"] = smoothed_box
                self.tracks[stable_id]["last_seen"] = frame_idx
                self.tracks[stable_id]["last_conf"] = conf
                if feat is not None:
                    old_feat = self.tracks[stable_id].get("feat")
                    if old_feat is None:
                        self.tracks[stable_id]["feat"] = feat.copy()
                    else:
                        self.tracks[stable_id]["feat"] = 0.7 * feat + 0.3 * old_feat

            if raw_id is not None:
                self.raw_to_stable[raw_id] = stable_id

            used_stable_ids.add(stable_id)
            visible_items.append({
                "stable_id": stable_id,
                "box": self.tracks[stable_id]["box"].copy(),
                "ghost": False
            })

        # 输出短时 ghost 框，减少闪烁
        for sid, track in self.tracks.items():
            if sid in used_stable_ids:
                continue
            age = frame_idx - track["last_seen"]
            if 1 <= age <= self.hold_frames:
                visible_items.append({
                    "stable_id": sid,
                    "box": track["box"].copy(),
                    "ghost": True
                })

        # 清理过期 track 和 raw id 映射
        expired = [sid for sid, t in self.tracks.items() if frame_idx - t["last_seen"] > self.max_reid_frames]
        if expired:
            expired_set = set(expired)
            for sid in expired:
                del self.tracks[sid]
            self.raw_to_stable = {
                raw_id: sid for raw_id, sid in self.raw_to_stable.items() if sid not in expired_set
            }

        return visible_items


def _extract_persons_with_merge(result, image):
    """
    从单帧 tracking 结果中提取“只显示人的目标”：
    - 人与车重叠 -> merge 为一个 person 目标
    - 不显示 bicycle 独立框
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()

    detections = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        detections.append({
            "idx": i,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "cls_id": int(classes[i]),
            "conf": float(confs[i]),
            "track_id": int(track_ids[i]) if track_ids is not None else None
        })

    n = len(detections)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = detections[i]["cls_id"], detections[j]["cls_id"]
            if {ci, cj} != {0, 1}:
                continue
            if _has_intersection(detections[i]["box"], detections[j]["box"]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    merged_indices = set()

    persons = []

    # merge 后的人车组合框（以 person 作为主目标）
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        if len(comp) <= 1:
            continue
        comp_cls = {detections[k]["cls_id"] for k in comp}
        if comp_cls != {0, 1}:
            continue

        x1 = min(detections[k]["box"][0] for k in comp)
        y1 = min(detections[k]["box"][1] for k in comp)
        x2 = max(detections[k]["box"][2] for k in comp)
        y2 = max(detections[k]["box"][3] for k in comp)

        person_ids = sorted(set(
            detections[k]["track_id"] for k in comp
            if detections[k]["cls_id"] == 0 and detections[k]["track_id"] is not None
        ))
        primary_id = person_ids[0] if person_ids else None
        person_conf = max(
            (detections[k]["conf"] for k in comp if detections[k]["cls_id"] == 0),
            default=max((detections[k]["conf"] for k in comp), default=0.0)
        )
        persons.append({
            "box": [x1, y1, x2, y2],
            "raw_id": primary_id,
            "conf": person_conf,
            "feat": _extract_hsv_feature(image, [x1, y1, x2, y2])
        })
        merged_indices.update(comp)

    # 未 merge 的普通 person
    for det in detections:
        if det["idx"] in merged_indices:
            continue

        if det["cls_id"] != 0:
            continue

        persons.append({
            "box": det["box"],
            "raw_id": det["track_id"],
            "conf": det["conf"],
            "feat": _extract_hsv_feature(image, det["box"])
        })

    return persons


def draw_stable_persons(image, visible_items):
    for item in visible_items:
        sid = item["stable_id"]
        x1, y1, x2, y2 = item["box"].astype(int).tolist()
        color = _color_from_track_id(sid)
        thickness = 3
        label = f"P{sid}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def _load_map_keyframes(msg_path):
    import msgpack
    from scipy.spatial.transform import Rotation as R

    with open(msg_path, "rb") as f:
        msg = msgpack.Unpacker(f, raw=False).unpack()

    keyframes = []
    for kf in msg["keyframes"].values():
        rot_cw = R.from_quat(kf["rot_cw"]).as_matrix()
        trans_cw = np.array(kf["trans_cw"], dtype=np.float64).reshape(3, 1)
        rot_wc = rot_cw.T
        pos_w = (-rot_wc @ trans_cw).reshape(3)
        keyframes.append({
            "ts": float(kf["ts"]),
            "rot_wc": rot_wc,
            "pos_w": pos_w,
        })
    keyframes.sort(key=lambda x: x["ts"])
    return keyframes


def _estimate_global_track_plane(keyframes):
    positions = np.asarray([kf["pos_w"] for kf in keyframes], dtype=np.float64)
    if len(positions) < 3:
        return positions[0], np.array([0.0, 1.0, 0.0], dtype=np.float64)

    center = positions.mean(axis=0)
    _, _, vh = np.linalg.svd(positions - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-6)
    first_up = keyframes[0]["rot_wc"] @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if float(np.dot(normal, first_up)) < 0.0:
        normal = -normal
    return center, normal


def _load_frame_trajectory(path):
    from scipy.spatial.transform import Rotation as R

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


def _estimate_pose_frame_step(frame_poses, fps, override_step):
    if override_step is not None and override_step > 0:
        return float(override_step)
    if not frame_poses or fps <= 0.0:
        return 1.0
    timestamps = np.asarray([pose["ts"] for pose in frame_poses], dtype=np.float64)
    if len(timestamps) < 2:
        return 1.0
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 1e-9]
    if len(deltas) == 0:
        return 1.0
    return max(1.0, float(np.median(deltas)) * fps)


def _interpolate_pose(frame_poses, video_frame_idx, pose_frame_step):
    from scipy.spatial.transform import Rotation as R, Slerp

    if not frame_poses:
        return None

    pose_index = float(video_frame_idx) / max(pose_frame_step, 1e-6)
    pose_index = float(np.clip(pose_index, 0.0, len(frame_poses) - 1))
    low = int(np.floor(pose_index))
    high = int(np.ceil(pose_index))
    if low == high:
        return frame_poses[low]

    alpha = float(pose_index - low)
    pose_a = frame_poses[low]
    pose_b = frame_poses[high]
    pos_w = (1.0 - alpha) * pose_a["pos_w"] + alpha * pose_b["pos_w"]
    slerp = Slerp([0.0, 1.0], R.from_matrix([pose_a["rot_wc"], pose_b["rot_wc"]]))
    rot_wc = slerp([alpha]).as_matrix()[0]
    ts = (1.0 - alpha) * pose_a["ts"] + alpha * pose_b["ts"]
    return {"ts": ts, "pos_w": pos_w, "rot_wc": rot_wc}


def _build_export_context(map_path, frame_traj_path, fps, slam_frame_skip):
    if not map_path or not frame_traj_path:
        return None

    keyframes = _load_map_keyframes(map_path)
    frame_poses = _load_frame_trajectory(frame_traj_path)
    plane_point, plane_normal = _estimate_global_track_plane(keyframes)
    pose_frame_step = _estimate_pose_frame_step(frame_poses, fps, slam_frame_skip)
    return {
        "frame_poses": frame_poses,
        "plane_point": plane_point,
        "plane_normal": plane_normal,
        "pose_frame_step": pose_frame_step,
        "track_radius": 6.0,
    }


def _world_point_from_plane(item, frame_idx, frame_width, frame_height, export_context):
    if export_context is None:
        return None

    pose = _interpolate_pose(
        export_context["frame_poses"],
        frame_idx,
        export_context["pose_frame_step"],
    )
    if pose is None:
        return None

    x1, y1, x2, y2 = item["box"].astype(float).tolist()
    panel_w = frame_width / 4.0
    panel_h = float(frame_height)
    if panel_w <= 1.0 or panel_h <= 1.0:
        return None

    u = 0.5 * (x1 + x2)
    v = y2
    view_idx = int(np.clip(np.floor(u / panel_w), 0, 3))
    u_local = float(np.clip(u - view_idx * panel_w, 0.0, panel_w - 1.0))
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
    plane_point = export_context["plane_point"]
    plane_normal = export_context["plane_normal"]
    denom = float(np.dot(plane_normal, dir_world))
    if abs(denom) > 1e-6:
        t = float(np.dot(plane_normal, plane_point - origin) / denom)
        if t > 0:
            world_point = origin + t * dir_world
        else:
            world_point = None
    else:
        world_point = None

    if world_point is None:
        fallback = origin + export_context["track_radius"] * dir_world
        signed_dist = float(np.dot(plane_normal, fallback - plane_point))
        world_point = fallback - signed_dist * plane_normal

    return {
        "view_idx": view_idx,
        "local_center_xy": [u_local, 0.5 * (y1 + y2)],
        "ray_world": dir_world.tolist(),
        "world_point": world_point.tolist(),
        "world_point_method": "ground_plane_intersection_from_map",
    }


def _build_track_export_item(item, frame_idx, frame_width, frame_height, export_context=None):
    x1, y1, x2, y2 = item["box"].astype(float).tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    payload = {
        "frame_idx": int(frame_idx),
        "stable_id": int(item["stable_id"]),
        "ghost": bool(item.get("ghost", False)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_xy": [cx, cy],
        "center_x_norm": float(cx / max(frame_width, 1)),
    }
    extra = _world_point_from_plane(item, frame_idx, frame_width, frame_height, export_context)
    if extra is not None:
        payload.update(extra)
    return payload


def _save_track_export(path, video_path, total_frames, fps, frame_width, frame_height, items, export_context=None):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    tracks = {}
    allowed_ids = {1, 2, 3}
    for item in items:
        if int(item["stable_id"]) not in allowed_ids:
            continue
        sid = str(item["stable_id"])
        entry = {
            "frame_idx": item["frame_idx"],
            "ghost": item["ghost"],
            "bbox_xyxy": item["bbox_xyxy"],
            "center_xy": item["center_xy"],
            "center_x_norm": item["center_x_norm"],
        }
        for key in ("view_idx", "local_center_xy", "ray_world", "point_cam", "world_point", "world_point_method"):
            if key in item:
                entry[key] = item[key]
        tracks.setdefault(sid, []).append(entry)

    payload = {
        "video_path": video_path,
        "total_frames": int(total_frames),
        "fps": float(fps),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "tracks": tracks,
    }
    if export_context is not None:
        payload["ground_plane_point"] = export_context["plane_point"].tolist()
        payload["ground_plane_normal"] = export_context["plane_normal"].tolist()
        payload["slam_pose_frame_step"] = float(export_context["pose_frame_step"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _resolve_video_output_path(video_path, output_arg):
    output_arg = output_arg or "output_object_detection.mp4"
    output_arg = os.path.normpath(output_arg)
    suffix = os.path.splitext(output_arg)[1].lower()

    if suffix in VIDEO_EXTENSIONS:
        parent_dir = os.path.dirname(output_arg)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return output_arg

    os.makedirs(output_arg, exist_ok=True)
    input_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
    return os.path.join(output_arg, f"{input_stem}_object_detection.mp4")

def get_four_views_combined(equ_manager, fov=90, h=720, w=1080):
    """
    获取前后左右四个视角的透视图并水平拼接。
    """
    # 定义四个方向的角度：前(0), 右(90), 后(180), 左(270)
    angles = [0, 90, 180, 270]
    views = []
    
    for theta in angles:
        # 获取单一视角的透视图
        view = equ_manager.GetPerspective(FOV=fov, THETA=theta, PHI=0, height=h, width=w)
        
        # 在图片上标注方向，方便调试
        label = {0: "FRONT", 90: "RIGHT", 180: "BACK", 270: "LEFT"}[theta]
        cv2.putText(view, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        views.append(view)

    # 水平拼接所有视角 (Horizontal Concatenate)
    combined = cv2.hconcat(views)
    return combined

def run_video_multi_view(video_path, video_save_path='output_object_detection.mp4', track_save_path=None,
                         map_path=None, frame_traj_path=None, slam_frame_skip=None):
    # 1. 加载 YOLO 模型 (确保已安装 ultralytics: pip install ultralytics)
    # 建议使用 yolov8n.pt 以获得最快速度
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return

    # 获取视频原始 FPS 用于同步播放速度
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: original_fps = 30.0
    frame_delay = 1.0 / original_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设定输出单张切片的大小
    slice_h, slice_w = 480, 640 
    # 拼接后总宽度为 640*4 = 2560，总高度为 480

    video_save_path = _resolve_video_output_path(video_path, video_save_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    combined_w, combined_h = slice_w * 4, slice_h
    out = cv2.VideoWriter(video_save_path, fourcc, original_fps, (combined_w, combined_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频文件: {video_save_path}")

    # 视角参数
    angles = [0, 90, 180, 270] # 前、右、后、左
    labels = {0: "FRONT", 90: "RIGHT", 180: "BACK", 270: "LEFT"}

    # 创建可调整大小的窗口，防止 2560 宽度超出屏幕
    cv2.namedWindow('360 Surround Detection (F-R-B-L)', cv2.WINDOW_NORMAL)
    stable_tracker = StablePersonTracker(
        frame_width=combined_w,
        hold_frames=10,
        reid_iou_thresh=0.25,
        smooth_alpha=0.65,
        max_reid_frames=int(original_fps * 10),  # 最多约10秒内尝试重关联
        appearance_thresh=0.4  # （更低更容易按外观接回）
    )
    frame_idx = 0
    exported_track_items = []
    export_context = _build_export_context(map_path, frame_traj_path, original_fps, slam_frame_skip)
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video", unit="frame")

    while cap.isOpened():
        start_time = time.time() # 记录处理开始时刻

        ret, frame = cap.read()
        if not ret:
            break

        # 2. 投影转换 (使用类变量缓存加速)
        equ = E2P.Equirectangular(frame)
        
        views = []
        for theta in angles:
            # 获取单张透视图
            view = equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=slice_h, width=slice_w)
            
            # 在子图左上角标注方位
            cv2.putText(view, labels[theta], (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            views.append(view)

        # 3. 横向拼接成一排 (1x4 布局)
        combined_display = np.hstack(views)

        # 4. 在拼接后的长图上运行 tracking
        # classes=[0, 1] 仅检测人(0)和自行车(1)
        # persist=True: 同一目标跨帧保持 track id
        results = model.track(
            combined_display,
            imgsz=2560,
            classes=[0, 1],
            conf=0.5,
            iou=0.6,
            tracker='bytetrack.yaml',
            persist=True,
            verbose=False
        )

        # merge + 稳定 ID + anti-flicker 绘制
        annotated_frame = combined_display.copy()
        if results and len(results) > 0:
            persons = _extract_persons_with_merge(results[0], combined_display)
            visible_items = stable_tracker.update(persons, frame_idx)
            annotated_frame = draw_stable_persons(annotated_frame, visible_items)
            if track_save_path:
                for item in visible_items:
                    if item.get("ghost", False):
                        continue
                    exported_track_items.append(
                        _build_track_export_item(
                            item,
                            frame_idx,
                            combined_w,
                            combined_h,
                            export_context=export_context,
                        )
                    )
        frame_idx += 1
        progress.update(1)

        out.write(annotated_frame)

        # 5. FPS 同步与显示逻辑
        elapsed = time.time() - start_time
        # 计算需要等待的毫秒数
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        
        # 计算当前实际运行的 FPS
        actual_fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"Video FPS: {original_fps:.1f} | Real-time FPS: {actual_fps:.1f}", 
                    (20, annotated_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 6. 显示结果
        cv2.imshow('360 Degree Surround View (F-R-B-L)', annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    progress.close()
    cv2.destroyAllWindows()
    if track_save_path:
        _save_track_export(
            track_save_path,
            video_path=video_path,
            total_frames=total_frames,
            fps=original_fps,
            frame_width=combined_w,
            frame_height=combined_h,
            items=exported_track_items,
            export_context=export_context,
        )
        print(f"轨迹已保存至: {track_save_path}")
    print(f"视频已保存至: {video_save_path}")


def run_cyclist_distance_detection(video_path):
    model = YOLO('yolov8n.pt') 
    cap = cv2.VideoCapture(video_path)

    # 相机参数设置
    CAMERA_HEIGHT = 1.2  # 相机离地高度 (单位: 米)
    SLICE_W, SLICE_H = 640, 480
    FOV = 90
    
    # 预计算焦距 (用于测距)
    f = 0.5 * SLICE_W / np.tan(0.5 * FOV / 180.0 * np.pi)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        equ = E2P.Equirectangular(frame)
        angles = [0, 90, 180, 270] # 前、右、后、左
        views = []

        for theta in angles:
            view = equ.GetPerspective(FOV=FOV, THETA=theta, PHI=0, height=SLICE_H, width=SLICE_W)
            
            # 只检测人(0)和自行车(1)
            results = model.predict(view, classes=[0, 1], conf=0.3, verbose=False)
            
            for box in results[0].boxes:
                # 获取底部中心坐标 (x, y_bottom)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                y_bottom = y2
                
                # 计算距离逻辑 (单目测距原理)
                # 1. 计算物体底部相对于图像中心的像素偏移
                v_offset = y_bottom - (SLICE_H / 2)
                # 2. 计算俯角 (Angle of Depression)
                alpha = np.arctan2(v_offset, f)
                # 3. 物理距离 D = H / tan(alpha)
                # 注意：如果 alpha <= 0 说明物体在水平线以上，通常是远方或误检
                if alpha > 0:
                    distance = CAMERA_HEIGHT / np.tan(alpha)
                    
                    # 在画面上画框并标注距离
                    label = f"{model.names[int(box.cls[0])]}: {distance:.2f}m"
                    cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(view, label, (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            views.append(view)

        # 拼接并显示
        combined = cv2.hconcat(views)
        cv2.imshow('Cyclist Detection & Distance', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def run_yolo_on_pano(video_path):
    # 1. 加载模型 (推荐使用 Nano 版本以保证实时性)
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(video_path)
    
    # 设定子图尺寸
    slice_h, slice_w = 480, 640 
    angles = [0, 90, 180, 270]
    labels = {0: "FRONT", 90: "RIGHT", 180: "BACK", 270: "LEFT"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 实例化投影转换类
        equ = E2P.Equirectangular(frame)
        processed_views = []

        for theta in angles:
            # A. 获取透视图
            view = equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=slice_h, width=slice_w)
            
            # B. 运行 YOLO 检测
            # classes=[0, 1] 代表只检测人(0)和自行车(1)
            results = model.predict(view, classes=[0, 1], conf=0.3, verbose=False)
            
            # C. 在视角上绘制检测框
            annotated_view = results[0].plot() 
            
            # 标注方位
            cv2.putText(annotated_view, labels[theta], (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            processed_views.append(annotated_view)

        # D. 水平拼接四张已画框的图
        combined_display = cv2.hconcat(processed_views)

        # 显示结果
        cv2.imshow('Pano Object Detection (360 Degree)', combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="data/arrow360/VIDEO_0665/video.mp4",
        help="Input equirectangular video path.",
    )
    parser.add_argument(
        "--output",
        default="output_object_detection.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--track-output",
        default="data/arrow360/VIDEO_0665/object_tracks.json",
        help="Optional exported track JSON path. Pass empty string to disable.",
    )
    parser.add_argument(
        "--map-path",
        default="data/arrow360/VIDEO_0665/my_saved_map.msg",
        help="Map file used for world point export.",
    )
    parser.add_argument(
        "--frame-traj-path",
        default="data/arrow360/VIDEO_0665/frame_trajectory.txt",
        help="Frame trajectory file used for world point export.",
    )
    parser.add_argument(
        "--slam-frame-skip",
        type=int,
        default=2,
        help="Frame skip used by SLAM trajectory export.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_video_multi_view(
        args.video,
        video_save_path=args.output,
        track_save_path=args.track_output or None,
        map_path=args.map_path,
        frame_traj_path=args.frame_traj_path,
        slam_frame_skip=args.slam_frame_skip,
    )

    # run_yolo_on_pano(video_path)

    # run_cyclist_distance_detection(video_path)

# python tools/equirec2perspec_video.py --output path/to/output.mp4
