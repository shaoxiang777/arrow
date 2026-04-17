import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path


import cv2
import json
import matplotlib
import numpy as np
import time
import yaml
from scipy.ndimage import convolve
from scipy.sparse import csr_array, vstack
from scipy.sparse.linalg import lsmr
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
import tools.Equirec2Perspec as E2P

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}
DEPTH_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

_MERGE_CACHE = None
_MERGE_DEPTH_VIDEOS = None
_MERGE_OUTPUT_SIZE = None


class _TorchvisionCompatFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "torchvision==" not in msg or "incompatible with torch==" not in msg


ULTRALYTICS_LOGGER.addFilter(_TorchvisionCompatFilter())


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
            "feat": feat.copy() if feat is not None else None,
            "label": "person",
            "cls_id": 0,
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
            label = person.get("label", "person")
            cls_id = int(person.get("cls_id", 0))

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
                self.tracks[stable_id]["label"] = label
                self.tracks[stable_id]["cls_id"] = cls_id
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
                "ghost": False,
                "confidence": float(self.tracks[stable_id].get("last_conf", 0.0)),
                "label": self.tracks[stable_id].get("label", "person"),
                "cls_id": int(self.tracks[stable_id].get("cls_id", 0)),
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
                    "ghost": True,
                    "confidence": float(track.get("last_conf", 0.0)),
                    "label": track.get("label", "person"),
                    "cls_id": int(track.get("cls_id", 0)),
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
            "feat": _extract_hsv_feature(image, [x1, y1, x2, y2]),
            "label": "person",
            "cls_id": 0,
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
            "feat": _extract_hsv_feature(image, det["box"]),
            "label": "person",
            "cls_id": 0,
        })

    return persons


def draw_stable_persons(image, visible_items):
    for item in visible_items:
        sid = item["stable_id"]
        x1, y1, x2, y2 = item["box"].astype(int).tolist()
        color = _color_from_track_id(sid)
        thickness = 3
        label = f"P{sid}"
        distance_m = item.get("distance_m")
        if distance_m is not None:
            label = f"{label} {distance_m:.2f}m"
        else:
            depth_median = item.get("depth_median")
            if depth_median is not None:
                label = f"{label} {depth_median:.2f}m"
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


def _estimate_pose_frame_step(frame_poses, total_video_frames, override_step):
    if override_step is not None and override_step > 0:
        return float(override_step)
    if not frame_poses:
        return 1.0

    pose_count = len(frame_poses)
    if pose_count <= 0:
        return 1.0
    if total_video_frames is None or int(total_video_frames) <= 0:
        return 1.0

    total_video_frames = int(total_video_frames)
    if total_video_frames == pose_count:
        return 1.0

    return max(1.0, float(total_video_frames) / float(pose_count))


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


def _build_export_context(map_path, frame_traj_path, total_video_frames, slam_frame_skip):
    if not map_path or not frame_traj_path:
        return None
    if not os.path.isfile(map_path):
        print(f"[WARN] map file not found, world-point export disabled: {map_path}")
        return None
    if not os.path.isfile(frame_traj_path):
        print(f"[WARN] frame trajectory file not found, world-point export disabled: {frame_traj_path}")
        return None

    keyframes = _load_map_keyframes(map_path)
    frame_poses = _load_frame_trajectory(frame_traj_path)
    if not keyframes or not frame_poses:
        print("[WARN] map or frame trajectory is empty, world-point export disabled.")
        return None
    plane_point, plane_normal = _estimate_global_track_plane(keyframes)
    pose_frame_step = _estimate_pose_frame_step(frame_poses, total_video_frames, slam_frame_skip)
    return {
        "frame_poses": frame_poses,
        "plane_point": plane_point,
        "plane_normal": plane_normal,
        "pose_frame_step": pose_frame_step,
        "track_radius": 6.0,
    }


def _panel_geometry_from_bbox(box, frame_width, frame_height):
    x1, y1, x2, y2 = [float(v) for v in box]
    panel_w = frame_width / 4.0
    panel_h = float(frame_height)
    if panel_w <= 1.0 or panel_h <= 1.0:
        return None

    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    view_idx = int(np.clip(np.floor(u / panel_w), 0, 3))
    u_local = float(np.clip(u - view_idx * panel_w, 0.0, panel_w - 1.0))
    v_local = float(np.clip(v, 0.0, panel_h - 1.0))
    return {
        "panel_w": panel_w,
        "panel_h": panel_h,
        "view_idx": view_idx,
        "u_local": u_local,
        "v_local": v_local,
    }


def _depth_world_point_from_combined_depth(item, frame_idx, frame_width, frame_height, depth_frame, export_context):
    if export_context is None or depth_frame is None:
        return None

    pose = _interpolate_pose(
        export_context["frame_poses"],
        frame_idx,
        export_context["pose_frame_step"],
    )
    if pose is None:
        return None

    x1, y1, x2, y2 = item["box"].astype(float).tolist()
    depth_h, depth_w = depth_frame.shape[:2]
    if depth_h <= 0 or depth_w <= 0:
        return None

    # Sample the middle-lower bbox area, then use the median of the nearest 10% depths.
    rx1 = int(np.clip(round(x1 + 0.15 * (x2 - x1)), 0, depth_w - 1))
    ry1 = int(np.clip(round(y1 + 0.45 * (y2 - y1)), 0, depth_h - 1))
    rx2 = int(np.clip(round(x2 - 0.15 * (x2 - x1)), rx1 + 1, depth_w))
    ry2 = int(np.clip(round(y2 - 0.05 * (y2 - y1)), ry1 + 1, depth_h))
    depth_patch = depth_frame[ry1:ry2, rx1:rx2]
    valid = np.isfinite(depth_patch) & (depth_patch > 0)
    if not np.any(valid):
        return None

    valid_depths = np.sort(depth_patch[valid].reshape(-1))
    nearest_count = max(1, int(np.ceil(0.10 * valid_depths.size)))
    nearest_depths = valid_depths[:nearest_count]
    depth_median = float(np.median(nearest_depths))
    panel = _panel_geometry_from_bbox([x1, y1, x2, y2], frame_width, frame_height)
    if panel is None:
        return None

    panel_w = panel["panel_w"]
    panel_h = panel["panel_h"]
    u_local = panel["u_local"]
    v_local = panel["v_local"]
    view_idx = panel["view_idx"]

    f = 0.5 * panel_w / np.tan(np.deg2rad(90.0) * 0.5)
    cx = 0.5 * panel_w
    cy = 0.5 * panel_h

    point_panel = np.array([
        (u_local - cx) / max(f, 1e-6) * depth_median,
        (v_local - cy) / max(f, 1e-6) * depth_median,
        depth_median,
    ], dtype=np.float64)

    yaw_centers_deg = [0.0, 90.0, 180.0, 270.0]
    yaw = np.deg2rad(yaw_centers_deg[view_idx])
    rot_y = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ], dtype=np.float64)
    point_cam = rot_y @ point_panel
    distance_m = float(np.linalg.norm(point_cam))
    bearing_yaw_deg = float(np.degrees(np.arctan2(point_cam[0], point_cam[2])))
    bearing_pitch_deg = float(np.degrees(np.arctan2(point_cam[1], np.linalg.norm(point_cam[[0, 2]]))))

    origin = np.asarray(pose["pos_w"], dtype=np.float64)
    world_point = origin + pose["rot_wc"] @ point_cam

    return {
        "view_idx": view_idx,
        "local_center_xy": [u_local, v_local],
        "point_cam": point_cam.tolist(),
        "distance_m": distance_m,
        "depth_median": depth_median,
        "bearing_yaw_deg": bearing_yaw_deg,
        "bearing_pitch_deg": bearing_pitch_deg,
        "world_point": world_point.tolist(),
        "world_point_method": "median_depth_backprojection_from_perspective",
    }


def _attach_depth_metrics_to_visible_items(visible_items, frame_idx, frame_width, frame_height, depth_frame, export_context):
    if depth_frame is None:
        return visible_items

    enriched_items = []
    for item in visible_items:
        enriched = dict(item)
        x1, y1, x2, y2 = item["box"].astype(float).tolist()
        depth_h, depth_w = depth_frame.shape[:2]
        if depth_h > 0 and depth_w > 0:
            rx1 = int(np.clip(round(x1 + 0.15 * (x2 - x1)), 0, depth_w - 1))
            ry1 = int(np.clip(round(y1 + 0.45 * (y2 - y1)), 0, depth_h - 1))
            rx2 = int(np.clip(round(x2 - 0.15 * (x2 - x1)), rx1 + 1, depth_w))
            ry2 = int(np.clip(round(y2 - 0.05 * (y2 - y1)), ry1 + 1, depth_h))
            depth_patch = depth_frame[ry1:ry2, rx1:rx2]
            valid = np.isfinite(depth_patch) & (depth_patch > 0)
            if np.any(valid):
                valid_depths = np.sort(depth_patch[valid].reshape(-1))
                nearest_count = max(1, int(np.ceil(0.10 * valid_depths.size)))
                nearest_depths = valid_depths[:nearest_count]
                enriched["depth_median"] = float(np.median(nearest_depths))
                enriched["distance_m"] = enriched["depth_median"]
        extra = _depth_world_point_from_combined_depth(
            item,
            frame_idx,
            frame_width,
            frame_height,
            depth_frame,
            export_context,
        )
        if extra is not None:
            for key in ("point_cam", "distance_m", "depth_median", "world_point", "world_point_method", "view_idx", "local_center_xy"):
                if key in extra:
                    enriched[key] = extra[key]
        enriched_items.append(enriched)
    return enriched_items


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


def _build_track_export_item(item, frame_idx, frame_width, frame_height, export_context=None, depth_frame=None):
    x1, y1, x2, y2 = item["box"].astype(float).tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    payload = {
        "frame_idx": int(frame_idx),
        "stable_id": int(item["stable_id"]),
        "ghost": bool(item.get("ghost", False)),
        "label": item.get("label", "person"),
        "cls_id": int(item.get("cls_id", 0)),
        "confidence": float(item.get("confidence", 0.0)),
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_xy": [cx, cy],
        "center_x_norm": float(cx / max(frame_width, 1)),
    }
    extra = _depth_world_point_from_combined_depth(
        item,
        frame_idx,
        frame_width,
        frame_height,
        depth_frame,
        export_context,
    )
    if extra is None:
        extra = _world_point_from_plane(item, frame_idx, frame_width, frame_height, export_context)
    if extra is not None:
        payload.update(extra)
    return payload


def _save_track_export(path, video_path, total_frames, fps, frame_width, frame_height, items, export_context=None):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    tracks = {}
    for item in items:
        sid = str(item["stable_id"])
        entry = {
            "frame_idx": item["frame_idx"],
            "ghost": item["ghost"],
            "label": item.get("label", "person"),
            "cls_id": int(item.get("cls_id", 0)),
            "confidence": float(item.get("confidence", 0.0)),
            "bbox_xyxy": item["bbox_xyxy"],
            "center_xy": item["center_xy"],
            "center_x_norm": item["center_x_norm"],
        }
        for key in (
            "view_idx",
            "local_center_xy",
            "ray_world",
            "point_cam",
            "depth_median",
            "distance_m",
            "bearing_yaw_deg",
            "bearing_pitch_deg",
            "world_point",
            "world_point_method",
        ):
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


def _resolve_video_output_path(video_path, output_arg, metric=False):
    output_arg = output_arg or "output_object_detection.mp4"
    output_arg = os.path.normpath(output_arg)
    stem, suffix = os.path.splitext(output_arg)
    name_suffix = "_metric" if metric else ""

    if suffix in VIDEO_EXTENSIONS:
        parent_dir = os.path.dirname(output_arg)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if metric and not stem.endswith("_metric"):
            return f"{stem}_metric{suffix}"
        return output_arg

    os.makedirs(output_arg, exist_ok=True)
    input_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
    return os.path.join(output_arg, f"{input_stem}_object_detection{name_suffix}.mp4")


def _default_path_from_video(video_path, filename):
    video_path = os.path.abspath(video_path)
    return os.path.join(os.path.dirname(video_path), filename)


def _resolve_aux_output_path(video_path, output_arg, suffix_name, metric=False):
    output_arg = output_arg or "output_object_detection.mp4"
    output_arg = os.path.normpath(output_arg)
    stem, suffix = os.path.splitext(output_arg)
    metric_suffix = "_metric" if metric else ""
    input_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"

    if suffix in VIDEO_EXTENSIONS:
        parent_dir = os.path.dirname(output_arg)
        base_dir = parent_dir if parent_dir else "."
    else:
        base_dir = output_arg
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{input_stem}_{suffix_name}{metric_suffix}.mp4")


def _resolve_split_view_output_paths(video_path, output_arg):
    output_arg = output_arg or "output_object_detection.mp4"
    output_arg = os.path.normpath(output_arg)
    _, suffix = os.path.splitext(output_arg)
    input_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"

    if suffix in VIDEO_EXTENSIONS:
        parent_dir = os.path.dirname(output_arg)
        base_dir = parent_dir if parent_dir else "."
    else:
        base_dir = output_arg
    os.makedirs(base_dir, exist_ok=True)

    return [
        os.path.join(base_dir, f"{input_stem}_view{view_idx}.mp4")
        for view_idx in range(1, 5)
    ]


def _save_frames_to_video(frames, output_path, fps):
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频文件: {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _play_video_loop(video_path, window_name, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_delay_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    try:
        while True:
            try:
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break
            if visible < 1:
                break
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(window_name, frame)
            cv2.waitKey(frame_delay_ms)
    finally:
        cap.release()


def _load_video_frames_bgr(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"视频中没有可读取的帧: {video_path}")

    return frames, fps


def _uv_map(height, width, dtype=np.float32):
    u = np.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=dtype)
    v = np.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=-1)


def _uv_to_pixel(uv, size_hw):
    size_hw = np.asarray(size_hw, dtype=np.float32)
    return uv * np.array([size_hw[1], size_hw[0]], dtype=np.float32) - 0.5


def _intrinsics_from_fov(fov_x_deg, fov_y_deg):
    fov_x = np.deg2rad(fov_x_deg)
    fov_y = np.deg2rad(fov_y_deg)
    fx = 1.0 / (2.0 * np.tan(fov_x * 0.5))
    fy = 1.0 / (2.0 * np.tan(fov_y * 0.5))
    return np.array([[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def _extrinsics_look_at(eye, look_at, up):
    eye = np.asarray(eye, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z = look_at - eye
    x = np.cross(-up, z)
    y = np.cross(z, x)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    rotation = np.stack([x, y, z], axis=-2)
    translation = -rotation @ eye[..., None]
    extrinsics = np.zeros((look_at.shape[0], 4, 4), dtype=np.float32)
    extrinsics[:, :3, :3] = rotation
    extrinsics[:, :3, 3:] = translation
    extrinsics[:, 3, 3] = 1.0
    return extrinsics


def _unproject_cv(uv, depth, intrinsics, extrinsics):
    points = np.concatenate(
        [uv, np.ones((*uv.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    ) * depth[..., None]
    points_h = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    intrinsics_h = np.eye(4, dtype=np.float32)
    intrinsics_h[:3, :3] = intrinsics
    transform = intrinsics_h @ extrinsics
    points_world = points_h @ np.linalg.inv(transform).swapaxes(-1, -2)
    return points_world[..., :3]


def _project_cv(points, intrinsics, extrinsics):
    points_h = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    intrinsics_h = np.eye(4, dtype=np.float32)
    intrinsics_h[:3, :3] = intrinsics
    transform = intrinsics_h @ extrinsics
    projected = points_h @ transform.swapaxes(-1, -2)
    uv = projected[..., :2] / np.clip(projected[..., 2:3], 1e-6, None)
    depth = projected[..., 2]
    return uv, depth


def _spherical_uv_to_directions(uv):
    theta = (1.0 - uv[..., 0]) * (2.0 * np.pi)
    phi = uv[..., 1] * np.pi
    return np.stack(
        [
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
            np.sin(phi) * np.cos(theta),
        ],
        axis=-1,
    )


def _directions_to_spherical_uv(directions):
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    theta = np.mod(np.arctan2(directions[..., 0], directions[..., 2]), 2.0 * np.pi)
    u = 1.0 - theta / (2.0 * np.pi)
    v = np.arccos(np.clip(directions[..., 1], -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=-1)


def _get_four_view_cameras(angles_deg):
    look_at = []
    for theta_deg in angles_deg:
        theta = np.deg2rad(theta_deg)
        look_at.append([np.sin(theta), 0.0, np.cos(theta)])
    look_at = np.asarray(look_at, dtype=np.float32)
    eye = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    extrinsics = _extrinsics_look_at(eye, look_at, up)
    intrinsics = [_intrinsics_from_fov(90.0, 90.0) for _ in angles_deg]
    return extrinsics, intrinsics


def _split_four_view_videos(frames, angles, labels, slice_h, slice_w):
    combined_frames = []
    split_videos = [[] for _ in angles]
    for frame in tqdm(frames, desc="Building four-view frames", unit="frame"):
        equ = E2P.Equirectangular(frame)
        display_views = []
        for view_idx, theta in enumerate(angles):
            view = equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=slice_h, width=slice_w)
            split_videos[view_idx].append(view)
            display_view = view.copy()
            cv2.putText(display_view, labels[theta], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display_views.append(display_view)
        combined_frames.append(np.hstack(display_views))
    split_videos = [np.stack(video, axis=0) for video in split_videos]
    return split_videos, np.stack(combined_frames, axis=0)


def _export_split_view_videos(video_path, output_arg, angles, slice_h, slice_w):
    output_paths = _resolve_split_view_output_paths(video_path, output_arg)
    if all(os.path.exists(path) for path in output_paths):
        print(f"[INFO] Reusing existing split-view videos: {', '.join(output_paths)}")
        return output_paths

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []

    try:
        for output_path in output_paths:
            writer = cv2.VideoWriter(output_path, fourcc, fps, (slice_w, slice_h))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建输出视频文件: {output_path}")
            writers.append(writer)

        progress = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Exporting split four-view videos",
            unit="frame",
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            equ = E2P.Equirectangular(frame)
            for writer, theta in zip(writers, angles):
                view = equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=slice_h, width=slice_w)
                writer.write(view)
            progress.update(1)
        progress.close()
    finally:
        cap.release()
        for writer in writers:
            writer.release()

    print(f"[INFO] Saved split-view videos: {', '.join(output_paths)}")
    return output_paths


def _poisson_equation(width, height, wrap_x=False, wrap_y=False):
    grid_index = np.arange(height * width).reshape(height, width)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode="wrap" if wrap_x else "edge")
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode="wrap" if wrap_y else "edge")
    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(height * width, axis=0).reshape(-1)
    indices = np.stack(
        [
            grid_index[1:-1, 1:-1],
            grid_index[:-2, 1:-1],
            grid_index[2:, 1:-1],
            grid_index[1:-1, :-2],
            grid_index[1:-1, 2:],
        ],
        axis=-1,
    ).reshape(-1)
    indptr = np.arange(0, height * width * 5 + 1, 5)
    return csr_array((data, indices, indptr), shape=(height * width, height * width))


def _grad_equation(width, height, wrap_x=False, wrap_y=False):
    grid_index = np.arange(width * height).reshape(height, width)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode="wrap")
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode="wrap")
    data = np.concatenate(
        [
            np.concatenate(
                [
                    np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
                    -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            ).reshape(-1),
            np.concatenate(
                [
                    np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),
                    -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            ).reshape(-1),
        ]
    )
    indices = np.concatenate(
        [
            np.concatenate([grid_index[:, :-1].reshape(-1, 1), grid_index[:, 1:].reshape(-1, 1)], axis=1).reshape(-1),
            np.concatenate([grid_index[:-1, :].reshape(-1, 1), grid_index[1:, :].reshape(-1, 1)], axis=1).reshape(-1),
        ]
    )
    indptr = np.arange(
        0,
        grid_index.shape[0] * (grid_index.shape[1] - 1) * 2
        + (grid_index.shape[0] - 1) * grid_index.shape[1] * 2
        + 1,
        2,
    )
    return csr_array(
        (data, indices, indptr),
        shape=(
            grid_index.shape[0] * (grid_index.shape[1] - 1)
            + (grid_index.shape[0] - 1) * grid_index.shape[1],
            height * width,
        ),
    )


def _build_merge_cache(width, height, distance_map_shapes, extrinsics, intrinsics):
    uv = _uv_map(height, width)
    spherical_directions = _spherical_uv_to_directions(uv)
    view_caches = []
    for view_idx, distance_map_shape in enumerate(distance_map_shapes):
        projected_uv, projected_depth = _project_cv(spherical_directions, intrinsics[view_idx], extrinsics[view_idx])
        projection_valid_mask = (
            (projected_depth > 0)
            & (projected_uv > 0).all(axis=-1)
            & (projected_uv < 1).all(axis=-1)
        )
        projected_pixels = _uv_to_pixel(np.clip(projected_uv, 0, 1), distance_map_shape).astype(np.float32)
        view_caches.append(
            {
                "projection_valid_mask": projection_valid_mask,
                "projected_pixels": projected_pixels,
            }
        )
    return {
        "width": width,
        "height": height,
        "view_caches": view_caches,
        "grad_equation": _grad_equation(width, height, wrap_x=True, wrap_y=False),
        "poisson_equation": _poisson_equation(width, height, wrap_x=True, wrap_y=False),
    }


def _merge_panorama_depth_cached(distance_maps, pred_masks, cache):
    width = cache["width"]
    height = cache["height"]
    panorama_log_distance_grad_maps = []
    panorama_grad_masks = []
    panorama_log_distance_laplacian_maps = []
    panorama_laplacian_masks = []
    panorama_pred_masks = []

    for view_idx, view_cache in enumerate(cache["view_caches"]):
        projection_valid_mask = view_cache["projection_valid_mask"]
        projected_pixels = view_cache["projected_pixels"]
        log_splitted_distance = np.log(np.clip(distance_maps[view_idx], 1e-8, None))
        panorama_log_distance_map = np.where(
            projection_valid_mask,
            cv2.remap(
                log_splitted_distance,
                projected_pixels[..., 0],
                projected_pixels[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ),
            0,
        )
        panorama_pred_mask = projection_valid_mask & (
            cv2.remap(
                pred_masks[view_idx].astype(np.uint8),
                projected_pixels[..., 0],
                projected_pixels[..., 1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            ) > 0
        )

        padded = np.pad(panorama_log_distance_map, ((0, 0), (0, 1)), mode="wrap")
        grad_x = padded[:, :-1] - padded[:, 1:]
        grad_y = padded[:-1, :] - padded[1:, :]
        padded_mask = np.pad(panorama_pred_mask, ((0, 0), (0, 1)), mode="wrap")
        mask_x = padded_mask[:, :-1] & padded_mask[:, 1:]
        mask_y = padded_mask[:-1, :] & padded_mask[1:, :]
        panorama_log_distance_grad_maps.append((grad_x, grad_y))
        panorama_grad_masks.append((mask_x, mask_y))

        padded_lap = np.pad(panorama_log_distance_map, ((1, 1), (0, 0)), mode="edge")
        padded_lap = np.pad(padded_lap, ((0, 0), (1, 1)), mode="wrap")
        laplacian = convolve(
            padded_lap,
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        )[1:-1, 1:-1]
        padded_mask = np.pad(panorama_pred_mask, ((1, 1), (0, 0)), mode="edge")
        padded_mask = np.pad(padded_mask, ((0, 0), (1, 1)), mode="wrap")
        lap_mask = convolve(
            padded_mask.astype(np.uint8),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        )[1:-1, 1:-1] == 5
        panorama_log_distance_laplacian_maps.append(laplacian)
        panorama_laplacian_masks.append(lap_mask)
        panorama_pred_masks.append(panorama_pred_mask)

    grad_x_stack = np.stack([g[0] for g in panorama_log_distance_grad_maps], axis=0)
    grad_y_stack = np.stack([g[1] for g in panorama_log_distance_grad_maps], axis=0)
    mask_x_stack = np.stack([m[0] for m in panorama_grad_masks], axis=0)
    mask_y_stack = np.stack([m[1] for m in panorama_grad_masks], axis=0)
    grad_x = np.sum(grad_x_stack * mask_x_stack, axis=0) / np.sum(mask_x_stack, axis=0).clip(1e-3)
    grad_y = np.sum(grad_y_stack * mask_y_stack, axis=0) / np.sum(mask_y_stack, axis=0).clip(1e-3)

    laplacian_maps = np.stack(panorama_log_distance_laplacian_maps, axis=0)
    laplacian_masks = np.stack(panorama_laplacian_masks, axis=0)
    laplacian = np.sum(laplacian_maps * laplacian_masks, axis=0) / np.sum(laplacian_masks, axis=0).clip(1e-3)

    grad_x_mask = np.any(mask_x_stack, axis=0).reshape(-1)
    grad_y_mask = np.any(mask_y_stack, axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    laplacian_mask = np.any(laplacian_masks, axis=0).reshape(-1)

    system = vstack(
        [
            cache["grad_equation"][grad_mask],
            cache["poisson_equation"][laplacian_mask],
        ]
    )
    rhs = np.concatenate(
        [
            grad_x.reshape(-1)[grad_x_mask],
            grad_y.reshape(-1)[grad_y_mask],
            laplacian.reshape(-1)[laplacian_mask],
        ]
    )
    solution, *_ = lsmr(system, rhs, atol=1e-5, btol=1e-5, show=False)
    panorama_depth = np.exp(solution).reshape(height, width).astype(np.float32)
    panorama_mask = np.any(np.stack(panorama_pred_masks, axis=0), axis=0)
    return panorama_depth, panorama_mask


def _merge_frame_worker(frame_idx):
    distance_maps = [_MERGE_DEPTH_VIDEOS[view_idx][frame_idx] for view_idx in range(len(_MERGE_DEPTH_VIDEOS))]
    pred_masks = [distance_map > 0 for distance_map in distance_maps]
    panorama_depth, _ = _merge_panorama_depth_cached(distance_maps, pred_masks, _MERGE_CACHE)
    output_width, output_height = _MERGE_OUTPUT_SIZE
    if panorama_depth.shape[1] != output_width or panorama_depth.shape[0] != output_height:
        panorama_depth = cv2.resize(panorama_depth, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    return panorama_depth


def _interpolate_panorama_depth_frame(merged_depths_by_frame, frame_idx):
    keyframes = sorted(merged_depths_by_frame.keys())
    if not keyframes:
        raise ValueError("No fused panorama depths available for interpolation")

    if frame_idx <= keyframes[0]:
        return merged_depths_by_frame[keyframes[0]]
    if frame_idx >= keyframes[-1]:
        return merged_depths_by_frame[keyframes[-1]]

    for left_idx, right_idx in zip(keyframes[:-1], keyframes[1:]):
        if left_idx <= frame_idx <= right_idx:
            if frame_idx == left_idx:
                return merged_depths_by_frame[left_idx]
            if frame_idx == right_idx:
                return merged_depths_by_frame[right_idx]
            left_depth = merged_depths_by_frame[left_idx]
            right_depth = merged_depths_by_frame[right_idx]
            alpha = float(frame_idx - left_idx) / float(right_idx - left_idx)
            return ((1.0 - alpha) * left_depth + alpha * right_depth).astype(np.float32)

    return merged_depths_by_frame[keyframes[-1]]


def _merge_four_view_depths(splitted_depth_videos, extrinsics, intrinsics, panorama_width, panorama_height,
                            output_width=None, output_height=None, merge_workers=1, frame_skip=1):
    merge_cache = _build_merge_cache(
        panorama_width,
        panorama_height,
        [video.shape[1:] for video in splitted_depth_videos],
        extrinsics,
        intrinsics,
    )
    num_frames = splitted_depth_videos[0].shape[0]
    final_output_width = output_width if output_width is not None else panorama_width
    final_output_height = output_height if output_height is not None else panorama_height
    frame_skip = max(1, int(frame_skip))
    frame_indices = list(range(0, num_frames, frame_skip))
    if frame_indices[-1] != num_frames - 1:
        frame_indices.append(num_frames - 1)

    global _MERGE_CACHE, _MERGE_DEPTH_VIDEOS, _MERGE_OUTPUT_SIZE
    _MERGE_CACHE = merge_cache
    _MERGE_DEPTH_VIDEOS = splitted_depth_videos
    _MERGE_OUTPUT_SIZE = (final_output_width, final_output_height)
    try:
        if merge_workers <= 1:
            merged_depths_by_frame = {}
            for frame_idx in tqdm(frame_indices, desc="Merging panorama depths", unit="frame"):
                merged_depths_by_frame[frame_idx] = _merge_frame_worker(frame_idx)
        else:
            mp_context = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=merge_workers, mp_context=mp_context) as executor:
                futures = {
                    executor.submit(_merge_frame_worker, frame_idx): frame_idx
                    for frame_idx in frame_indices
                }
                merged_depths_by_frame = {}
                for future in tqdm(
                    as_completed(futures),
                    total=len(frame_indices),
                    desc=f"Merging panorama depths ({merge_workers} workers)",
                    unit="frame",
                ):
                    frame_idx = futures[future]
                    merged_depths_by_frame[frame_idx] = future.result()
        if frame_skip > 1:
            print(f"[INFO] Fused keyframes ready; skipped frames will be interpolated on demand (frame_skip={frame_skip})")
        return merged_depths_by_frame
    finally:
        _MERGE_CACHE = None
        _MERGE_DEPTH_VIDEOS = None
        _MERGE_OUTPUT_SIZE = None


def _build_panorama_to_view_projection_cache(panorama_height, panorama_width, extrinsics, intrinsics, view_height, view_width):
    uv = _uv_map(view_height, view_width)
    unit_depth = np.ones((view_height, view_width), dtype=np.float32)
    cache = []
    for view_idx in range(len(extrinsics)):
        directions = _unproject_cv(uv, unit_depth, intrinsics[view_idx], extrinsics[view_idx]).astype(np.float32)
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-6)
        spherical_uv = _directions_to_spherical_uv(directions)
        panorama_pixels = _uv_to_pixel(spherical_uv, (panorama_height, panorama_width)).astype(np.float32)
        cache.append(
            {
                "directions": directions,
                "panorama_pixels": panorama_pixels,
            }
        )
    return cache


def _project_panorama_depth_frame_to_views(panorama_depth, projection_cache, extrinsics, intrinsics):
    view_depth_frames = []
    for view_idx, cached in enumerate(projection_cache):
        sampled_distance = cv2.remap(
            panorama_depth,
            cached["panorama_pixels"][..., 0],
            cached["panorama_pixels"][..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        directions = cached["directions"]
        points = directions * sampled_distance[..., None]
        _, view_depth = _project_cv(points, intrinsics[view_idx], extrinsics[view_idx])
        view_depth = np.where(np.isfinite(view_depth) & (view_depth > 0), view_depth.astype(np.float32), 0.0)
        view_depth_frames.append(view_depth)
    return view_depth_frames


def _remap_panorama_to_perspective_views(panorama_frame, angles, view_height, view_width, fov=90):
    equ = E2P.Equirectangular(np.repeat(panorama_frame[..., None], 3, axis=2))
    remapped_views = []
    for theta in angles:
        view = equ.GetPerspective(FOV=fov, THETA=theta, PHI=0, height=view_height, width=view_width)
        remapped_views.append(view[..., 0].astype(np.float32))
    return remapped_views


def _load_depth_model(depth_encoder="vitl", metric=True):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    depth_repo_root = repo_root / "third_part" / "video_da2"
    if str(depth_repo_root) not in sys.path:
        sys.path.insert(0, str(depth_repo_root))

    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    checkpoint_name = "metric_video_depth_anything" if metric else "video_depth_anything"
    checkpoint_path = depth_repo_root / "checkpoints" / f"{checkpoint_name}_{depth_encoder}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到深度模型权重: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_fp32 = device == "cpu"
    model = VideoDepthAnything(**DEPTH_ENCODER_CONFIGS[depth_encoder], metric=metric)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    return model, device, infer_fp32


def _apply_depth_colormap(depth_norm, colormap_name="inferno"):
    cmap_name = (colormap_name or "inferno").strip()
    cv2_colormaps = {
        "autumn": cv2.COLORMAP_AUTUMN,
        "bone": cv2.COLORMAP_BONE,
        "jet": cv2.COLORMAP_JET,
        "winter": cv2.COLORMAP_WINTER,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "ocean": cv2.COLORMAP_OCEAN,
        "summer": cv2.COLORMAP_SUMMER,
        "spring": cv2.COLORMAP_SPRING,
        "cool": cv2.COLORMAP_COOL,
        "hsv": cv2.COLORMAP_HSV,
        "pink": cv2.COLORMAP_PINK,
        "hot": cv2.COLORMAP_HOT,
        "parula": cv2.COLORMAP_PARULA,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "cividis": cv2.COLORMAP_CIVIDIS,
        "turbo": cv2.COLORMAP_TURBO,
    }
    if cmap_name.lower() in cv2_colormaps:
        return cv2.applyColorMap(depth_norm, cv2_colormaps[cmap_name.lower()])

    colored = matplotlib.colormaps[cmap_name](depth_norm.astype(np.float32) / 255.0)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def _render_depth_visualizations(depths, metric=True, colormap_name="inferno"):
    valid = np.isfinite(depths)
    vis_min = None
    vis_max = None
    if metric and np.any(valid):
        vis_min = float(np.percentile(depths[valid], 2.0))
        vis_max = float(np.percentile(depths[valid], 98.0))
        if vis_max <= vis_min:
            vis_max = vis_min + 1e-6

    depth_frames = []
    for depth in depths:
        frame_valid = np.isfinite(depth)
        if np.any(frame_valid):
            d_min = vis_min if vis_min is not None else float(np.percentile(depth[frame_valid], 2.0))
            d_max = vis_max if vis_max is not None else float(np.percentile(depth[frame_valid], 98.0))
            if d_max <= d_min:
                d_max = d_min + 1e-6
            depth_norm = ((np.clip(depth, d_min, d_max) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        depth_color = _apply_depth_colormap(depth_norm, colormap_name=colormap_name)
        cv2.putText(depth_color, "DEPTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        depth_frames.append(depth_color)

    return depth_frames


def _render_single_depth_visualization(depth, metric=True, vis_min=None, vis_max=None, colormap_name="inferno"):
    frame_valid = np.isfinite(depth)
    if np.any(frame_valid):
        d_min = vis_min if vis_min is not None else float(np.percentile(depth[frame_valid], 2.0))
        d_max = vis_max if vis_max is not None else float(np.percentile(depth[frame_valid], 98.0))
        if d_max <= d_min:
            d_max = d_min + 1e-6
        depth_norm = ((np.clip(depth, d_min, d_max) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    depth_color = _apply_depth_colormap(depth_norm, colormap_name=colormap_name)
    cv2.putText(depth_color, "DEPTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return depth_color


def _robust_depth_scale(source_depth, reference_depth):
    valid = (
        np.isfinite(source_depth)
        & np.isfinite(reference_depth)
        & (source_depth > 1e-6)
        & (reference_depth > 1e-6)
    )
    if not np.any(valid):
        return 1.0
    ratios = (reference_depth[valid] / source_depth[valid]).astype(np.float32)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    low = np.percentile(ratios, 10.0)
    high = np.percentile(ratios, 90.0)
    ratios = ratios[(ratios >= low) & (ratios <= high)]
    if ratios.size == 0:
        return 1.0
    scale = float(np.median(ratios))
    return float(np.clip(scale, 0.1, 10.0))


def _blend_four_view_depth_frame(base_view_depths, shifted_view_depths, overlap=32):
    if not base_view_depths or not shifted_view_depths:
        raise ValueError("base_view_depths and shifted_view_depths must not be empty")

    blended_views = [depth.astype(np.float32).copy() for depth in base_view_depths]
    num_views = len(blended_views)
    panel_w = blended_views[0].shape[1]
    half_w = panel_w // 2
    panel_scales = [1.0] * num_views

    for seam_idx in range(num_views - 1):
        left_view = blended_views[seam_idx]
        right_view = blended_views[seam_idx + 1]
        seam_reference = shifted_view_depths[seam_idx].astype(np.float32)

        ref_left = seam_reference[:, :half_w]
        ref_right = seam_reference[:, half_w:]
        left_overlap = left_view[:, half_w:]
        right_overlap = right_view[:, :half_w]

        scale_left = _robust_depth_scale(left_overlap, ref_left)
        scale_right = _robust_depth_scale(right_overlap, ref_right)
        relative_scale = scale_left / max(scale_right, 1e-6)
        relative_scale = float(np.clip(relative_scale, 0.1, 10.0))
        panel_scales[seam_idx + 1] = panel_scales[seam_idx] * relative_scale

    for view_idx in range(num_views):
        blended_views[view_idx] *= panel_scales[view_idx]

    return np.concatenate(blended_views, axis=1)


def _infer_depth_frames(combined_frames, fps, depth_encoder="vitl", metric=True, input_size=518):
    model, device, infer_fp32 = _load_depth_model(depth_encoder=depth_encoder, metric=metric)
    frames_rgb = combined_frames[..., ::-1].copy()

    print()
    print("\nStart to infer video depth...")
    depths, _ = model.infer_video_depth(
        frames_rgb,
        fps,
        input_size=input_size,
        device=device,
        fp32=infer_fp32,
    )
    return depths, _render_depth_visualizations(depths, metric=metric)


def _ensure_video_frame_exports(video_path, raw_frames):
    video_path = Path(video_path).resolve()
    frame_dir = video_path.parent / f"{video_path.stem}_images"
    txt_path = video_path.parent / f"{video_path.stem}_images.txt"
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing_pngs = sorted(frame_dir.glob("*.png"))
    if len(existing_pngs) != len(raw_frames):
        for stale_png in existing_pngs:
            stale_png.unlink()
        for frame_idx, frame in enumerate(tqdm(raw_frames, desc="Exporting ERP frames for DAP", unit="frame"), start=1):
            frame_path = frame_dir / f"{frame_idx:05d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"无法写入 DAP 输入帧: {frame_path}")
        existing_pngs = sorted(frame_dir.glob("*.png"))

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in existing_pngs:
            f.write(f"{frame_path}\n")

    return frame_dir, txt_path


def _ensure_video_frame_exports_from_video(video_path):
    video_path = Path(video_path).resolve()
    frame_dir = video_path.parent / f"{video_path.stem}_images"
    txt_path = video_path.parent / f"{video_path.stem}_images.txt"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    existing_pngs = sorted(frame_dir.glob("*.png"))
    if len(existing_pngs) != total_frames:
        for stale_png in existing_pngs:
            stale_png.unlink()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频 {video_path}")
        frame_idx = 0
        progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Exporting ERP frames for DAP", unit="frame")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                frame_path = frame_dir / f"{frame_idx:05d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"无法写入 DAP 输入帧: {frame_path}")
                progress.update(1)
        finally:
            progress.close()
            cap.release()
        existing_pngs = sorted(frame_dir.glob("*.png"))

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in existing_pngs:
            f.write(f"{frame_path}\n")

    return frame_dir, txt_path, total_frames


def _resolve_aux_output_dir(video_path, output_arg, suffix_name):
    output_arg = output_arg or "output_object_detection.mp4"
    output_arg = os.path.normpath(output_arg)
    stem, suffix = os.path.splitext(output_arg)
    input_stem = os.path.splitext(os.path.basename(video_path))[0] or "video"

    if suffix in VIDEO_EXTENSIONS:
        parent_dir = os.path.dirname(output_arg)
        base_dir = parent_dir if parent_dir else "."
    else:
        base_dir = output_arg
    aux_dir = os.path.join(base_dir, f"{input_stem}_{suffix_name}")
    os.makedirs(aux_dir, exist_ok=True)
    return aux_dir


def _load_dap_panorama_depth_cache(video_path, output_arg, frame_skip=1, max_depth=None):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    dap_infer_script = repo_root / "third_part" / "DAP" / "test" / "infer.py"
    dap_config_path = repo_root / "third_part" / "DAP" / "config" / "infer.yaml"
    frame_skip = max(1, int(frame_skip))
    suffix_name = "dap_depth" if frame_skip == 1 else f"dap_depth_fs{frame_skip}"
    if max_depth is not None:
        if float(max_depth).is_integer():
            max_depth_tag = str(int(max_depth))
        else:
            max_depth_tag = str(max_depth).replace(".", "p")
        suffix_name = f"{suffix_name}_md{max_depth_tag}"
    dap_output_dir = Path(_resolve_aux_output_dir(video_path, output_arg, suffix_name))
    depth_npy_dir = dap_output_dir / "depth_npy"

    _, txt_path, total_frames = _ensure_video_frame_exports_from_video(video_path)
    frame_indices = list(range(0, total_frames, frame_skip))
    existing_npys = sorted(depth_npy_dir.glob("*.npy")) if depth_npy_dir.exists() else []
    expected_stems = [f"{frame_idx + 1:05d}" for frame_idx in frame_indices]
    existing_stems = [path.stem for path in existing_npys]
    cache_ready = len(existing_npys) == len(frame_indices) and existing_stems == expected_stems
    if cache_ready:
        print(
            f"[INFO] Reusing existing DAP depth cache: {depth_npy_dir} "
            f"({len(existing_npys)} files)"
        )
    else:
        if existing_npys:
            print(
                f"[INFO] DAP depth cache incomplete or mismatched, rerunning inference: "
                f"expected {len(frame_indices)} files, found {len(existing_npys)}"
            )
        cmd = [
            sys.executable,
            str(dap_infer_script),
            "--config",
            str(dap_config_path),
            "--txt",
            str(txt_path),
            "--output",
            str(dap_output_dir),
            "--frame_skip",
            str(frame_skip),
        ]
        if max_depth is not None:
            cmd.extend(["--max_depth", str(max_depth)])
        print()
        print("[INFO] Running DAP on ERP frames...")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        existing_npys = sorted(depth_npy_dir.glob("*.npy"))
        existing_stems = [path.stem for path in existing_npys]

    if len(existing_npys) != len(frame_indices) or existing_stems != expected_stems:
        raise RuntimeError(
            "DAP depth output mismatch: "
            f"expected_files={len(frame_indices)}, actual_files={len(existing_npys)}"
        )

    depth_keyframe_paths = {}
    vis_values = []
    for npy_path in tqdm(existing_npys, desc="Scanning DAP ERP depths", unit="frame"):
        try:
            frame_idx = int(npy_path.stem) - 1
        except ValueError:
            raise RuntimeError(f"Unexpected DAP depth filename: {npy_path.name}")
        depth_keyframe_paths[frame_idx] = npy_path
        depth = np.load(str(npy_path)).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            vis_values.append(depth[valid])

    vis_min = None
    vis_max = None
    if vis_values:
        all_valid = np.concatenate(vis_values, axis=0)
        vis_min = float(np.percentile(all_valid, 2.0))
        vis_max = float(np.percentile(all_valid, 98.0))
        if vis_max <= vis_min:
            vis_max = vis_min + 1e-6
    if frame_skip > 1:
        print()
        print(f"[INFO] DAP keyframes ready; skipped ERP frames are filled by temporal interpolation (frame_skip={frame_skip})")
    return {
        "total_frames": total_frames,
        "frame_skip": frame_skip,
        "keyframe_indices": sorted(depth_keyframe_paths.keys()),
        "keyframe_paths": depth_keyframe_paths,
        "array_cache": {},
        "vis_min": vis_min,
        "vis_max": vis_max,
    }


def _load_dap_visualization_settings():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    dap_config_path = repo_root / "third_part" / "DAP" / "config" / "infer.yaml"
    with dap_config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return {
        "colormap_name": config.get("inference", {}).get("colormap_type", "jet"),
    }


def _load_dap_keyframe_depth(depth_state, frame_idx):
    if frame_idx in depth_state["array_cache"]:
        return depth_state["array_cache"][frame_idx]
    depth = np.load(str(depth_state["keyframe_paths"][frame_idx])).astype(np.float32)
    depth_state["array_cache"][frame_idx] = depth
    if len(depth_state["array_cache"]) > 4:
        oldest_key = next(iter(depth_state["array_cache"]))
        if oldest_key != frame_idx:
            depth_state["array_cache"].pop(oldest_key, None)
    return depth


def _interpolate_dap_panorama_depth(depth_state, frame_idx):
    keyframes = depth_state["keyframe_indices"]
    if not keyframes:
        raise ValueError("No DAP keyframe depths available")
    if frame_idx <= keyframes[0]:
        return _load_dap_keyframe_depth(depth_state, keyframes[0])
    if frame_idx >= keyframes[-1]:
        return _load_dap_keyframe_depth(depth_state, keyframes[-1])

    for left_idx, right_idx in zip(keyframes[:-1], keyframes[1:]):
        if left_idx <= frame_idx <= right_idx:
            if frame_idx == left_idx:
                return _load_dap_keyframe_depth(depth_state, left_idx)
            if frame_idx == right_idx:
                return _load_dap_keyframe_depth(depth_state, right_idx)
            left_depth = _load_dap_keyframe_depth(depth_state, left_idx)
            right_depth = _load_dap_keyframe_depth(depth_state, right_idx)
            alpha = float(frame_idx - left_idx) / float(right_idx - left_idx)
            return ((1.0 - alpha) * left_depth + alpha * right_depth).astype(np.float32)

    return _load_dap_keyframe_depth(depth_state, keyframes[-1])


def _build_four_view_display_from_frame(frame, angles, labels, slice_h, slice_w):
    equ = E2P.Equirectangular(frame)
    display_views = []
    for theta in angles:
        view = equ.GetPerspective(FOV=90, THETA=theta, PHI=0, height=slice_h, width=slice_w)
        display_view = view.copy()
        cv2.putText(display_view, labels[theta], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        display_views.append(display_view)
    return np.hstack(display_views)


def _infer_depth_frames_dap(video_path, output_arg, angles, slice_h, slice_w, frame_skip=1, max_depth=None):
    dap_cache = _load_dap_panorama_depth_cache(
        video_path,
        output_arg,
        frame_skip=frame_skip,
        max_depth=max_depth,
    )
    dap_vis = _load_dap_visualization_settings()

    return {
        "total_frames": dap_cache["total_frames"],
        "dap_cache": dap_cache,
        "angles": angles,
        "slice_h": slice_h,
        "slice_w": slice_w,
        "metric": True,
        "vis_min": dap_cache["vis_min"],
        "vis_max": dap_cache["vis_max"],
        "colormap_name": dap_vis["colormap_name"],
    }


def _infer_depth_frames_four_view(raw_frames, fps, angles, labels, slice_h, slice_w,
                                  depth_encoder="vitl", metric=True, input_size=518,
                                  merge_max_width=640, merge_max_height=320, merge_workers=1,
                                  frame_skip=1):
    model, device, infer_fp32 = _load_depth_model(depth_encoder=depth_encoder, metric=metric)
    shifted_angles = [45, 135, 225, 315]
    shifted_labels = {45: "SEAM_45", 135: "SEAM_135", 225: "SEAM_225", 315: "SEAM_315"}
    split_videos, combined_frames = _split_four_view_videos(raw_frames, angles, labels, slice_h, slice_w)
    shifted_videos, shifted_combined_frames = _split_four_view_videos(raw_frames, shifted_angles, shifted_labels, slice_h, slice_w)

    print()
    print("\nStart to infer video depth...")
    splitted_depth_videos = []
    for view_idx, split_video in enumerate(split_videos):
        print(f"[INFO] Infer depth for view {view_idx + 1}/{len(split_videos)}")
        view_depths, _ = model.infer_video_depth(
            split_video[..., ::-1].copy(),
            fps,
            input_size=input_size,
            device=device,
            fp32=infer_fp32,
        )
        splitted_depth_videos.append(view_depths.astype(np.float32))

    shifted_depth_videos = []
    for view_idx, split_video in enumerate(shifted_videos):
        print(f"[INFO] Infer shifted seam depth for view {view_idx + 1}/{len(shifted_videos)}")
        view_depths, _ = model.infer_video_depth(
            split_video[..., ::-1].copy(),
            fps,
            input_size=input_size,
            device=device,
            fp32=infer_fp32,
        )
        shifted_depth_videos.append(view_depths.astype(np.float32))

    vis_min = None
    vis_max = None
    if metric:
        per_view_mins = []
        per_view_maxs = []
        for view_depths in splitted_depth_videos + shifted_depth_videos:
            valid = np.isfinite(view_depths)
            if np.any(valid):
                per_view_mins.append(float(np.percentile(view_depths[valid], 2.0)))
                per_view_maxs.append(float(np.percentile(view_depths[valid], 98.0)))
        if per_view_mins and per_view_maxs:
            vis_min = min(per_view_mins)
            vis_max = max(per_view_maxs)
            if vis_max <= vis_min:
                vis_max = vis_min + 1e-6

    return {
        "combined_frames": combined_frames,
        "shifted_combined_frames": shifted_combined_frames,
        "base_depth_videos": splitted_depth_videos,
        "shifted_depth_videos": shifted_depth_videos,
        "metric": metric,
        "vis_min": vis_min,
        "vis_max": vis_max,
        "seam_blend_width": 32,
    }

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
                         map_path=None, frame_traj_path=None, slam_frame_skip=None,
                         with_depth=True, depth_encoder="vitl", depth_metric=False, depth_input_size=518,
                         depth_model="video_da2", max_depth=None,
                         merge_max_width=640, merge_max_height=320, merge_workers=1, frame_skip=1):
    if track_save_path is None:
        track_save_path = _default_path_from_video(video_path, "object_tracks.json")
    if map_path is None:
        map_path = _default_path_from_video(video_path, "my_saved_map.msg")
    if frame_traj_path is None:
        frame_traj_path = _default_path_from_video(video_path, "frame_trajectory.txt")

    # 先导出前/右/后/左四个独立视角视频，便于后续单独查看。
    split_angles = [0, 90, 180, 270]
    _export_split_view_videos(video_path, video_save_path, split_angles, 480, 640)

    # 1. 加载 YOLO 模型 (确保已安装 ultralytics: pip install ultralytics)
    # 建议使用 yolov8n.pt 以获得最快速度
    model = YOLO('yolov8n.pt') 
    shifted_model = YOLO('yolov8n.pt') if with_depth and depth_model == "video_da2" else None
    
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

    video_save_path = _resolve_video_output_path(video_path, video_save_path, metric=depth_metric)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    combined_w, combined_h = slice_w * 4, slice_h
    output_h = combined_h * 2 if with_depth else combined_h
    out = cv2.VideoWriter(video_save_path, fourcc, original_fps, (combined_w, output_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频文件: {video_save_path}")

    # 视角参数
    angles = [0, 90, 180, 270] # 前、右、后、左
    labels = {0: "FRONT", 90: "RIGHT", 180: "BACK", 270: "LEFT"}
    window_name = '360 Degree Surround View (F-R-B-L)'

    # 创建可调整大小的窗口，防止 2560 宽度超出屏幕
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    stable_tracker = StablePersonTracker(
        frame_width=combined_w,
        hold_frames=10,
        reid_iou_thresh=0.25,
        smooth_alpha=0.65,
        max_reid_frames=int(original_fps * 10),  # 最多约10秒内尝试重关联
        appearance_thresh=0.4  # （更低更容易按外观接回）
    )
    shifted_stable_tracker = StablePersonTracker(
        frame_width=combined_w,
        hold_frames=10,
        reid_iou_thresh=0.25,
        smooth_alpha=0.65,
        max_reid_frames=int(original_fps * 10),
        appearance_thresh=0.4
    )
    frame_idx = 0
    exported_track_items = []
    export_context = _build_export_context(map_path, frame_traj_path, total_frames, slam_frame_skip)
    if track_save_path and export_context is None:
        print("[INFO] Saving 2D/depth track export without world coordinates.")
    progress = None
    base_debug_out = None
    shifted_debug_out = None

    if with_depth:
        if depth_model == "dap":
            depth_state = _infer_depth_frames_dap(
                video_path,
                video_save_path,
                angles,
                slice_h,
                slice_w,
                frame_skip=frame_skip,
                max_depth=max_depth,
            )
            total_frames = depth_state["total_frames"]
            frame_source = None
        else:
            cap.release()
            raw_frames, original_fps = _load_video_frames_bgr(video_path)
            frame_delay = 1.0 / original_fps
            total_frames = len(raw_frames)
            depth_state = _infer_depth_frames_four_view(
                raw_frames,
                original_fps,
                angles,
                labels,
                slice_h,
                slice_w,
                depth_encoder=depth_encoder,
                metric=depth_metric,
                input_size=depth_input_size,
                merge_max_width=merge_max_width,
                merge_max_height=merge_max_height,
                merge_workers=merge_workers,
                frame_skip=frame_skip,
            )
            frame_source = depth_state["combined_frames"]
        if depth_model == "video_da2":
            base_stitched_path = _resolve_aux_output_path(
                video_path,
                video_save_path,
                "stitched_object_detection",
                metric=depth_metric,
            )
            shifted_stitched_path = _resolve_aux_output_path(
                video_path,
                video_save_path,
                "shifted_stitched_object_detection",
                metric=depth_metric,
            )
            base_debug_out = cv2.VideoWriter(base_stitched_path, fourcc, original_fps, (combined_w, output_h))
            shifted_debug_out = cv2.VideoWriter(shifted_stitched_path, fourcc, original_fps, (combined_w, output_h))
            if not base_debug_out.isOpened() or not shifted_debug_out.isOpened():
                raise RuntimeError("无法创建额外拼接调试视频文件")
        depth_values = None
        depth_frames = None
        progress = tqdm(total=total_frames, desc="Processing video", unit="frame")
    else:
        depth_values = None
        depth_frames = None
        depth_state = None
        frame_source = None
        progress = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video", unit="frame")

    while True:
        start_time = time.time() # 记录处理开始时刻

        if with_depth:
            if depth_model == "dap":
                ret, frame = cap.read()
                if not ret:
                    break
                combined_display = _build_four_view_display_from_frame(frame, angles, labels, slice_h, slice_w)
                current_panorama_depth = _interpolate_dap_panorama_depth(depth_state["dap_cache"], frame_idx)
                current_base_view_depths = _remap_panorama_to_perspective_views(
                    current_panorama_depth,
                    depth_state["angles"],
                    depth_state["slice_h"],
                    depth_state["slice_w"],
                )
                current_base_depth_value = np.concatenate(current_base_view_depths, axis=1)
                current_depth_value = current_base_depth_value
                shifted_combined_display = None
                shifted_depth_value = None
            else:
                if frame_idx >= len(frame_source):
                    break
                combined_display = frame_source[frame_idx].copy()
                current_base_view_depths = [
                    depth_state["base_depth_videos"][view_idx][frame_idx]
                    for view_idx in range(len(depth_state["base_depth_videos"]))
                ]
                current_base_depth_value = np.concatenate(current_base_view_depths, axis=1)
                current_shifted_view_depths = [
                    depth_state["shifted_depth_videos"][view_idx][frame_idx]
                    for view_idx in range(len(depth_state["shifted_depth_videos"]))
                ]
                current_depth_value = _blend_four_view_depth_frame(
                    current_base_view_depths,
                    current_shifted_view_depths,
                    overlap=depth_state["seam_blend_width"],
                )
                shifted_combined_display = depth_state["shifted_combined_frames"][frame_idx].copy()
                shifted_depth_value = np.concatenate(
                    [
                        depth_state["shifted_depth_videos"][view_idx][frame_idx]
                        for view_idx in range(len(depth_state["shifted_depth_videos"]))
                    ],
                    axis=1,
                )
            current_depth_frame = _render_single_depth_visualization(
                current_depth_value,
                metric=depth_state["metric"],
                vis_min=depth_state["vis_min"],
                vis_max=depth_state["vis_max"],
                colormap_name=depth_state.get("colormap_name", "inferno"),
            )
            current_base_depth_frame = _render_single_depth_visualization(
                current_base_depth_value,
                metric=depth_state["metric"],
                vis_min=depth_state["vis_min"],
                vis_max=depth_state["vis_max"],
                colormap_name=depth_state.get("colormap_name", "inferno"),
            )
            shifted_depth_frame = None
            if depth_model == "video_da2":
                shifted_depth_frame = _render_single_depth_visualization(
                    shifted_depth_value,
                    metric=depth_state["metric"],
                    vis_min=depth_state["vis_min"],
                    vis_max=depth_state["vis_max"],
                    colormap_name=depth_state.get("colormap_name", "inferno"),
                )
        else:
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
        stitched_annotated_frame = combined_display.copy() if with_depth else None
        shifted_annotated_frame = shifted_combined_display.copy() if shifted_combined_display is not None else None
        if results and len(results) > 0:
            persons = _extract_persons_with_merge(results[0], combined_display)
            visible_items = stable_tracker.update(persons, frame_idx)
            if with_depth:
                visible_items = _attach_depth_metrics_to_visible_items(
                    visible_items,
                    frame_idx,
                    combined_w,
                    combined_h,
                    current_depth_value,
                    export_context,
                )
            annotated_frame = draw_stable_persons(annotated_frame, visible_items)
            if with_depth:
                stitched_visible_items = _attach_depth_metrics_to_visible_items(
                    [dict(item) for item in visible_items],
                    frame_idx,
                    combined_w,
                    combined_h,
                    current_base_depth_value,
                    export_context,
                )
                stitched_annotated_frame = draw_stable_persons(stitched_annotated_frame, stitched_visible_items)
            if with_depth and depth_model == "video_da2":
                shifted_results = shifted_model.track(
                    shifted_combined_display,
                    imgsz=2560,
                    classes=[0, 1],
                    conf=0.5,
                    iou=0.6,
                    tracker='bytetrack.yaml',
                    persist=True,
                    verbose=False
                )
                if shifted_results and len(shifted_results) > 0:
                    shifted_persons = _extract_persons_with_merge(shifted_results[0], shifted_combined_display)
                    shifted_visible_items = shifted_stable_tracker.update(shifted_persons, frame_idx)
                    shifted_visible_items = _attach_depth_metrics_to_visible_items(
                        shifted_visible_items,
                        frame_idx,
                        combined_w,
                        combined_h,
                        shifted_depth_value,
                        export_context,
                    )
                    shifted_annotated_frame = draw_stable_persons(shifted_annotated_frame, shifted_visible_items)
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
                            depth_frame=current_depth_value if with_depth else None,
                        )
                    )
        frame_idx += 1
        progress.update(1)

        output_frame = annotated_frame
        if with_depth:
            output_frame = np.vstack([annotated_frame, current_depth_frame])
            if depth_model == "video_da2":
                shifted_output_frame = np.vstack([
                    shifted_annotated_frame if shifted_annotated_frame is not None else shifted_combined_display,
                    shifted_depth_frame,
                ])
                stitched_output_frame = np.vstack([
                    stitched_annotated_frame if stitched_annotated_frame is not None else combined_display,
                    current_base_depth_frame,
                ])

        out.write(output_frame)
        if base_debug_out is not None and depth_model == "video_da2":
            base_debug_out.write(stitched_output_frame)
        if shifted_debug_out is not None and depth_model == "video_da2":
            shifted_debug_out.write(shifted_output_frame)

        # 5. FPS 同步与显示逻辑
        elapsed = time.time() - start_time
        # 计算需要等待的毫秒数
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        
        # 计算当前实际运行的 FPS
        actual_fps = 1.0 / (time.time() - start_time)
        cv2.putText(output_frame, f"Video FPS: {original_fps:.1f} | Real-time FPS: {actual_fps:.1f}", 
                    (20, output_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 6. 显示结果
        cv2.imshow(window_name, output_frame)

        # 按 'q' 退出
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    if cap.isOpened():
        cap.release()
    out.release()
    if base_debug_out is not None:
        base_debug_out.release()
        print(f"直接拼接视频已保存至: {base_stitched_path}")
    if shifted_debug_out is not None:
        shifted_debug_out.release()
        print(f"偏移拼接视频已保存至: {shifted_stitched_path}")
    if progress is not None:
        progress.close()
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
    _play_video_loop(video_save_path, window_name, original_fps)
    cv2.destroyAllWindows()


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
        default=None,
        help="Optional exported track JSON path. Defaults to sibling object_tracks.json beside --video. Pass empty string to disable.",
    )
    parser.add_argument(
        "--map-path",
        default=None,
        help="Map file used for world point export. Defaults to sibling my_saved_map.msg beside --video.",
    )
    parser.add_argument(
        "--frame-traj-path",
        default=None,
        help="Frame trajectory file used for world point export. Defaults to sibling frame_trajectory.txt beside --video.",
    )
    parser.add_argument(
        "--slam-frame-skip",
        type=int,
        default=None,
        help="Optional override for the video-frame to SLAM-pose step. By default it is inferred from total video frames and frame_trajectory.txt length.",
    )
    parser.add_argument(
        "--depth-encoder",
        choices=sorted(DEPTH_ENCODER_CONFIGS.keys()),
        default="vitl",
        help="Depth model encoder.",
    )
    parser.add_argument(
        "--depth-input-size",
        type=int,
        default=518,
        help="Depth model inference input size.",
    )
    parser.add_argument(
        "--depth-model",
        choices=["video_da2", "dap"],
        default="video_da2",
        help="Depth backend. 'video_da2' keeps the existing four-view pipeline; 'dap' infers depth on ERP frames first.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Override DAP max_depth (meters). Only meaningful when --depth-model dap.",
    )
    parser.add_argument(
        "--metric",
        action="store_true",
        help="Use metric depth checkpoint instead of the default relative depth.",
    )
    parser.add_argument(
        "--merge-max-width",
        type=int,
        default=640,
        help="Panorama width used during Poisson depth fusion before upsampling.",
    )
    parser.add_argument(
        "--merge-max-height",
        type=int,
        default=320,
        help="Panorama height used during Poisson depth fusion before upsampling.",
    )
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) // 2)),
        help="Number of worker processes for panorama depth fusion.",
    )
    parser.add_argument(
        "--frame-skip", "--frame_skip",
        type=int,
        default=1,
        help="Apply frame skipping only during Poisson fusion; skipped frames are filled by temporal interpolation.",
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
        with_depth=True,
        depth_encoder=args.depth_encoder,
        depth_metric=args.metric,
        depth_input_size=args.depth_input_size,
        depth_model=args.depth_model,
        max_depth=args.max_depth,
        merge_max_width=args.merge_max_width,
        merge_max_height=args.merge_max_height,
        merge_workers=args.merge_workers,
        frame_skip=args.frame_skip,
    )

    # run_yolo_on_pano(video_path)

    # run_cyclist_distance_detection(video_path)


# python tools/equirec2perspec_video.py \
#   --video ./data/arrow360/VIDEO_0665/video_30s.mp4 \
#   --output ./output/arrow360/VIDEO_0665 \
#   --metric \
#   --merge-workers 4 \
#   --frame-skip 5
