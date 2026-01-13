import msgpack
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def visualize_stella_map(msg_path):
    # 1. 加载数据
    with open(msg_path, "rb") as f:
        u = msgpack.Unpacker(f)
        msg = u.unpack()

    keyframes = msg["keyframes"]
    landmarks = msg["landmarks"]

    # 2. 解析关键帧并计算相机中心位置
    kf_list = []
    for kf in keyframes.values():
        # 获取相机到世界的变换矩阵
        rot_cw = R.from_quat(kf["rot_cw"]).as_matrix()
        trans_cw = np.array(kf["trans_cw"]).reshape(3, 1)
        
        # 计算世界坐标系下的相机位置: Ow = -R_cw^T * t_cw
        rot_wc = rot_cw.T
        trans_wc = -rot_wc @ trans_cw
        
        kf_list.append({
            "ts": kf["ts"],
            "pos": trans_wc.flatten(),
            "rot": rot_wc
        })

    # 按时间戳排序
    kf_list.sort(key=lambda x: x["ts"])
    kf_positions = np.array([kf["pos"] for kf in kf_list])

    # 3. 创建相机轨迹线 (LineSet)
    lines = [[i, i + 1] for i in range(len(kf_positions) - 1)]
    colors = [[0, 1, 0] for _ in range(len(lines))] # 绿色轨迹线
    trajectory_lines = o3d.geometry.LineSet()
    trajectory_lines.points = o3d.utility.Vector3dVector(kf_positions)
    trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
    trajectory_lines.colors = o3d.utility.Vector3dVector(colors)

    # 4. 创建路标点云 (PointCloud)
    landmark_points = np.array([lm["pos_w"] for lm in landmarks.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmark_points)
    # 给点云上色（红色）
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(landmark_points), 1)))

    # 5. 可视化
    print(f"Total Keyframes: {len(kf_list)}")
    print(f"Total Landmarks: {len(landmark_points)}")
    
    # 添加一个坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 启动可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Stella VSLAM Map Viewer", width=1280, height=720)
    
    vis.add_geometry(pcd)
    vis.add_geometry(trajectory_lines)
    vis.add_geometry(coord_frame)
    
    # 可选：为每个关键帧画一个小坐标轴（如果帧数太多会卡顿，可以每隔5帧画一个）
    for i in range(0, len(kf_list), 5):
        kf_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # 构造 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = kf_list[i]["rot"]
        T[:3, 3] = kf_list[i]["pos"]
        kf_axes.transform(T)
        vis.add_geometry(kf_axes)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 请确保路径正确
    map_msg_path = "data/aist_store_2/my_saved_map.msg"
    visualize_stella_map(map_msg_path)