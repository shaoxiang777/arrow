import cv2
import numpy as np
import time
from ultralytics import YOLO
import tools.Equirec2Perspec as E2P

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

def run_video_multi_view(video_path, video_save_path='output_object_detection.mp4'):
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

    # 设定输出单张切片的大小
    slice_h, slice_w = 480, 640 
    # 拼接后总宽度为 640*4 = 2560，总高度为 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    combined_w, combined_h = slice_w * 4, slice_h
    out = cv2.VideoWriter(video_save_path, fourcc, original_fps, (combined_w, combined_h))

    # 视角参数
    angles = [0, 90, 180, 270] # 前、右、后、左
    labels = {0: "FRONT", 90: "RIGHT", 180: "BACK", 270: "LEFT"}

    # 创建可调整大小的窗口，防止 2560 宽度超出屏幕
    cv2.namedWindow('360 Surround Detection (F-R-B-L)', cv2.WINDOW_NORMAL)

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

        # 4. 在拼接后的长图上运行一次 YOLO 推理
        # classes=[0, 1] 仅检测人(0)和自行车(1)
        # results = model.predict(combined_display, classes=[0, 1], conf=0.3, verbose=False)
        results = model.predict(combined_display, imgsz=2560, classes=[0, 1], conf=0.25)   # 我这里尝试  更大的image
        
        # 绘制检测结果（画框）
        annotated_frame = results[0].plot()

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
    cv2.destroyAllWindows()
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

if __name__ == '__main__':

    video_path = 'data/arrow360/VIDEO_0665/video.mp4'

    run_video_multi_view(video_path)

    # run_yolo_on_pano(video_path)

    # run_cyclist_distance_detection(video_path)

# python tools/equirec2perspec_video.py