import ffmpeg
import os

def cut_video(input_path, output_path, start_time, end_time):
    """
    start_time / end_time 格式: '00:01:30'
    """
    print(f"[INFO] Cutting video: {input_path}")
    print(f"       Time range: {start_time} -> {end_time}")
    print(f"       Output: {output_path}")

    (
        ffmpeg
        .input(input_path, ss=start_time, to=end_time)
        .output(output_path, codec='copy')  # 不重新编码，快
        .run(overwrite_output=True)
    )

    print("[INFO] Cutting finished.")


def extract_frames(input_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Extracting frames from: {input_path}")
    print(f"       FPS: {fps}")
    print(f"       Output folder: {output_dir}")

    (
        ffmpeg
        .input(input_path)
        .filter('fps', fps)
        .output(f"{output_dir}/frame_%04d.jpg", start_number=0)
        .run(overwrite_output=True)
    )

    print("[INFO] Frame extraction finished.")


def main():
    data_dir = "data/ARROW360"
    data_name = 'VIDEO_0664'
    input_video = f"{data_dir}/{data_name}.mp4"

    output_video = f"{data_dir}/{data_name}/clip.mp4"
    output_frames_dir = os.path.join(data_dir, data_name, "images")

    # --- Fix: create directories ---
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    start_time = "00:07:38"
    end_time   = "00:07:42"
    fps = 2

    cut_video(
        input_path=input_video,
        output_path=output_video,
        start_time=start_time,
        end_time=end_time
    )

    extract_frames(
        input_path=output_video,
        output_dir=output_frames_dir,
        fps=fps
    )


if __name__ == "__main__":
    main()