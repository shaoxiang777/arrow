import argparse
from pathlib import Path
import subprocess


def parse_time_to_seconds(value):
    text = str(value).strip()
    if not text:
        raise ValueError("time value is empty")

    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"invalid time '{text}', expected MM:SS or HH:MM:SS"
        )

    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            f"invalid time '{text}', expected numeric MM:SS or HH:MM:SS"
        ) from exc

    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0.0
    else:
        hours, minutes, seconds = numbers

    if minutes < 0 or seconds < 0 or hours < 0:
        raise ValueError("time must be non-negative")

    return hours * 3600.0 + minutes * 60.0 + seconds


def build_default_output_path(video_path, start_seconds, end_seconds):
    stem = video_path.stem
    suffix = video_path.suffix or ".mp4"
    start_tag = format_seconds_for_name(start_seconds)
    end_tag = format_seconds_for_name(end_seconds)
    return video_path.with_name(f"{stem}_{start_tag}_to_{end_tag}{suffix}")


def format_seconds_for_name(seconds):
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def probe_duration_seconds(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def crop_video(video_path, start_seconds, end_seconds, output_path):
    duration_seconds = probe_duration_seconds(video_path)
    if start_seconds >= duration_seconds:
        raise ValueError(
            f"start time {start_seconds:.2f}s exceeds video duration {duration_seconds:.2f}s"
        )

    if end_seconds > duration_seconds:
        end_seconds = duration_seconds

    if end_seconds <= start_seconds:
        raise ValueError("end time must be greater than start time")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-to",
        str(end_seconds),
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)

    return end_seconds - start_seconds


def main():
    parser = argparse.ArgumentParser(
        description="Crop a video into a specified time range."
    )
    parser.add_argument(
        "--video-path",
        default="data/arrow360/VIDEO_0665/video.mp4",
        help="input video path",
    )
    parser.add_argument(
        "--start-time",
        default="0:00",
        help="start time, format MM:SS or HH:MM:SS",
    )
    parser.add_argument(
        "--end-time",
        default="1:30",
        help="end time, format MM:SS or HH:MM:SS",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="optional output path; defaults to inputname_HH-MM-SS_to_HH-MM-SS.mp4",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video file does not exist: {video_path}")

    start_seconds = parse_time_to_seconds(args.start_time)
    end_seconds = parse_time_to_seconds(args.end_time)
    if end_seconds <= start_seconds:
        raise ValueError("end time must be greater than start time")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else build_default_output_path(video_path, start_seconds, end_seconds)
    )

    written_duration = crop_video(
        video_path=video_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        output_path=output_path,
    )

    print(f"input: {video_path}")
    print(f"output: {output_path}")
    print(f"start_time: {args.start_time} ({start_seconds:.2f}s)")
    print(f"end_time: {args.end_time} ({end_seconds:.2f}s)")
    print(f"written_duration: {written_duration:.2f}s")


if __name__ == "__main__":
    main()


# python3 tools/crop_video.py \
#   --video-path data/arrow360/VIDEO_0665/video.mp4 \
#   --start-time 0:00 \
#   --end-time 1:30 \
#   --output-path data/arrow360/VIDEO_0665/video_clip.mp4
