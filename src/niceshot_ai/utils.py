import json, subprocess, os, csv, sys, shutil
from pathlib import Path


import cv2


def get_duration(clip_path: str) -> float:
    """Returns the duration of a video using OpenCV."""

    cap = cv2.VideoCapture(clip_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {clip_path}")

    return frame_count / fps


def add_to_csv_(output_dir: str, filename: str, events: list):
        output_filename = os.path.join(output_dir, filename)
        fieldnames = ["Timestamp", "Event"]
        
        with open(output_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if file.tell() == 0:
                writer.writeheader()
            
            for event in events:
                writer.writerow(event)


def resource_path(filename: str) -> str:
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    else:
        base_path = Path(__file__).resolve().parent.parent.parent

    return base_path / filename


def add_to_json(filename: str, events: list):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except:
            data = []
    else:
        data = []

    data.extend([e.to_dict() for e in events])

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def move_clips_to_folder(clips_paths: list, montage_length: int, output_dir: str, new_folder: str):
    print(f"Moving clips to {output_dir}/{new_folder}\n")

    root_dir = Path(output_dir)
    
    final_clips = []
    current_length = 0
    while current_length <= montage_length:
        if not len(clips_paths) > 0:
            break
        
        vid_path = clips_paths.pop(0)
        vid_path = list(root_dir.rglob(vid_path))[0]
        print(vid_path)

        if vid_path:
            final_clips.append(vid_path)
            current_length += get_duration(vid_path)

    for clip in final_clips:
        shutil.copy(clip, new_folder)


def get_data_path(filename):
    base_path = Path.home() / ".my_app"
    base_path.mkdir(exist_ok=True)
    return base_path / filename


def report_progress(output_dir, numerator, denominator, progress, msg):
    current_percent = numerator*100//denominator
    if current_percent in progress:
        with open(f"{output_dir}/progress.json", "w") as file:
            json.dump({"PROGRESS": current_percent, "MSG": msg}, file)
        progress.discard(current_percent)


def reencode_to_h264(input_file, output_dir):
        input_path = Path(input_file)
        output_file = (
            Path(output_dir)
            / f"{input_path.stem}_fixed.mp4"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-err_detect", "ignore_err",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-vsync", "1",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_file)
        ]

        try:
            subprocess.run(cmd, check=True)
            return str(output_file)

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed: {e}")
            return None