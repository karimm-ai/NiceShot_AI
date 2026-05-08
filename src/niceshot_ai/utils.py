import json, subprocess, os, csv, sys, shutil
from pathlib import Path


def get_duration(clip_path: str) -> float:
    """Returns the duration of a video using ffprobe"""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        clip_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    return float(info['format']['duration'])


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
    return filename


def add_to_json(filename: str, events: list):
    #filename = get_data_path(filename)
    #filename.parent.mkdir(parents=True, exist_ok=True)
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
    final_clips = []
    current_length = 0
    while current_length <= montage_length:
        if not len(clips_paths) > 0:
            break
        
        vid_path = clips_paths.pop(0)#[0]
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
        progress.remove(current_percent)