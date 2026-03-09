import json, subprocess, os, csv, sys


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