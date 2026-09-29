from utils import report_progress

import subprocess, os, json
from queue import Queue
import threading
from tqdm import tqdm
import logging


class Clipper:
    """Clipper class for clipping segments from a video path"""

    def __init__(self, output_dir: str, video_path: str, ffmpeg_path: str, vertical_format: bool, max_workers: int):
        self.ffmpeg_path = ffmpeg_path
        self.vertical_format = vertical_format
        self.output_dir = output_dir
        self.video_path = video_path
        self.max_workers = max_workers
        self.crop_width = 608
        self.crop_height = 1080
        self.x_offset = "(in_w - {0})/2".format(self.crop_width)
        self.y_offset = "(in_h - {0})/2".format(self.crop_height)
        self.clip_queue = Queue()
        self.total_clips_extracted = 0
        self.percentages = set(range(1, 101, 1))


    def clip_event(self, output_dir: str, event: dict, video_path: str):
        output_path = os.path.join(output_dir, event['desc'])

        if not self.vertical_format:
            subprocess.run([
            self.ffmpeg_path,
            "-ss", str(event['timestart']),
            "-i", video_path,
            "-to", str(event['timeend'] - event['timestart']),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-loglevel", "error",
            "-y",
            output_path
            ])
        
        else:
            cmd = [
                self.ffmpeg_path,
                "-ss", str(event['timestart']),
                "-i", video_path,
                "-to", str(event['timeend'] - event['timestart']),
                "-filter:v",
                f"crop={self.crop_width}:{self.crop_height}:{self.x_offset}:{self.y_offset},scale=1080:1920,setsar=1",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-loglevel", "error",
                "-y",
                output_path
            ]


            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                #print(f"✅ Successfully extracted vertical TikTok video: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg error: {e.stderr.decode()}")


    def _process_clips(self, meta_file: str):
        self.clip_progress = self.percentages.copy()
        with open(f"{self.output_dir}/{meta_file}", 'r') as f:
            events = json.load(f)
            clip_events = []
            for event in events:
                video_path = self.video_path#[int(event['desc'][-14])-1]
                out_dir = ''.join((self.output_dir, "/", event['type']))
                os.makedirs(out_dir, exist_ok=True)
                clip_events.append((out_dir, event, video_path))

        print(clip_events)

        progress_bar = tqdm(total=len(clip_events), desc="Extracting clips", unit="clip")
        self.total_clips = len(clip_events)

        for _ in range(self.max_workers):
            threading.Thread(target=self.clip_worker, args=(progress_bar,), daemon=True).start()

        for arg in clip_events:
            self.clip_queue.put(arg)

        self.clip_queue.join()

        for _ in range(self.max_workers):
            self.clip_queue.put(None)

        progress_bar.close()


    def clip_worker(self, progress_bar):
        while True:
            args = self.clip_queue.get()
            if args is None:
                break
            try:
                self.clip_event(*args)
            except Exception as e:
                logging.error(f"Clip extraction failed: {e}")
            finally:
                self.clip_queue.task_done()
                progress_bar.update(1)
                self.total_clips_extracted+=1
                report_progress(self.output_dir, self.total_clips_extracted, self.total_clips, self.clip_progress, "EXTRACTING CLIPS...")