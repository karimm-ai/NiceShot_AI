from event_types import Event
from utils import add_to_csv_, resource_path, add_to_json, report_progress

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import logging
import os, threading, time
import numpy
import logging


class YOLODetector:
    """Main Class for detecting events"""

    def __init__(self,
                 model_path: str,
                 events_config: dict,
                 conf_thresholds: dict,
                 video_path: str,
                 total_hours: float = 100,
                 output_dir: str = ".",
                 frame_idx_start: int = 0,
                 frames_to_skip: int = 8,
                 add_to_csv: bool = False,
                 event_confirm = None):
        
        if event_confirm is not None:
            self.event_confirm = event_confirm
            self.last_known_context = None

        self.events_config = events_config
        self.conf_thresholds = conf_thresholds
        self.output_dir = output_dir
        self.video_path = [video_path]
        self.total_hours = total_hours
        self.frame_idx_start = frame_idx_start
        self.frames_to_skip = frames_to_skip
        self.add_to_csv = add_to_csv
        self.events = []
        self.filename = f"{self.output_dir}/events_temp.json"
        
        self.model = YOLO(resource_path(model_path)).to("cuda")
        logging.info("Loaded Model Successfully!")

        self.ffmpeg_path = resource_path("src/niceshot_ai/ffmpeg.exe")
        print(f"FFMPEG PATH: {self.ffmpeg_path}")
            
        if self.add_to_csv:
            self.events_csv = []
            self.events_csv_lock = threading.Lock()

        self.percentages = set(range(1, 101, 1))


    def detect_events(self, progress_bar = None):
        self.vid_process_progress = self.percentages.copy()

        os.makedirs(self.output_dir, exist_ok=True)

        trackers = self._init_trackers()

        logging.info("Loaded Trackers Successfully!")

        for video_index, video_path in enumerate(self.video_path, start=1):
            self.csv_file = f"video{video_index}.csv"

            self._process_video(
                video_path,
                video_index,
                self.model,
                trackers,
                progress_bar)

            return self.csv_file


    def _update_progress(self, frame_idx: int, pbar, progress_bar = None):
        pbar.update(1)

        if not progress_bar:
            return

        total = max(self.TOTAL_FRAMES_TO_BE_ANALYZED, 1)
        progress_bar["value"] = min(100, (frame_idx / total) * 100)
        progress_bar.update()


    def _init_trackers(self) -> dict:
        trackers = {}
        for event, val in self.events_config.items():
            trackers[event] = val['tracker']

        return trackers


    def _process_video(self, video_path: str, video_index: int, model: YOLO, trackers: dict, progress_bar):
        logging.info(f"Processing video {video_path}")

        self.cap = cv2.VideoCapture(video_path)

        self._init_video_metadata(self.cap)

        clip_frames = {}
        temp_ids = {}

        for key, val in self.events_config.items():
            temp_ids[key] = set()
            if val.get('clip_eligible'):
                clip_frames[key] = []


        with tqdm(total=self.TOTAL_FRAMES_TO_BE_ANALYZED, desc="Processing video") as pbar:
            frame_idx = 0
            while self.cap.isOpened() and frame_idx < self.TOTAL_FRAMES_TO_BE_ANALYZED:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame_idx+=1
                    continue

                if frame_idx >= self.TOTAL_FRAMES_TO_BE_ANALYZED:
                    break

                if self._should_process_frame(frame_idx):
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                    report_progress(self.output_dir, frame_idx, self.TOTAL_FRAMES_TO_BE_ANALYZED, self.vid_process_progress, "ANALYZING GAMEPLAY")

                    detections = self._collect_detections(model, frame)

                    if any(detections.values()) and hasattr(self, "event_confirm") and self.last_known_context is None:
                        for key, _ in detections.items():
                            context = self.events_config[key].get("context")

                            if context:
                                self.last_known_context = {}
                                x = 0
                                for key_, val_ in context.items():
                                    x+=1
                                    roi = self.event_confirm.crop_frame2(frame, val_[0], val_[1])
                                    roi = self.event_confirm.pre_process_frame(roi)

                                    data = self.event_confirm.read_text(roi)
                                    self.last_known_context[key_] = data

                    tracks = self._update_trackers(trackers, detections, frame)
                    self._handle_tracks(
                        tracks,
                        video_index,
                        temp_ids,
                        clip_frames,
                        timestamp
                    )

                self._update_progress(frame_idx, pbar, progress_bar)
                frame_idx += 1

        self._finalize_video_events(video_index, clip_frames)
        logging.info(f"Finalizing video {video_index}")
        self.cap.release()

        if self.add_to_csv:
            add_to_csv_(self.output_dir, self.csv_file, self.events_csv)
            self.events_csv.clear()


    def _init_video_metadata(self, cap: cv2.VideoCapture):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fps = cap.get(cv2.CAP_PROP_FPS)

        duration_hours = total_frames / self.fps / 3600
        max_hours = min(duration_hours, self.total_hours)

        self.TOTAL_FRAMES_TO_BE_ANALYZED = int(max_hours * 3600 * self.fps)

        self.DURATION_TO_BE_ANALYZED = max_hours*60

        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {duration_hours}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")
        print(f"Video Resolution {self.video_width}x{self.video_height}")

        logging.info(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {duration_hours}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")


    def _should_process_frame(self, frame_idx: int) -> bool:
        return (
            frame_idx >= self.frame_idx_start
            and frame_idx % self.frames_to_skip == 0
        )
    

    def _collect_detections(self, model: YOLO, frame: numpy.ndarray) -> dict:
        results = model(frame, verbose=False)[0]
        
        detections = {}        
        for key, val in self.events_config.items():
            detections[key] = []
        
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()

            if conf < self.conf_thresholds.get(cls, 1):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]

            for key, val in self.events_config.items():
                if cls != val.get("cls_label"):
                    continue

                if val.get("confirm_event") and hasattr(self, "event_confirm"):
                    if self.event_confirm.is_invalid_event(frame):
                        continue
                detections[key].append((bbox, conf, cls))

        return detections


    def _update_trackers(self, trackers: dict, detections: dict, frame: numpy.ndarray) -> dict:
        tracks = {}

        for key, tracker in trackers.items():
            tracks[key] = tracker.update_tracks(detections[key], frame=frame)

        return tracks


    def _handle_tracks(
        self,
        tracks: dict,
        video_index: int,
        temp_ids: dict,
        clip_frames: dict,
        timestamp: int
    ):
        processed_event_tracks = []

        for key, val in clip_frames.items():
            self._add_clipable_tracks(
                tracks.get(key, []),
                temp_ids[key],
                val,
                video_index,
                key,
                timestamp
            )
            processed_event_tracks.append(key)

        for key, val in tracks.items():
            if key not in processed_event_tracks:
                for track in tracks.get(key, []):
                    track_id = track["track_id"] if isinstance(track, dict) else track.track_id
                    if track_id in temp_ids[key]:
                        continue

                    temp_ids[key].add(track_id)

                    if self.add_to_csv:
                        timestamp_mod = time.strftime(
                            "%H:%M:%S",
                            time.gmtime(timestamp)
                        )

                        with self.events_csv_lock:
                            self.events_csv.append({
                                "Timestamp": timestamp_mod,
                                "Event": key
                            })


    def _add_clipable_tracks(
        self,
        tracks: dict,
        seen_ids: dict,
        frame_buffer: list,
        video_index: int,
        event_type: str,
        timestamp
    ):
        for track in tracks:
            if track.track_id not in seen_ids:
                seen_ids.add(track.track_id)
                if frame_buffer:
                    self.add_event(frame_buffer, video_index, event_type)
                    frame_buffer.clear()
            frame_buffer.append(timestamp)


    def _finalize_video_events(self, video_index: int, clip_frames: dict):
        for key, val in clip_frames.items():
            self.add_event(val, video_index, key)

        if self.events:
            add_to_json(self.filename, self.events)
            self.events.clear()


    def find_event_times(self, event_times: list, event_type: str) -> tuple | None:       
        seconds_before = self.events_config[event_type]['pre']
        seconds_after = self.events_config[event_type]['post']
        starting_time = min(event_times) - seconds_before
        ending_time = max(event_times) + seconds_after
        
        if starting_time <= 0:
            starting_time = 0
        
        if ending_time >= self.TOTAL_FRAMES_TO_BE_ANALYZED/self.fps:
            ending_time = self.TOTAL_FRAMES_TO_BE_ANALYZED/self.fps
        
        # Avoids large clips of irrelevant events (some cases)
        estimated_clip_time = seconds_before + 2 + seconds_after
        if ending_time - starting_time > estimated_clip_time:
            ending_time = starting_time + estimated_clip_time
            
        return starting_time, ending_time


    def add_event(self, event_times: list, video_num: int, event_type: str):
        if not event_times:
            return
        starting_time, ending_time = self.find_event_times(event_times, event_type)
        if hasattr(self, "event_confirm"):
            if self.last_known_context is not None:
                event = Event(event_type, starting_time, ending_time, video_num, **self.last_known_context)
                self.last_known_context = None
            
            else:
                event = Event(event_type, starting_time, ending_time, video_num)

        else:
            event = Event(event_type, starting_time, ending_time, video_num)

        self.events.append(event)
        
        if self.add_to_csv:
            with self.events_csv_lock:
                self.events_csv.append({"Timestamp": time.strftime("%H:%M:%S", time.gmtime(starting_time)),
                                    "Event": event_type})