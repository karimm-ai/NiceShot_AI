from twitch_handler import TwitchHandler
from video_clipper import Clipper
from kill_events_process import KillEventsProcessor
from event_types import Event
from montage import Montage
from utils import add_to_csv_, resource_path, add_to_json, move_clips_to_folder, report_progress, reencode_to_h264
from event_confirm import EventConfirm
from report import ReportMaker

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import logging
from queue import Queue
import os, threading, time
import json
import numpy
import logging



class EventDetector:
    """Main Class for detecting events"""

    def __init__(self,
                 game_name: str,
                 video_path: str,
                 total_hours: float = 100,
                 save_clips: bool = True,
                 output_dir: str = ".",
                 max_workers: int = 2,
                 frame_idx_start: int = 0,
                 frames_to_skip: int = 8,
                 add_to_csv: bool = False,
                 create_montage: bool = True,
                 montage_length_sec: int = 20,
                 max_videos: int = 1,
                 vertical_format: bool = False,
                 advanced_detection: bool = True,
                 session_analysis = False
                 ):
        
        if game_name.lower()  == "call of duty: black ops 6":
            from events_config import cod_bo6_config
            self.events_config = cod_bo6_config
            self.model_path = resource_path("../game_models/yolov8n-cod_bo6.pt")
        
            if session_analysis:
                from charts_config import cod_bo6_chart_config
                self.report_config = cod_bo6_chart_config

        elif game_name.lower() == "call of duty: black ops 7":
            from events_config import cod_bo7_config
            self.events_config = cod_bo7_config
            self.model_path = resource_path("../game_models/yolov8n-cod_bo7.pt")

            if session_analysis:
                from charts_config import cod_bo7_chart_config
                self.report_config = cod_bo7_chart_config


        self.output_dir = output_dir
        self.video_path = [video_path]
        self.max_workers = max_workers
        self.total_hours = total_hours
        self.save_clips = save_clips
        self.frame_idx_start = frame_idx_start
        self.frames_to_skip = frames_to_skip
        self.add_to_csv = add_to_csv
        self.create_montage = create_montage
        self.events = []
        self.filename = f"{self.output_dir}/events_temp.json"
        self.montage_length_sec = montage_length_sec
        self.vertical_format = vertical_format

        self.ffmpeg_path = resource_path("ffmpeg.exe")
        print(f"FFMPEG PATH: {self.ffmpeg_path}")

        if advanced_detection:
            self.event_confirm = EventConfirm()

        if 'twitch' in self.video_path[0]:
            twitch_handler = TwitchHandler(self.video_path[0], max_videos, self.output_dir)
            vods = twitch_handler.get_all_videos(game_name)
            with open ('vods.txt', 'w') as file:
                for vod in vods:
                    file.write(f"{vod}\n")
            
            twitch_handler.download_channel_videos(vods)
            self.video_path = [f"{self.output_dir}/Downloads/{file}" for file in os.listdir(f"{self.output_dir}/Downloads")]
            
        if self.save_clips:
            self.clip_queue = Queue()
            self.clipper = Clipper(self.ffmpeg_path, self.vertical_format)
            self.total_clips_extracted = 0         
        
        if self.add_to_csv:
            self.events_csv = []
            self.events_csv_lock = threading.Lock()

        logging.basicConfig(
            filename="tracker.log",
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.percentages = set(range(1, 101, 1))

    def clip_worker(self, progress_bar):
        while True:
            args = self.clip_queue.get()
            if args is None:
                break
            try:
                self.clipper.clip_event(*args)
            except Exception as e:
                logging.error(f"Clip extraction failed: {e}")
            finally:
                self.clip_queue.task_done()
                progress_bar.update(1)
                self.total_clips_extracted+=1
                report_progress(self.output_dir, self.total_clips_extracted, self.total_clips, self.clip_progress, "EXTRACTING CLIPS...")


    def detect_events(self, progress_bar = None):
        self.vid_process_progress = self.percentages.copy()
        os.makedirs(self.output_dir, exist_ok=True)
        model = YOLO(self.model_path).to("cuda")
        logging.info("Loaded Model Successfully!")
        trackers = self._init_trackers()
        logging.info("Loaded Trackers Successfully!")

        for video_index, video_path in enumerate(self.video_path, start=1):
            self.csv_file = f"video{video_index}.csv"
            #reencoded_video_path = reencode_to_h264(video_path, self.output_dir)
            self._process_video(
                video_path,
                video_index,
                model,
                trackers,
                progress_bar
            )
            if hasattr(self, "report_config") and self.add_to_csv:
                bucket_len = int(self.DURATION_TO_BE_ANALYZED // 10)# // 1 * 10)    # minutes
                if bucket_len == 0:
                    bucket_len = 1
                print(bucket_len)
                report = ReportMaker(self.output_dir, f"{self.output_dir}/{self.csv_file}",
                                     self.events_config, self.report_config, bucket_len)
                for chart in self.report_config["charts"]:
                    func = getattr(report, chart["name"])
                    func(self.report_config["color_pallete"], chart["width"], chart["height"])
                #report.fig.show()
                report.save_report(video_index)
            #os.remove(reencoded_video_path)

        
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
                if not ret:
                    break

                if self._should_process_frame(frame_idx):
                    report_progress(self.output_dir, frame_idx, self.TOTAL_FRAMES_TO_BE_ANALYZED, self.vid_process_progress, "ANALYZING GAMEPLAY")
                    detections = self._collect_detections(model, frame)
                    tracks = self._update_trackers(trackers, detections, frame)
                    self._handle_tracks(
                        tracks,
                        frame_idx,
                        video_index,
                        temp_ids,
                        clip_frames
                    )

                self._update_progress(frame_idx, pbar, progress_bar)
                frame_idx += 1

        self._finalize_video_events(video_index, clip_frames)
        logging.info(f"Finalizing video {video_index}")
        self.cap.release()

        if "Kill" in self.events_config.keys():
            self.kills_proc = KillEventsProcessor(self.model_path, self.output_dir)
            self.kills_proc.concat_kill_streaks(video_index)

        if self.add_to_csv:
            add_to_csv_(self.output_dir, self.csv_file, self.events_csv)
            self.events_csv.clear()

        if self.save_clips:
            self._process_clips()
        
        if self.save_clips and (self.create_montage and self.montage_length_sec > 0):
            self._create_montage()


    def _init_video_metadata(self, cap: cv2.VideoCapture):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        duration_hours = total_frames / self.fps / 3600

        max_hours = min(duration_hours, self.total_hours)
        self.TOTAL_FRAMES_TO_BE_ANALYZED = int(max_hours * 3600 * self.fps)
        self.DURATION_TO_BE_ANALYZED = max_hours*60
        print(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {duration_hours}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")
        logging.info(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {duration_hours}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")


    def _should_process_frame(self, frame_idx: int) -> bool:
        return (
            frame_idx >= self.frame_idx_start
            and frame_idx % self.frames_to_skip == 0
        )
    

    def _collect_detections(self, model: YOLO, frame: numpy.ndarray) -> dict:
        logging.info("Collecting detections")
        results = model(frame, verbose=False)[0]
        
        detections = {}
        conf_thresholds = {}
        for key, val in self.events_config.items():
            detections[key] = []
            conf_thresholds[val['cls_label']] = val['conf_thres']
        
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()

            if conf < conf_thresholds.get(cls, 1):
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
        frame_idx: int,
        video_index: int,
        temp_ids: dict,
        clip_frames: dict,
    ):
        processed_event_tracks = []

        for key, val in clip_frames.items():
            self._handle_event_tracks(
                tracks.get(key, []),
                temp_ids[key],
                val,
                frame_idx,
                video_index,
                key
            )
            processed_event_tracks.append(key)

        for key, val in tracks.items():
            if key not in processed_event_tracks:
                for track in tracks.get(key, []):
                    if track.track_id not in temp_ids[key]:
                        temp_ids[key].add(track.track_id)
                        if self.add_to_csv:
                            timestamp = time.strftime("%H:%M:%S", time.gmtime(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                            with self.events_csv_lock:
                                self.events_csv.append({"Timestamp": timestamp, "Event": key})


    def _handle_event_tracks(
        self,
        tracks: dict,
        seen_ids: dict,
        frame_buffer: list,
        frame_idx: int,
        video_index: int,
        event_type: str
    ):
        for track in tracks:
            if track.track_id not in seen_ids:
                seen_ids.add(track.track_id)
                if frame_buffer:
                    self.finalize_event(frame_buffer, video_index, event_type)
                    frame_buffer.clear()
            frame_buffer.append(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)


    def _finalize_video_events(self, video_index: int, clip_frames: dict):
        for key, val in clip_frames.items():
            self.finalize_event(val, video_index, key)

        if self.events:
            add_to_json(self.filename, self.events)
            self.events.clear()


    def _process_clips(self):
         self.clip_progress = self.percentages.copy()
         with open(f"{self.output_dir}/events_temp_2.json", 'r') as f:
            events = json.load(f)
            clip_events = []
            for event in events:
                video_path = self.video_path[int(event['desc'][-14])-1]
                out_dir = ''.join((self.output_dir, "/", event['type']))
                os.makedirs(out_dir, exist_ok=True)
                clip_events.append((out_dir, event, video_path))

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
        

    def _create_montage(self):
        montage = Montage()

        if "Kill" in self.events_config:
            best_kill_clips = self.kills_proc.find_best_kills()
            new_folder = ''.join((self.output_dir, '/best_kill_clips'))
            os.makedirs(new_folder, exist_ok=True)
            move_clips_to_folder(best_kill_clips, self.montage_length_sec, self.output_dir, new_folder)

        for dir in os.listdir(self.output_dir):
            if os.path.isdir(os.path.join(self.output_dir, dir)) and dir not in ("Kill", "KillStreak"):
                clips = []
                for clip in os.listdir(os.path.join(self.output_dir, dir)):
                    if clip.endswith("mp4"):
                        clips.append(os.path.join(self.output_dir, dir, clip))
                new_folder = ''.join((self.output_dir, f"/{dir}_compilation_clips"))
                os.makedirs(new_folder, exist_ok=True)
                move_clips_to_folder(clips, self.montage_length_sec, self.output_dir, new_folder)
                montage.make_compilation(os.path.join(self.output_dir, f"{dir}_compilation_clips"),
                                         os.path.join(self.output_dir, f"{dir}_highlight_reel.mp4"))

                if not self.vertical_format:
                    montage.make_tiktok(os.path.join(self.output_dir, f"{dir}_highlight_reel.mp4"),
                                        os.path.join(self.output_dir, f"{dir}_highlight_reel_tiktok.mp4"))
        

    def find_event_frames(self, event_frames: list, event_type: str) -> tuple | None:       
        seconds_before = self.events_config[event_type]['pre']
        seconds_after = self.events_config[event_type]['post']
        starting_time = min(event_frames) - seconds_before
        ending_time = max(event_frames) + seconds_after
        
        if starting_time <= 0:
            starting_time = 0
        
        if ending_time >= self.TOTAL_FRAMES_TO_BE_ANALYZED/self.fps:
            ending_time = self.TOTAL_FRAMES_TO_BE_ANALYZED/self.fps
        
        # Avoid large clips of irrelevant events (some cases)
        estimated_clip_time = seconds_before + 2 + seconds_after
        if ending_time - starting_time > estimated_clip_time:
            ending_time = starting_time + estimated_clip_time
            
        return starting_time, ending_time


    def finalize_event(self, event_frames: list, video_num: int, event_type: str):
        if not event_frames:
            return
        starting_time, ending_time = self.find_event_frames(event_frames, event_type)
        event = Event(event_type, starting_time, ending_time, video_num)
        self.events.append(event)
        
        if self.add_to_csv:
            with self.events_csv_lock:
                self.events_csv.append({"Timestamp": time.strftime("%H:%M:%S", time.gmtime(starting_time)),
                                    "Event": event_type})