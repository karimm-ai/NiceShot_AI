from .twitch_handler import TwitchHandler
from .video_clipper import Clipper
from .kill_events_process import KillEventsProcessor
from .event_types import EventType, Event
from .montage import Montage
from .utils import add_to_csv_, resource_path, add_to_json
from .events_config import cod_bo6_config
from .event_confirm import EventConfirm

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import logging
from queue import Queue
import os, threading, time
import json
import easyocr

class EventDetector:
    """Main Class for detecting events"""

    def __init__(self, model_path,
                 ffmpeg_path,
                 video_path,
                 events_config=cod_bo6_config,
                 csv_file=None,
                 events_to_detect=["Kill", "Death", "Medal"],
                 total_hours=10,
                 save_clips=True,
                 output_dir=".",
                 max_workers=2, 
                 frame_idx_start= 0,
                 frames_to_skip=5,
                 add_to_csv=False,
                 create_montage=False,
                 montage_length_sec=20,
                 max_videos = 1,
                 vertical_format = False,
                 advanced_detection=False
                 ):

        self.output_dir = output_dir
        self.video_path = [video_path]
        self.csv_file = csv_file
        self.events_config = events_config
        self.events_to_detect = events_to_detect
        self.max_workers = max_workers
        self.total_hours = total_hours
        self.save_clips = save_clips
        self.frame_idx_start = frame_idx_start
        self.frames_to_skip = frames_to_skip
        self.add_to_csv = add_to_csv
        self.create_montage = create_montage
        self.events = []
        self.filename = 'events_temp.json'
        self.montage_length_sec = montage_length_sec
        self.vertical_format = vertical_format

        self.model_path = resource_path(model_path)
        self.ffmpeg_path = resource_path(ffmpeg_path)
        print(f"FFMPEG PATH: {self.ffmpeg_path}")

        if advanced_detection:
            self.event_confirm = EventConfirm()

        if 'twitch' in self.video_path[0]:
            twitch_handler = TwitchHandler(self.video_path[0], max_videos, self.output_dir)
            vods = twitch_handler.get_all_videos()
            with open ('vods.txt', 'w') as file:
                for vod in vods:
                    file.write(f"{vod}\n")
            
            twitch_handler.download_channel_videos(vods)
            self.video_path = [f"{self.output_dir}/Downloads/{file}" for file in os.listdir(f"{self.output_dir}/Downloads")]
            
        if self.save_clips:
            self.clip_queue = Queue()
            
            self.clipper = Clipper(self.ffmpeg_path, self.vertical_format)
            
            self.KILL_DIR = ''.join((self.output_dir, '/Kills'))
            self.DEATH_DIR = ''.join((self.output_dir, '/Deaths'))

            os.makedirs(self.KILL_DIR, exist_ok=True)
            os.makedirs(self.DEATH_DIR, exist_ok=True)
            self.kills_proc = KillEventsProcessor(self.model_path, self.output_dir)
        
        if self.add_to_csv:
            self.events_csv = []
            self.events_csv_lock = threading.Lock()
            if self.csv_file == None:
                self.csv_file = 'timestamps.csv'
   

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
   

    def detect_events(self, progress_bar=None):
        os.makedirs(self.output_dir, exist_ok=True)
        model = YOLO(self.model_path).to("cuda")
        trackers = self._init_trackers()

        for video_index, video_path in enumerate(self.video_path, start=1):
            print(f"Processing video {video_path}")
            self._process_video(
                video_path,
                video_index,
                model,
                trackers,
                progress_bar
            )

        
    def _update_progress(self, frame_idx, pbar, progress_bar=None):
        pbar.update(1)

        if not progress_bar:
            return

        total = max(self.TOTAL_FRAMES_TO_BE_ANALYZED, 1)
        progress_bar["value"] = min(100, (frame_idx / total) * 100)
        progress_bar.update()


    def _init_trackers(self):
        trackers = {}
        for event, val in self.events_config.items():
            trackers[event] = val['tracker']

        return trackers


    def _process_video(self, video_path, video_index, model, trackers, progress_bar):
        cap = cv2.VideoCapture(video_path)
        self._init_video_metadata(cap)

        kill_frames, death_frames = [], []
        temp_ids = {}

        for key, _ in self.events_config.items():
            temp_ids[key] = set()

        with tqdm(total=self.TOTAL_FRAMES_TO_BE_ANALYZED, desc="Processing video") as pbar:
            frame_idx = 0
            while cap.isOpened() and frame_idx < self.TOTAL_FRAMES_TO_BE_ANALYZED:
                ret, frame = cap.read()
                if not ret:
                    break

                if self._should_process_frame(frame_idx):
                    detections = self._collect_detections(model, frame)
                    tracks = self._update_trackers(trackers, detections, frame)
                    self._handle_tracks(
                        tracks,
                        frame_idx,
                        video_index,
                        temp_ids,
                        kill_frames,
                        death_frames
                    )

                self._update_progress(frame_idx, pbar, progress_bar)
                frame_idx += 1
        #print(inv_frames)
        #return
        self._finalize_video_events(video_index, kill_frames, death_frames)
        cap.release()

        if "Kill" in self.events_config.keys():
            self.kills_proc.concat_kill_streaks_new(video_index)

        if self.add_to_csv:
            add_to_csv_(self.output_dir, self.csv_file, self.events_csv)
            self.events_csv.clear()

        if self.save_clips:
            self._process_clips()
        
        if self.save_clips and (self.create_montage or self.montage_length_sec > 0):
            self._create_montage()


    def _init_video_metadata(self, cap):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        duration_hours = total_frames / self.fps / 3600

        max_hours = min(duration_hours, self.total_hours)
        self.TOTAL_FRAMES_TO_BE_ANALYZED = int(max_hours * 3600 * self.fps)
        print(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {duration_hours}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")


    def _should_process_frame(self, frame_idx):
        return (
            frame_idx >= self.frame_idx_start
            and frame_idx % self.frames_to_skip == 0
        )
    

    def _collect_detections(self, model, frame):
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

            if cls == 0 and "Kill" in self.events_config.keys():
                if hasattr(self, "event_confirm"):
                    if not self.event_confirm.is_invalid_event(frame):
                        detections["Kill"].append((bbox, conf, cls))

            elif cls == 1 and "Medal" in self.events_config.keys():
                if hasattr(self, "event_confirm"):
                    if not self.event_confirm.is_invalid_event(frame):
                        detections["Medal"].append((bbox, conf, cls))
            
            elif cls == 2 and "Death" in self.events_config.keys():
                detections["Death"].append((bbox, conf, cls))

        return detections


    def _update_trackers(self, trackers, detections, frame):
        tracks = {}

        for key, tracker in trackers.items():
            tracks[key] = tracker.update_tracks(detections[key], frame=frame)

        return tracks


    def _handle_tracks(
        self,
        tracks,
        frame_idx,
        video_index,
        temp_ids,
        kill_frames,
        death_frames,
    ):
        self._handle_event_tracks(
            tracks.get("Kill", []),
            temp_ids["Kill"],
            kill_frames,
            frame_idx,
            video_index,
            EventType.KILL
        )

        self._handle_event_tracks(
            tracks.get("Death", []),
            temp_ids["Death"],
            death_frames,
            frame_idx,
            video_index,
            EventType.DEATH
        )

        for track in tracks.get("Medal", []):
            if track.track_id not in temp_ids["Medal"]:
                temp_ids["Medal"].add(track.track_id)
                #self._log_medal(frame_idx)


    def _handle_event_tracks(
        self,
        tracks,
        seen_ids,
        frame_buffer,
        frame_idx,
        video_index,
        event_type
    ):
        for track in tracks:
            if track.track_id not in seen_ids:
                seen_ids.add(track.track_id)
                if frame_buffer:
                    self.finalize_event(frame_buffer, video_index, event_type)
                    frame_buffer.clear()
            frame_buffer.append(frame_idx)


    def _finalize_video_events(self, video_index, kill_frames, death_frames):
        if kill_frames and "Kill" in self.events_to_detect:
            self.finalize_event(kill_frames, video_index, EventType.KILL)

        if death_frames and "Death" in self.events_to_detect:
            self.finalize_event(death_frames, video_index, EventType.DEATH)

        if self.events:
            add_to_json(self.filename, self.events)
            self.events.clear()


    def _process_clips(self):
         with open('events_temp_2.json', 'r') as f:
            events = json.load(f)
            clip_events = []
            for event in events:
                video_path = self.video_path[int(event['desc'][-14])-1]
                if (event['type'] == 'KILL' or event['type'] == 'KILLSTREAK') and "Kill" in self.events_to_detect:
                    clip_events.append((self.KILL_DIR, event, video_path))
                elif event['type'] == 'DEATH' and "Death" in self.events_to_detect:
                    clip_events.append((self.DEATH_DIR, event, video_path))
            progress_bar = tqdm(total=len(clip_events), desc="Extracting clips", unit="clip")

            for _ in range(self.max_workers):
                threading.Thread(target=self.clip_worker, args=(progress_bar,), daemon=True).start()

            for arg in clip_events:
                self.clip_queue.put(arg)

            self.clip_queue.join()

            for _ in range(self.max_workers):
                self.clip_queue.put(None)

            progress_bar.close()
        

    def _create_montage(self):
        best_kill_clips = self.kills_proc.find_best_kills()
        #best_kill_clips = [('Clients/Kills/KILLSTREAKin3@00.06.50.mp4', 7), ('Clients/Kills/KILLSTREAKin3@00.38.25.mp4', 4), ('Clients/Kills/KILLin1@00.56.39.mp4', 3), ('Clients/Kills/KILLin3@00.12.12.mp4', 3), ('Clients/Kills/KILLin2@00.09.48.mp4', 2), ('Clients/Kills/KILLin2@00.10.00.mp4', 2), ('Clients/Kills/KILLin2@00.10.43.mp4', 2), ('Clients/Kills/KILLin2@00.24.06.mp4', 2), ('Clients/Kills/KILLin2@00.44.39.mp4', 2), ('Clients/Kills/KILLin2@00.45.03.mp4', 2), ('Clients/Kills/KILLin2@00.53.12.mp4', 2), ('Clients/Kills/KILLin2@00.53.35.mp4', 2), ('Clients/Kills/KILLin3@00.08.29.mp4', 2), ('Clients/Kills/KILLin3@00.30.58.mp4', 2), ('Clients/Kills/KILLin2@00.08.33.mp4', 1), ('Clients/Kills/KILLin2@00.11.29.mp4', 1), ('Clients/Kills/KILLin2@00.11.31.mp4', 1), ('Clients/Kills/KILLin2@00.13.17.mp4', 1), ('Clients/Kills/KILLin2@00.13.32.mp4', 1), ('Clients/Kills/KILLin2@00.14.20.mp4', 1), ('Clients/Kills/KILLin2@00.14.27.mp4', 1), ('Clients/Kills/KILLin2@00.16.30.mp4', 1), ('Clients/Kills/KILLin2@00.16.42.mp4', 1), ('Clients/Kills/KILLin2@00.17.15.mp4', 1), ('Clients/Kills/KILLin2@00.17.40.mp4', 1), ('Clients/Kills/KILLin2@00.22.33.mp4', 1), ('Clients/Kills/KILLin2@00.23.11.mp4', 1), ('Clients/Kills/KILLin2@00.23.23.mp4', 1), ('Clients/Kills/KILLin2@00.23.44.mp4', 1), ('Clients/Kills/KILLin2@00.24.18.mp4', 1), ('Clients/Kills/KILLin2@00.24.20.mp4', 1), ('Clients/Kills/KILLin2@00.24.31.mp4', 1), ('Clients/Kills/KILLin2@00.24.47.mp4', 1), ('Clients/Kills/KILLin2@00.26.36.mp4', 1), ('Clients/Kills/KILLin2@00.27.52.mp4', 1), ('Clients/Kills/KILLin2@00.44.16.mp4', 1), ('Clients/Kills/KILLin2@00.52.53.mp4', 1), ('Clients/Kills/KILLin2@00.56.59.mp4', 1), ('Clients/Kills/KILLin2@00.57.07.mp4', 1), ('Clients/Kills/KILLin2@00.58.21.mp4', 1), ('Clients/Kills/KILLin2@00.58.30.mp4', 1), ('Clients/Kills/KILLin3@00.08.15.mp4', 1), ('Clients/Kills/KILLin3@00.12.21.mp4', 1), ('Clients/Kills/KILLin3@00.23.57.mp4', 1), ('Clients/Kills/KILLSTREAKin3@00.09.29.mp4', 1), ('Clients/Kills/KILLSTREAKin3@00.37.35.mp4', 1), ('Clients/Kills/KILLin1@00.14.42.mp4', 0), ('Clients/Kills/KILLin1@00.15.31.mp4', 0), ('Clients/Kills/KILLin1@00.15.57.mp4', 0), ('Clients/Kills/KILLin1@00.16.16.mp4', 0), ('Clients/Kills/KILLin1@00.17.36.mp4', 0), ('Clients/Kills/KILLin1@00.17.47.mp4', 0), ('Clients/Kills/KILLin1@00.18.04.mp4', 0), ('Clients/Kills/KILLin1@00.18.22.mp4', 0), ('Clients/Kills/KILLin1@00.18.48.mp4', 0), ('Clients/Kills/KILLin1@00.19.36.mp4', 0), ('Clients/Kills/KILLin1@00.19.38.mp4', 0), ('Clients/Kills/KILLin1@00.20.24.mp4', 0), ('Clients/Kills/KILLin1@00.21.56.mp4', 0), ('Clients/Kills/KILLin1@00.22.13.mp4', 0), ('Clients/Kills/KILLin1@00.23.18.mp4', 0), ('Clients/Kills/KILLin1@00.23.28.mp4', 0), ('Clients/Kills/KILLin1@00.23.39.mp4', 0), ('Clients/Kills/KILLin1@00.23.52.mp4', 0), ('Clients/Kills/KILLin1@00.47.42.mp4', 0), ('Clients/Kills/KILLin1@00.53.32.mp4', 0), ('Clients/Kills/KILLin1@00.58.02.mp4', 0), ('Clients/Kills/KILLin1@00.58.18.mp4', 0), ('Clients/Kills/KILLin2@00.08.05.mp4', 0), ('Clients/Kills/KILLin2@00.10.04.mp4', 0), ('Clients/Kills/KILLin2@00.13.02.mp4', 0), ('Clients/Kills/KILLin2@00.17.36.mp4', 0), ('Clients/Kills/KILLin2@00.21.57.mp4', 0), ('Clients/Kills/KILLin2@00.24.00.mp4', 0), ('Clients/Kills/KILLin2@00.25.14.mp4', 0), ('Clients/Kills/KILLin2@00.25.56.mp4', 0), ('Clients/Kills/KILLin2@00.26.09.mp4', 0), ('Clients/Kills/KILLin2@00.27.05.mp4', 0), ('Clients/Kills/KILLin2@00.40.56.mp4', 0), ('Clients/Kills/KILLin2@00.40.57.mp4', 0), ('Clients/Kills/KILLin2@00.41.59.mp4', 0), ('Clients/Kills/KILLin2@00.43.18.mp4', 0), ('Clients/Kills/KILLin2@00.46.27.mp4', 0), ('Clients/Kills/KILLin2@00.46.38.mp4', 0), ('Clients/Kills/KILLin2@00.51.09.mp4', 0), ('Clients/Kills/KILLin2@00.53.30.mp4', 0), ('Clients/Kills/KILLin2@00.54.00.mp4', 0), ('Clients/Kills/KILLin2@00.54.53.mp4', 0), ('Clients/Kills/KILLin2@00.55.38.mp4', 0), ('Clients/Kills/KILLin2@00.56.42.mp4', 0), ('Clients/Kills/KILLin2@00.56.48.mp4', 0), ('Clients/Kills/KILLin2@00.58.01.mp4', 0), ('Clients/Kills/KILLin2@00.58.53.mp4', 0), ('Clients/Kills/KILLin3@00.09.10.mp4', 0), ('Clients/Kills/KILLin3@00.15.10.mp4', 0), ('Clients/Kills/KILLin3@00.21.30.mp4', 0), ('Clients/Kills/KILLin3@00.24.05.mp4', 0), ('Clients/Kills/KILLin3@00.30.05.mp4', 0), ('Clients/Kills/KILLin3@00.32.01.mp4', 0), ('Clients/Kills/KILLin3@00.39.23.mp4', 0), ('Clients/Kills/KILLin3@00.39.31.mp4', 0), ('Clients/Kills/KILLin3@00.42.10.mp4', 0), ('Clients/Kills/KILLin3@00.43.00.mp4', 0), ('Clients/Kills/KILLin3@00.58.41.mp4', 0), ('Clients/Kills/KILLin3@00.59.06.mp4', 0), ('Clients/Kills/KILLin3@00.59.24.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.11.13.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.11.48.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.14.32.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.15.19.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.16.11.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.16.13.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.16.24.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.17.06.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.18.59.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.19.20.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.20.04.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.21.24.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.21.43.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.22.52.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.23.16.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.25.01.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.27.19.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.35.28.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.41.31.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.43.16.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.43.46.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.44.08.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.54.42.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.55.08.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.55.29.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.56.03.mp4', 0), ('Clients/Kills/KILLSTREAKin3@00.56.24.mp4', 0)]
        new_folder = ''.join((self.output_dir, '/best_kill_clips'))
        os.makedirs(new_folder, exist_ok=True)

        self.kills_proc.move_best_kills_to_folder(best_kill_clips, self.montage_length_sec, new_folder)
        montage = Montage()
        montage.make_compilation(new_folder, f"{self.output_dir}/highlight_reel.mp4")
        os.remove('events_temp_2.json')
        
        if not self.vertical_format:
            montage.make_tiktok(f"{self.output_dir}/highlight_reel.mp4", f"{self.output_dir}/highlight_reel_tiktok.mp4")
        

    def find_event_frames(self, event_frames, event_type: EventType):
        seconds_before = self.events_config['Kill']['pre']
        seconds_after = self.events_config['Kill']['post']

        if event_type.name == 'DEATH':
            seconds_before = self.events_config['Death']['pre']
            seconds_after = self.events_config['Death']['post']

        starting_frame = min(event_frames)-(self.fps*seconds_before)
        ending_frame = max(event_frames)+(self.fps*seconds_after)
        
        if starting_frame - self.fps <= 0:
            starting_frame = 0
        
        if ending_frame >= self.TOTAL_FRAMES_TO_BE_ANALYZED:
            ending_frame = self.TOTAL_FRAMES_TO_BE_ANALYZED
            
        return starting_frame/60, ending_frame/60


    def finalize_event(self, event_frames, video_num, event_type: EventType):
        starting_time, ending_time = self.find_event_frames(event_frames, event_type)
        event = Event(event_type, starting_time, ending_time, video_num)
        self.events.append(event)
        
        if self.add_to_csv:
            with self.events_csv_lock:
                self.events_csv.append({"Timestamp": time.strftime("%H:%M:%S", time.gmtime(starting_time)),
                                    "Event": event_type.name})          