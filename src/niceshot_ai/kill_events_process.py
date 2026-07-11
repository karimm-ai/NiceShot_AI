from utils import get_duration, report_progress
from event_types import Event

import json, os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm


class KillEventsProcessor:
    """Finds top kill clips and kill streaks"""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir


    def find_best_kills(self) -> dict:
        """Kill clips where a lot of medals pop up"""
        self.progress = set(range(3, 101, 3))
        print("Extracting Best clips\n")

        model = YOLO(self.model_path).to("cuda")
        medal_tracker = DeepSort(max_age=30)
        clips_medals = {}
        kill_directories = [f"{self.output_dir}/Kill", f"{self.output_dir}/KillStreak"]
        
        for dir in kill_directories:
            for clip in os.listdir(dir):
                if clip.endswith("mp4"):
                    clips_medals[f"{dir}/{clip}"] = 0
            
            conf_threshold = 0.85
            temp = set()
        
        self.total_clips = len(clips_medals)
        self.analyzed_clips = 0
        
        for key, _ in clips_medals.items():
            cap = cv2.VideoCapture(key)
            TOTAL_FRAMES_TO_BE_ANALYZED = get_duration(key)*60

            with tqdm(total=TOTAL_FRAMES_TO_BE_ANALYZED, desc="Processing video", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, verbose=False)[0]
                    detections = []

                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        
                        if conf>=conf_threshold and cls == 1:
                            detections.append(([x1, y1, x2-x1, y2-y1] , conf, cls))
                        
                    tracks = medal_tracker.update_tracks(detections, frame=frame)

                    for track in tracks:
                        if track.track_id not in temp:
                            clips_medals[key]+=1
                            temp.add(track.track_id)

                    pbar.update(1)
            
            cap.release()
            self.analyzed_clips+=1
            report_progress(self.output_dir, self.analyzed_clips, self.total_clips, self.progress, "FINDING BEST KILLS...")

        sorted_clips_medals = sorted(clips_medals.items(), key=lambda item: item[1], reverse=True)
        final_clips = [clip_path for clip_path, _ in sorted_clips_medals]
        return final_clips


    def concat_kill_streaks(self, video_num: int):
        with open(f"{self.output_dir}/events_temp.json", 'r') as f:
            events = json.load(f)
        
        kill_streaks = []
        current_streak = []
        gap_threshold = 3.0
        temp_events = []

        for event in events:
            if event["type"] != "Kill":
                # reset streak if any non-KILL occurs
                if current_streak:
                    kill_streaks.append(current_streak)
                    current_streak = []
                continue

            if not current_streak:
                current_streak.append(event)

            else:
                prev = current_streak[-1]
                gap = event["timestart"] - prev["timeend"]
                if gap <= gap_threshold:
                    current_streak.append(event)
                else:
                    kill_streaks.append(current_streak)
                    current_streak = [event]

        if current_streak:
            kill_streaks.append(current_streak)

        for streak in kill_streaks:
            if len(streak) > 1:
                for kill in streak:
                    temp_events.append(kill)

        merged = []
        for streak in kill_streaks:
            if len(streak) > 6:
                merged.append(Event(type="KillStreak",
                    timestart=streak[0]["timestart"],
                    timeend=streak[-1]["timeend"],
                    video_num=video_num,
                    kills=len(streak)))

        merged = [event.to_dict() for event in merged]
        # for event in events:
        #     if event not in temp_events:
        #         merged.append(event)
        del temp_events, events, kill_streaks, current_streak

        with open(f"{self.output_dir}/events_temp_2.json", 'w') as f:
            json.dump(merged, f, indent=2)