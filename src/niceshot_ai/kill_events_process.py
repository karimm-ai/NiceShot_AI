from .utils import get_duration
from .event_types import Event, EventType

import json, os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import shutil

class KillEventsProcessor:
    """Finds top kill clips and kill streaks"""

    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir


    def find_best_kills(self):
        """Kill clips where a lot of medals pop up"""

        print("Extracting Best clips\n")

        model = YOLO(self.model_path)
        #model.to('cuda')
        medal_tracker = DeepSort(max_age=30)
        clips_medals = {}
        
        for clip in os.listdir(f"{self.output_dir}/Kills"):
            if clip.endswith("mp4"):
                clips_medals[f"{self.output_dir}/Kills/{clip}"] = 0
        
        conf_threshold = 0.85
        temp = set()
        
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
            #cv2.destroyAllWindows()

        sorted_clips_medals= sorted(clips_medals.items(), key=lambda item: item[1], reverse=True)
        print(sorted_clips_medals)
        return sorted_clips_medals


    def concat_temp_clips(self, clips_to_concat, output_file):
        # Create a text file with the list of input files
        #with open('file_list.txt', 'w') as file:
            #for clip, time in clips_to_concat:
                #file.write(f"file '{self.output_dir}/Kills/{clip}'\n")

        # Run FFmpeg to concatenate the videos
        #subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'file_list.txt', '-c', 'copy', output_file])
        # Clean up the temporary file list
        self.extract_segment(f"{self.output_dir}", 
                             round(clips_to_concat['time'][0]), 
                             round(clips_to_concat['time'][-1])+self.seconds_after_kill*2, 
                             output_file)
        #os.remove('file_list.txt')
        print(f"Found a kill streak. Videos concatenated successfully into {output_file}")
    

    def move_best_kills_to_folder(self, best_kills, montage_length, new_folder):
        print(f"Moving best clips to {self.output_dir}/{new_folder}\n")
        final_clips = []
        current_length = 0
        while current_length <= montage_length:
            if not len(best_kills) > 0:
                break
            
            vid_path = best_kills.pop(0)[0]
            final_clips.append(vid_path)
            current_length += get_duration(vid_path)

        for clip in final_clips:
            shutil.copy(clip, new_folder)


    def concat_kill_streaks_new(self, video_num):
        with open('events_temp.json', 'r') as f:
            events = json.load(f)
        
        kill_streaks = []
        current_streak = []
        gap_threshold = 3.0
        temp_events = []

        for event in events:
            if event["type"] != "KILL":
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
            if len(streak) > 1:
                merged.append(Event(EventType.KILLSTREAK,
                    streak[0]["timestart"],
                    streak[-1]["timeend"], video_num))

        merged = [event.to_dict() for event in merged]
        for event in events:
            if event not in temp_events:
                merged.append(event)
        del temp_events, events, kill_streaks, current_streak

        with open('events_temp_2.json', 'w') as f:
            json.dump(merged, f, indent=2)
        
        #os.remove('events_temp.json')