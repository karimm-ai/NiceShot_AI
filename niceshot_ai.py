from dataclasses import dataclass
import os, sys, subprocess, threading
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import logging
from queue import Queue
import csv, time, json, shutil
from rapidocr import RapidOCR
from enum import Enum, auto
import yt_dlp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


def get_duration(clip_path):
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


class EventType(Enum):
    """Types of Events: Model can detect only Kills, Deaths, Medal pop ups"""

    KILL = auto()
    MEDAL = auto()
    DEATH = auto()
    KILLSTREAK = auto()



@dataclass
class Event:
    """Attributes for an event"""

    type: EventType
    timestart: float
    timeend: float
    video_num: int
    timestamp: str = ""
    desc: str = ""
    
    def __post_init__(self):
        self.timestamp = time.strftime("%H:%M:%S", time.gmtime(self.timestart))
        self.timestamp = self.timestamp.replace(":", ".")
        self.desc = f"{self.type.name}in{str(self.video_num)}@{self.timestamp}.mp4"


    def to_dict(self):
        return {
            "type": self.type.name,
            "timestart": self.timestart,
            "timeend": self.timeend,
            "timestamp": self.timestamp,
            "desc": self.desc
        }



class Clipper:
    """Clipper class for clipping segments from a video path"""

    def __init__(self, ffmpeg_path, vertical_format):
        self.ffmpeg_path = ffmpeg_path
        self.vertical_format = vertical_format
        self.crop_width = 608
        self.crop_height = 1080
        self.x_offset = "(in_w - {0})/2".format(self.crop_width)
        self.y_offset = "(in_h - {0})/2".format(self.crop_height)


    def clip_event(self, output_dir: str, event, video_path):  
        output_path = os.path.join(output_dir, event['desc'])

        if not self.vertical_format:
            subprocess.run([
            self.ffmpeg_path,
            "-ss", str(event['timestart']),
            "-i", video_path,
            "-to", str(event['timeend'] - event['timestart']),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path
            ])
        
        else:
            cmd = [
                "ffmpeg",
                "-ss", str(event['timestart']),
                "-i", video_path,
                "-to", str(event['timeend'] - event['timestart']),
                "-filter:v",
                f"crop={self.crop_width}:{self.crop_height}:{self.x_offset}:{self.y_offset},scale=1080:1920,setsar=1",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-y",  # Overwrite output if exists
                output_path
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"✅ Successfully extracted vertical TikTok video: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg error: {e.stderr.decode()}")



class KillEventsProcessor:
    """Finds top kill clips and kill streaks"""

    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir


    def find_best_kills(self):
        """Kill clips where a lot of medals pop up"""

        print("Extracting Best clips\n")

        model = YOLO(self.model_path)
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
            cv2.destroyAllWindows()

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


    """Old Version
    def concat_kill_streaks(self,):
        time_old = 0
        time_new = 0
        clips_to_concat = {'clips': [], 'time': []}
        collected_clips = False
        print("Starting Kill Streaks Analysis!\n")
        for clip in os.listdir(f"{self.output_dir}/Kills"):
            time_old = time_new
            clip_imp = clip.replace(".mp4", "")[5:]
            clip_time = clip_imp.split('.')
            print(clip_time)
            if len(clip_time) == 2:
                time_new = int(clip_time[0])*60 + int(clip_time[1])

            elif len(clip_time) == 3:
                time_new = int(clip_time[0])*60*60 + int(clip_time[1])*60 + int(clip_time[2])

            print(time_new)

            if time_new > time_old+4 and len(clips_to_concat['clips']) >=2:
                collected_clips = True

            elif time_new > time_old+4 and len(clips_to_concat['clips']) ==1:
                clips_to_concat['clips'].pop()
                clips_to_concat['time'].pop()
                clips_to_concat['clips'].append(clip)
                clips_to_concat['time'].append(time_new)

            else:
                clips_to_concat['clips'].append(clip)
                clips_to_concat['time'].append(time_new)


            if collected_clips:
                kill_streak = len(clips_to_concat['clips'])
                self.concat_temp_clips(clips_to_concat, f"{self.output_dir}/Kills/{kill_streak}_kills_streak@{clips_to_concat['clips'][0][5:]}.mp4")
                clips_to_concat["clips"] = []
                clips_to_concat["time"] = []
                clips_to_concat["clips"].append(clip)
                clips_to_concat["time"].append(time_new)
                collected_clips = False
    """


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
        
        os.remove('events_temp.json')
    


class Montage:
    """Compiles all clips within a folder into 1 clip with simple edit and converts a video from horizontal aspect to vertical"""

    def __init__(self,):
        pass
   

    def make_compilation(self, input_folder, output_file, fade_duration=0.5):
        print("Creating Montage!\n")
        clips = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')])
        if not clips:
            print("❌ No clips found.")
            return

        input_args = []
        durations = []
        for clip in clips:
            path = os.path.join(input_folder, clip)
            input_args.extend(["-i", path])
            durations.append(get_duration(path))

        filter_parts = []
        v_streams = []
        a_streams = []

        for i, duration in enumerate(durations):
            fade_out_start = max(0, duration - fade_duration)
            filter_parts.append(
                f"[{i}:v]fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration},setpts=PTS-STARTPTS[v{i}];"
            )
            filter_parts.append(
                f"[{i}:a]afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration},asetpts=PTS-STARTPTS[a{i}];"
            )
            v_streams.append(f"[v{i}]")
            a_streams.append(f"[a{i}]")

        filter_parts.append(f"{''.join(v_streams)}concat=n={len(clips)}:v=1:a=0[v];")
        filter_parts.append(f"{''.join(a_streams)}concat=n={len(clips)}:v=0:a=1[a]")

        filter_complex = "".join(filter_parts)

        cmd = ["ffmpeg"]
        cmd.extend(input_args)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            output_file
        ])

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Montage with fades created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")


    def make_tiktok(self, video_path, output_path):
        # Crop width and height for center vertical slice
        crop_width = 608
        crop_height = 1080

        # Calculate x and y offsets (expressed as FFmpeg expressions)
        x_offset = "(in_w - {0})/2".format(crop_width)
        y_offset = "(in_h - {0})/2".format(crop_height)

        # FFmpeg command with crop and scale
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-filter:v",
            f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale=1080:1920,setsar=1",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-y",  # Overwrite output if exists
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully created vertical TikTok video: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")



class NiceShot_AI:
    """Main Class for detecting events"""

    def __init__(self, model_path,
                 ffmpeg_path,
                 video_path,
                 csv_file=None,
                 count_kills=True,
                 count_deaths=True,
                 count_medals=True,
                 total_hours=10,
                 save_clips=True,
                 output_dir=".",
                 max_workers=2,
                 seconds_before_kill=2, 
                 seconds_after_kill=2, 
                 seconds_before_death=2,
                 seconds_after_death=1,
                 frame_idx_start= 0,
                 frames_to_skip=5,
                 add_to_csv=False,
                 create_montage=False,
                 montage_length_sec=20,
                 max_videos = 1,
                 vertical_format = False
                 ):

        self.output_dir = output_dir
        self.video_path = [video_path]
        self.csv_file = csv_file
        self.count_kills = count_kills
        self.count_deaths = count_deaths
        self.count_medals = count_medals
        self.max_workers = max_workers
        self.total_hours = total_hours
        self.seconds_before_kill = seconds_before_kill
        self.seconds_after_kill = seconds_after_kill
        self.seconds_before_death = seconds_before_death
        self.seconds_after_death = seconds_after_death
        self.save_clips = save_clips
        self.frame_idx_start = frame_idx_start
        self.frames_to_skip = frames_to_skip
        self.add_to_csv = add_to_csv
        self.create_montage = create_montage
        self.events = []
        self.filename = 'events_temp.json'
        self.montage_length_sec = montage_length_sec
        self.vertical_format = vertical_format

        self.model_path = self.resource_path(model_path)
        self.ffmpeg_path = self.resource_path(ffmpeg_path)
        print(self.ffmpeg_path)
        
        self.ocr = RapidOCR()

        if 'twitch' in self.video_path[0]:
            twitch_handler = TwitchHandler(self.video_path[0], max_videos, self.output_dir)
            vods = twitch_handler.get_all_videos()
            with open ('vods.txt', 'w') as file:
                for vod in vods:
                    file.write(f"{vod}\n")
            
            twitch_handler.download_channel_videos(vods)
            self.video_path = [f"{self.output_dir}/Downloads/{file}" for file in os.listdir(f"{self.output_dir}/Downloads")]
            
        if self.save_clips:
            #self.ffmpeg_path = self.resource_path(ffmpeg_path)
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


    def resource_path(self, filename):
        if getattr(sys, 'frozen', False):
            return os.path.join(sys._MEIPASS, filename)
        return filename
    

    def add_to_csv_(self, filename, events):
        output_filename = os.path.join(self.output_dir, filename)
        fieldnames = ["Timestamp", "Event"]
        
        with open(output_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if file.tell() == 0:
                writer.writeheader()
            
            for event in events:
                writer.writerow(event)


    def detect_events(self, progress_bar=None):
        os.makedirs(self.output_dir, exist_ok=True)
        
        model = YOLO(self.model_path)

        if self.count_kills:
            kill_tracker = DeepSort(max_age=0)
        
        medal_tracker = DeepSort(max_age=30, nms_max_overlap=0.6)
        
        if self.count_deaths:
            death_tracker = DeepSort(max_age=30)

        for i, video_path in enumerate(self.video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration = total_frames/self.fps/60/60

            if video_duration >= self.total_hours:
                self.TOTAL_FRAMES_TO_BE_ANALYZED = self.total_hours*60*60*self.fps

            else:
                self.TOTAL_FRAMES_TO_BE_ANALYZED = video_duration*60*60*self.fps
            
            print(f"Total Frames {total_frames}\nFPS {self.fps}\nVideo Duration {video_duration}\nTotal Frames to be analyzed {self.TOTAL_FRAMES_TO_BE_ANALYZED}")

            conf_thresholds = {
                0: 0.6, # Kill
                1: 0.85, # Medal
                2: 0.8, # Death
            }

            frame_idx = 0
            kill_temp = set()
            medal_temp = set()
            death_temp = set()

            kill_frames = []
            death_frames = []

            with tqdm(total=self.TOTAL_FRAMES_TO_BE_ANALYZED-frame_idx, desc="Processing video", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_idx >= self.TOTAL_FRAMES_TO_BE_ANALYZED:
                        break
                    
                    if frame_idx >= self.frame_idx_start:
                        if frame_idx >= 0:
                            if frame_idx % self.frames_to_skip == 0:
                                results = model(frame, verbose=False)[0]
                                
                                kill_detections = []
                                medal_detections = []
                                death_detections = []

                                for box in results.boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    conf = box.conf.item()
                                    cls = int(box.cls.item())
                                    
                                    if conf>=conf_thresholds.get(cls) and cls == 0: # KILL
                                        if self.count_kills:
                                            if not self.is_invalid_event(frame):
                                                kill_detections.append(([x1, y1, x2-x1, y2-y1] , conf, cls))

                                    elif conf>=conf_thresholds.get(cls) and cls == 1: # MEDAL
                                        if not self.is_invalid_event(frame):
                                            medal_detections.append(([x1, y1, x2-x1, y2-y1] , conf, cls))

                                    elif conf>=conf_thresholds.get(cls) and cls == 2: # DEATH
                                        if self.count_deaths:
                                            death_detections.append(([x1, y1, x2-x1, y2-y1] , conf, cls))
                                        
                                if self.count_kills:
                                    kill_tracks = kill_tracker.update_tracks(kill_detections, frame=frame)
                                
                                medal_tracks = medal_tracker.update_tracks(medal_detections, frame=frame)
                                
                                if self.count_deaths:
                                    death_tracks = death_tracker.update_tracks(death_detections, frame=frame)
                                
                                if self.count_kills:
                                    for track in kill_tracks:
                                        if track.track_id not in kill_temp:
                                            kill_temp.add(track.track_id)

                                            if len(kill_frames) > 0:
                                                self.finalize_event(kill_frames, i+1, EventType.KILL)
                                                kill_frames.clear()
                                                kill_frames.append(frame_idx)
                                                
                                        else:
                                            kill_frames.append(frame_idx)
                                

                                for track in medal_tracks:
                                    if track.track_id not in medal_temp:
                                        medal_temp.add(track.track_id)

                                        if self.add_to_csv and self.count_medals:
                                            timestamp = time.strftime("%H:%M:%S", time.gmtime(frame_idx/self.fps))
                                            
                                            with self.events_csv_lock:
                                                self.events_csv.append({"Timestamp": timestamp, "Event": "Medal"})

                                if self.count_deaths:
                                    for track in death_tracks:
                                        if track.track_id not in death_temp:
                                            death_temp.add(track.track_id)

                                            if len(death_frames) > 0:
                                                self.finalize_event(death_frames, i+1, EventType.DEATH)
                                                death_frames.clear()
                                                death_frames.append(frame_idx)
                                                
                                        else:
                                            death_frames.append(frame_idx)
                            
                    frame_idx += 1
                    pbar.update(1)
                    if progress_bar:
                        progress_bar['value'] = (frame_idx / self.TOTAL_FRAMES_TO_BE_ANALYZED) * 100
                        progress_bar.update()

                    if len(self.events_csv) >= 100:
                        self.add_to_csv_(self.csv_file, self.events_csv)
                        self.events_csv.clear()

                    if len(self.events) >= 10:
                        self.add_to_json()
                        self.events.clear()


            if len(kill_frames) > 0 and self.count_kills:
                self.finalize_event(kill_frames, i+1, EventType.KILL)
                del kill_frames

            if len(death_frames) > 0 and self.count_deaths:
                self.finalize_event(death_frames, i+1, EventType.DEATH)
                del death_frames
                
            if len(self.events) > 0:
                self.add_to_json()
                self.events.clear()

            if self.count_kills:
                self.kills_proc.concat_kill_streaks_new(i+1)

            cap.release()
            cv2.destroyAllWindows()
            print(f"Finished procssing {video_path}\n")

        
        if self.add_to_csv and len(self.events_csv) > 0:
            self.add_to_csv_(self.csv_file, self.events_csv)
            del self.events_csv
            del self.events

        if self.save_clips:    
            with open('events_temp_2.json') as f:
                events = json.load(f)
            clip_events = []
            for event in events:
                video_path = self.video_path[int(event['desc'][-14])-1]
                if (event['type'] == 'KILL' or event['type'] == 'KILLSTREAK') and self.count_kills:
                    clip_events.append((self.KILL_DIR, event, video_path))
                elif event['type'] == 'DEATH' and self.count_deaths:
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

        if (self.create_montage or self.montage_length_sec > 0) and self.count_kills:# and self.save_clips
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
        seconds_before = self.seconds_before_kill
        seconds_after = self.seconds_after_kill

        if event_type.name == 'DEATH':
            seconds_before = self.seconds_before_death
            seconds_after = self.seconds_after_death

        starting_frame = min(event_frames)-(self.fps*seconds_before)
        ending_frame = max(event_frames)+(self.fps*seconds_after)
        
        if starting_frame - self.fps <= 0:
            starting_frame = 0
        
        if ending_frame >= self.TOTAL_FRAMES_TO_BE_ANALYZED:
            ending_frame = self.TOTAL_FRAMES_TO_BE_ANALYZED
            
        return starting_frame/60, ending_frame/60


    def extract_text(self, frame):
        #x, y, w, h = region
        #cropped = frame[y:y+h, x:x+w]
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = self.ocr(frame)
        text = ''.join([word for word in text.txts])
        return text


    def is_invalid_event(self, frame):
        killcam_text = self.extract_text(frame)

        words = ("KILLCAM", "KILLGAM", "BESTPLAY", "SPECTATING:", "FINAL KILL", "BEST PLAY")
        for word in words:
            if word.lower() in killcam_text.lower():
                return True
        return False
    

    def add_to_json(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.extend([e.to_dict() for e in self.events])

        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)


    def finalize_event(self, event_frames, video_num, event_type: EventType):
        starting_time, ending_time = self.find_event_frames(event_frames, event_type)
        event = Event(event_type, starting_time, ending_time, video_num)
        self.events.append(event)
        #if self.save_clips and self.count_kills:
        #    self.clip_queue.put((self.KILL_DIR, event))
        
        if self.add_to_csv:
            with self.events_csv_lock:
                self.events_csv.append({"Timestamp": time.strftime("%H:%M:%S", time.gmtime(starting_time)),
                                    "Event": event_type.name})
                


class TwitchHandler:
    """Handles Twitch videos grabbing and downloading using Selenium"""

    def __init__(self, channel_link, max_videos, output_dir):
        self.channel_link = channel_link
        self.max_videos = max_videos
        self.output_dir = output_dir


    def get_all_videos(self,):
        """Returns all detected Call of Duty: Black Ops 6 videos on a user's channel"""

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=options)
        driver.get(f"{self.channel_link}/videos?filter=all&sort=time")
        desired_game = "Call of Duty: Black Ops 6"
        video_urls = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        print("Fetching videos")

        while True:
            video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/videos/')]")
            current_video_urls = [element.get_attribute('href') for element in video_elements]

            new_video_urls = set(current_video_urls) - video_urls
            if not new_video_urls:
                print("No new videos found.")
                break
            
            video_urls.update(new_video_urls)

            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, '/videos/')]"))
            )

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("Reached the end of the page, no more content.")
                break
            last_height = new_height

        print(f"Found {len(video_urls)} unique videos.")

        filtered_video_urls = []
        for url in video_urls:
            driver.get(url)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/directory/')]"))
                )
                game_element = driver.find_element(By.XPATH, "//a[contains(@href, '/directory/')]")
                game_name = game_element.text.strip().lower()

                if game_name == desired_game.lower():
                    filtered_video_urls.append(url)

            except Exception as e:
                print(f"Error loading video {url}: {e}")

        print(f"Found {len(filtered_video_urls)} videos for {desired_game}.")
        for video in filtered_video_urls:
            print(video)
        
        driver.quit()
        return filtered_video_urls
    

    def download_video(self, video, name):
        """Downloads a single video from Twitch using yt-dlp"""

        save_path = f"{self.output_dir}/Downloads"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ydl_opts = {
            'outtmpl': os.path.join(save_path, f'{name}.%(ext)s'),
            'format': 'best',
        }

        try:
            print(f"Downloading: {video}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video])
            print(f"Download completed for: {video}")
        except Exception as e:
            print(f"Error downloading {video}: {e}")


    def download_channel_videos(self, links):
        """Downloads videos from the grabbed Twitch links"""

        for i in range(self.max_videos):
            print(f"Downloading Video {i+1} from {links[i]}")
            self.download_video(links[i], f'{i}')

            for file in os.listdir(f"{self.output_dir}/Downloads"):
                if not file.endswith('.mp4') or 'temp' in file:
                    os.remove(f"{self.output_dir}/Downloads/{file}")