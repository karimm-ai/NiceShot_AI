from utils import resource_path
from event_confirm import EventConfirm
from configs.games_config import *

import logging, os, json
import random
from multiprocessing import Process


def start_process(target):
        p = Process(target=target)
        p.start()
        p.join()



class NiceShot_AI:
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
        session_analysis: bool = False,
        coaching: str | None = None
        ):

        self.events_config = supported_games[game_name.lower()]
        self.conf_thresholds = {cls: val["conf_thres"] for cls, val in enumerate(self.events_config.values())}
        self.model_path = resource_path(models_paths[game_name.lower()])
        
        if session_analysis:
            self.report_config = available_charts[game_name.lower()]

        if advanced_detection:
            self.last_known_context = None

        self.output_dir = output_dir
        self.video_path = video_path
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
        self.coaching = coaching

        self.ffmpeg_path = resource_path("src/niceshot_ai/ffmpeg.exe")
        print(f"FFMPEG PATH: {self.ffmpeg_path}")

        if advanced_detection:
            self.event_confirm = EventConfirm()

        logging.basicConfig(
            filename="tracker.log",
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


    def run(self, progress_bar=None):
        os.makedirs(self.output_dir, exist_ok=True)

        start_process(self.run_detector)

        if "Kill" in self.events_config:
            start_process(self.handle_special_events)

        if self.save_clips:
            start_process(self.run_video_clipper)

        if self.save_clips and self.create_montage and self.montage_length_sec > 0:
            start_process(self.run_montage)

        if hasattr(self, "report_config") and self.add_to_csv:
            start_process(self.run_reporting)

        if self.coaching is not None:
            start_process(target=self.run_observer)
            start_process(target=self.run_coach)


    def run_detector(self):
        from detectors.yolo_detector import YOLODetector
        
        detector = YOLODetector(self.model_path,
                                self.events_config,
                                self.conf_thresholds,
                                self.video_path,
                                self.total_hours,
                                self.output_dir,
                                self.frame_idx_start,
                                self.frames_to_skip,
                                self.add_to_csv,
                                self.event_confirm)

        self.csv_file = detector.detect_events()



    def handle_special_events(self):
        from kill_events_process import KillEventsProcessor
        self.kills_proc = KillEventsProcessor(self.output_dir)
        self.kills_proc.concat_kill_streaks(1)


    def run_video_clipper(self, meta_file="events_temp_2.json"):
        from video_clipper import Clipper

        self.clipper = Clipper(self.output_dir, self.video_path, self.ffmpeg_path, self.vertical_format, self.max_workers)
        self.clipper._process_clips(meta_file)



    def run_reporting(self):
        from report import ReportMaker

        duration_to_be_analyzed = self.total_hours * 60
        bucket_len = int(duration_to_be_analyzed // 10)    # minutes
        if bucket_len == 0:
            bucket_len = 1
        print(bucket_len)
        report = ReportMaker(self.output_dir, f"{self.output_dir}/video1.csv",
                                self.events_config, self.report_config, bucket_len)
        for chart in self.report_config["charts"]:
            func = getattr(report, chart["name"])
            func(self.report_config["color_pallete"], chart["width"], chart["height"])
        report.save_report(1)



    def run_observer(self):
        coaching_events = []
        
        for key, val in self.events_config.items():
            if val.get('vlm_prompt'):
                coaching_events.append(key)

        from observer import Observer
        observer = Observer(self.output_dir, self.coaching)

        if not self.save_clips:
            events_file = "events_temp_3.json"
            all_events = {}
            with open(f"{self.output_dir}/events_temp_2.json", 'r') as f:
                events = json.load(f)

            for event in events:
                if event['type'] in coaching_events:
                    if event['type'] not in all_events.keys():
                        all_events[event['type']] = []
                        all_events[event['type']].append(event)

                    else:
                        all_events[event['type']].append(event)

            for key, val in all_events.items():
                k = max(1, int(len(val) * observer.sample_size))
                sample = random.sample(val, k=k)

                with open(f"{self.output_dir}/{events_file}", 'w') as f:
                    json.dump(sample, f, indent=2)

                self.run_video_clipper(meta_file=events_file)
                observer.sample_size = 1
        
        for key, val in self.events_config.items():
            if val.get('vlm_prompt'):
                observer.analyze_session(val.get('vlm_prompt'), f"{self.output_dir}/{key}")

        del observer


    def run_coach(self):
        from coach import Coach

        print("Running Coach...")
        ai_coach = Coach(self.output_dir)
        for key, val in self.events_config.items():
            if val.get('llm_prompt'):
                results = ai_coach.analyze_session(f"{self.output_dir}/{key}.jsonl", val.get('llm_prompt'))
                with open(f"{self.output_dir}/{key}_coaching.jsonl", 'w') as f:
                    json.dump(results, f, indent=2)

        del ai_coach


    def run_montage(self):
        from montage import Montage

        montage = Montage(self.output_dir, self.montage_length_sec, self.events_config, self.ffmpeg_path, self.vertical_format)
        montage._create_montage()