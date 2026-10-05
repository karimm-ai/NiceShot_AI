from utils import resource_path
from event_confirm import EventConfirm
from configs.games_config import *
from storage import SQLiteDB

import logging, os, json
import random
from multiprocessing import Process
from datetime import datetime
from pathlib import Path
import re


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
            self.event_confirm = EventConfirm()

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

        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.game_name = game_name


        self.ffmpeg_path = resource_path("src/niceshot_ai/ffmpeg.exe")
        print(f"FFMPEG PATH: {self.ffmpeg_path}")

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


    def init_db(self, path):
        self.sqlitedb = SQLiteDB(path)
        self.sqlitedb.create_table("Session",
                              """
                                session_id INTEGER PRIMARY KEY,
                                game_name TEXT NOT NULL,
                                summary TEXT
                              """)
        self.sqlitedb.create_table("Event",
                              """
                                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                session_id INTEGER NOT NULL,
                                event_type TEXT NOT NULL,
                                timestamp TEXT NOT NULL,
                                data TEXT NOT NULL
                              """)



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


    def run_montage(self):
        from montage import Montage

        montage = Montage(self.output_dir, self.montage_length_sec, self.events_config, self.ffmpeg_path, self.vertical_format)
        montage._create_montage()


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
        db_path = Path(__file__).resolve().parent / "niceshot.db"
        
        if not db_path.exists():
            self.init_db(db_path)

        self.sqlitedb = SQLiteDB(db_path)

        from coach import Coach

        print("Running Coach...")
        ai_coach = Coach(self.output_dir)
        for key, val in self.events_config.items():
            if val.get('llm_prompt'):
                results = ai_coach.analyze_session(f"{self.output_dir}/{key}.jsonl", val.get('llm_prompt'))
                
                self.add_events_to_db(key, results)
                
                with open(f"{self.output_dir}/{key}_coaching.jsonl", 'w') as f:
                    json.dump(results, f, indent=2)

        session_results = self.retrieve_session_results()

        llm_summary_prompt = f"""
            Analyze the player's session using the records below and provide a concise coaching summary.

            Identify:
            The most important general recurring patterns across the session.
            2–3 specific, actionable improvements for future sessions.

            Use only information supported by the records. Ignore corrupted, duplicated, incomplete, or irrelevant text, and do not invent events or details. If the data is unclear or conflicting, avoid making assumptions.
            Focus on practical insights rather than describing every event. Prioritize repeated or high-impact issues and keep the final summary clear, concise, and useful to the player.

            Records:
            {session_results}

        """
        messages = [
                    {
                        "role": "system",
                        "content": "You are a concise game-session coach. Use only the provided session records. Ignore corrupted or duplicated data. Never invent facts. Identify important patterns and give practical, actionable advice."
                    },
                    {
                        "role": "user",
                        "content": llm_summary_prompt
                    }
                    ]

        session_summary = ai_coach.infer(messages)
        print(f"SESSION SUMMARY\n{session_summary}")

        self.add_summary_to_db(session_summary)

        del ai_coach


    def add_events_to_db(self, event: str, records: list):
        counter = 0
        records_to_insert = []
        for record in records:
            timestamp = re.search(r'@(\d{2}\.\d{2}\.\d{2})', record['clip']).group(1).replace('.', ':')
            coahing_json = json.dumps(record['coaching'])
            rec = {"session_id": self.session_id, "event_type": event, "timestamp": timestamp, "data": coahing_json}
            records_to_insert.append(rec)
            counter += 1
            if counter >= 20 and len(records_to_insert) > 0:
                self.sqlitedb.insert_many("Event", records_to_insert)
                counter = 0
                records_to_insert = []

        if len(records_to_insert) > 0:
            self.sqlitedb.insert_many("Event", records_to_insert)       
            del records_to_insert


    def retrieve_session_results(self):
        records = self.sqlitedb.find("Event", {"session_id": self.session_id})
        text_length = 0
        results = ""
        for record in records:
            if text_length > 1000:
                break

            rec_val = record['data'] + "\n\n"
            results += rec_val
            text_length += len(results)

        return results


    def add_summary_to_db(self, summary: str):
        record = {"session_id": self.session_id, "game_name": self.game_name, "summary": summary}
        self.sqlitedb.insert("Session", record)
