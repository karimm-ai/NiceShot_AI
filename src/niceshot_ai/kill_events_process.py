from event_types import Event

import json
import csv
from datetime import datetime
from bisect import bisect_left, bisect_right


class KillEventsProcessor:
    """Finds top kill clips and kill streaks"""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir


    def find_best_kills(self) -> dict:   
        
        def timestamp_to_seconds(timestamp):
            t = datetime.strptime(timestamp, "%H:%M:%S")
            return t.hour * 3600 + t.minute * 60 + t.second

        with open(f"{self.output_dir}/events_temp_2.json", "r") as f:
            events = json.load(f)

        csv_timestamps = []

        with open(f"{self.output_dir}/video1.csv", "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                csv_timestamps.append(
                    timestamp_to_seconds(row["Timestamp"])
                )

        csv_timestamps.sort()

        for event in events:
            start = event["timestart"]
            end = event["timeend"]

            left = bisect_left(csv_timestamps, start)
            right = bisect_right(csv_timestamps, end)

            event["medals_count"] = right - left

        events.sort(key=lambda x: x.get("count", 0), reverse=True)

        with open(f"{self.output_dir}/events_temp_2.json", "w") as f:
            json.dump(events, f, indent=4)

        return [event["desc"] for event in events]


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
            if len(streak) > 2:
                for kill in streak:
                    temp_events.append(kill)

        merged = []
        for streak in kill_streaks:
            if len(streak) > 2:
                merged.append(Event(type="KillStreak",
                    timestart=streak[0]["timestart"],
                    timeend=streak[-1]["timeend"],
                    video_num=video_num,
                    kills=len(streak)))

        merged = [event.to_dict() for event in merged]
        for event in events:
            if event not in temp_events:
                merged.append(event)
        del temp_events, events, kill_streaks, current_streak

        with open(f"{self.output_dir}/events_temp_2.json", 'w') as f:
            json.dump(merged, f, indent=2)