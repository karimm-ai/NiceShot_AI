import numpy as np
from dataclasses import dataclass


@dataclass
class Track:
    track_id: int

    def __repr__(self) -> str:
        return f"<Track id={self.track_id} at {hex(id(self))}>"


class DeathTracker:
    def __init__(self, max_age: int = 10):
        self.max_age = max_age
        self.current_age = max_age
        self.available_id = 1
        self.tracks = []


    def update_tracks(self, detections: list, frame: np.ndarray) -> list:
        if detections:
            if self.current_age >= self.max_age:
                self.tracks.append(Track(self.available_id))
                self.available_id += 1
                self.current_age = 0
            else:
                self.current_age += 1
        else:
            self.current_age += 1

        return self.tracks
