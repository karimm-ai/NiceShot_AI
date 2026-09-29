import numpy as np
import math
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



@dataclass
class MedalTrack:
    track_id: int
    bbox: tuple
    missed: int = 0
    hits: int = 1
    locked: bool = False


class MedalTracker:
    def __init__(
        self,
        max_distance=8,
        max_missed=11,
        min_hits=1,
        smoothing=0.7,
        size_tolerance=0.5
    ):

        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.smoothing = smoothing
        self.size_tolerance = size_tolerance
        self.tracks = []
        self.next_id = 0


    def center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    def width_height(self, bbox):
        x1, y1, x2, y2 = bbox
        return (
            x2 - x1,
            y2 - y1
        )

    def distance(self, boxA, boxB):
        ax, ay = self.center(boxA)
        bx, by = self.center(boxB)

        return math.sqrt(
            (ax - bx) ** 2 +
            (ay - by) ** 2
        )

    def size_similarity(self, boxA, boxB):
        aw, ah = self.width_height(boxA)
        bw, bh = self.width_height(boxB)

        if bw == 0 or bh == 0:
            return 0

        w_ratio = min(aw, bw) / max(aw, bw)
        h_ratio = min(ah, bh) / max(ah, bh)

        return (w_ratio + h_ratio) / 2

    def smooth_bbox(self, old_box, new_box):
        alpha = self.smoothing

        return tuple(
            alpha * o + (1 - alpha) * n
            for o, n in zip(old_box, new_box)
        )


    def update_tracks(self, detections, frame):
        used_tracks = set()
        used_dets = set()
        matches = []

        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                bbox = det[0]
                dist = self.distance(track.bbox, bbox)
                if dist > self.max_distance:
                    continue

                size_sim = self.size_similarity(
                    track.bbox,
                    bbox
                )

                if size_sim < self.size_tolerance:
                    continue

                if dist > self.max_distance:
                    continue

                score = size_sim
                matches.append(
                    (
                        score,
                        t_idx,
                        d_idx
                    )
                )

        matches.sort(reverse=True)

        for score, t_idx, d_idx in matches:
            if t_idx in used_tracks:
                continue

            if d_idx in used_dets:
                continue

            track = self.tracks[t_idx]
            det_bbox = detections[d_idx][0]
            track.bbox = self.smooth_bbox(
                track.bbox,
                det_bbox
            )

            track.missed = 0
            track.hits += 1

            if track.hits >= 3:
                track.locked = True

            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        alive_tracks = []

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in used_tracks:
                track.missed += 1

            if track.missed <= self.max_missed:
                alive_tracks.append(track)

        self.tracks = alive_tracks

        for d_idx, det in enumerate(detections):
            if d_idx in used_dets:
                continue

            bbox = det[0]
            duplicate = False

            for track in self.tracks:
                dist = self.distance(
                    track.bbox,
                    bbox
                )
                if dist < self.max_distance * 1.2:
                    duplicate = True
                    break

            if duplicate:
                continue

            too_close = any(
                self.distance(t.bbox, det[0]) < self.max_distance
                for t in self.tracks
            )

            if too_close:
                continue

            self.tracks.append(
                MedalTrack(
                    track_id=self.next_id,
                    bbox=bbox
                )
            )
            self.next_id += 1

        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                results.append({
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "hits": track.hits,
                    "missed": track.missed,
                    "locked": track.locked
                })
        return results