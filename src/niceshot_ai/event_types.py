from dataclasses import dataclass
import time


@dataclass
class Event:
    """Attributes for an event"""

    type: str
    timestart: float
    timeend: float
    video_num: int
    timestamp: str = ""
    desc: str = ""
    kills: int = 0
    
    def __post_init__(self):
        self.timestamp = time.strftime("%H:%M:%S", time.gmtime(self.timestart))
        self.timestamp = self.timestamp.replace(":", ".")
        self.desc = f"{self.type}in{str(self.video_num)}@{self.timestamp}.mp4"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "timestart": self.timestart,
            "timeend": self.timeend,
            "timestamp": self.timestamp,
            "desc": self.desc,
            "kills": self.kills
        }