from dataclasses import dataclass
from enum import Enum, auto
import time

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