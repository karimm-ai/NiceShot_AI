from dataclasses import dataclass
import time


@dataclass(init=False)
class Event:
    """Attributes for an event"""

    type: str
    timestart: float
    timeend: float
    video_num: int
    timestamp: str = ""
    desc: str = ""
    kills: int = 0

    def __init__(
        self,
        type: str,
        timestart: float,
        timeend: float,
        video_num: int,
        timestamp: str = "",
        desc: str = "",
        kills: int = 0,

        **kwargs
    ):
        self.type = type
        self.timestart = timestart
        self.timeend = timeend
        self.video_num = video_num
        self.timestamp = timestamp
        self.desc = desc
        self.kills = kills

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self):
        self.timestamp = time.strftime(
            "%H:%M:%S",
            time.gmtime(self.timestart)
        ).replace(":", ".")

        self.desc = f"{self.type}in{self.video_num}@{self.timestamp}.mp4"

    def to_dict(self) -> dict:
        return self.__dict__.copy()