import easyocr
import numpy as np


class EventConfirm:

    def __init__(self):
        self.ocr = easyocr.Reader(['en'], gpu=True)


    def extract_text(self, frame: np.ndarray) -> str:
        texts = []

        detected_text_regions = self.ocr.detect(frame)[0]
        for box in detected_text_regions:
            text = self.ocr.recognize(frame, box, [], detail=0)
            texts.extend(text)
        return ''.join(texts)


    def is_invalid_event(self, frame: np.ndarray) -> bool:
        frame = self.crop_frame(frame)
        killcam_text = self.extract_text(frame)

        off_words = ("KILLCAM", "KILLGAM", "BESTPLAY", "SPECTATING:", "FINAL KILL", "BEST PLAY", "SPECTATING")
        for word in off_words:
            if word.lower() in killcam_text.lower():
                return True
        return False
    

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        top_part = frame[:int(0.3*h), :]
        bottom_part = frame[int(0.7*h):, :]

        side = int(0.25*w)

        top_part = top_part[:, side:w-side]
        bottom_part = bottom_part[:, side:w-side]

        result = np.vstack((top_part, bottom_part))
        return result
