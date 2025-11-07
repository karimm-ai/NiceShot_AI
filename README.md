# **NiceShot AI: Python Computer Vision Tool**

NiceShot AI is a powerful and easy-to-use Python tool powered by computer vision to analyze gameplay videos. With the integration of cutting-edge libraries like YOLO, OpenCV, and FFmpeg, NiceShot AI is designed to automatically detect, track and clip key gameplay events from Call of Duty: Black Ops 6.

---

## **Key Events Detected by the Model**

- **Kill**
- **Death**: When a player gets killed during gameplay.
- **Medal**: When a medal earned by the player pops up during gameplay.

---

## **Tool Features**

- **Event Clipping**: Clipping kill & death events from the gameplay video.
- **Clipping Kill Streaks**: Clipping multiple consecutive kills during gameplay by concatenating all unique kills detections within time threshold between each
                             detection and the following.
- **Extracting Best Kill Clips**: Extracting hot kill clips where multiple medals pop up during the event.
- **Videos Export in 16:9 & TikTok formats**
- **Generating Kills Highlight Reel**: Concatenating best/all extracted kill clips into one video with simple fade in & out transitions between clips in both
                                       vertical & horizontal formats.

---

## **Installation**

To get started with **NiceShot_AI**, you need to download & install ffmpeg from the official website: https://www.ffmpeg.org/download.html first.


### **Second: Install the Dependencies**
First, create a Python virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **Sample Run**
niceshot = NiceShot_AI(video_input='cod_gameplay.mp4')
