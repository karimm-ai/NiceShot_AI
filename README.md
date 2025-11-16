## **NiceShot_AI: Python Computer Vision Tool**

NiceShot AI is a Python tool powered by computer vision to analyze gameplay videos. With the integration of cutting-edge tools like YOLO, OpenCV, and FFmpeg, NiceShot AI is designed to automatically detect, track and clip key gameplay events from Call of Duty: Black Ops 6.

Simple demo showcasing tool results: (https://youtu.be/nFs7VJxT-Ig)

---

### **Key Events Detected by the Model**

- **Kill**
- **Death**: When a player gets killed during gameplay.
- **Medal**: When a medal earned by the player pops up during gameplay.

---

### **Model Description**

YOLOv8n by [Ultralytics](https://github.com/ultralytics/ultralytics). Fine-tuned on custom collected & annotated dataset of gameplay videos for Call of Duty: Black Ops 6. Model detects kill, death and medal pop up events with good confidence. Model is expected to generalize well on modern Call of Duty games as there core gameplay does not vastly change. Model is also tested on one Call of Duty: Black Ops 7 gameplay video providing initial very good detection results.

---

### **Tool Features**

- **Event Clipping**: Clipping kill & death events from the gameplay video.
- **Clipping Kill Streaks**: Clipping multiple consecutive kills during gameplay by concatenating all unique kills detections within time threshold between each detection and the following.
- **Extracting Best Kill Clips**: Extracting hot kill clips where multiple medals pop up during the event.
- **Clips Export in 16:9 & TikTok formats**
- **Generating Kills Highlight Reel**: Concatenating best or all extracted kill clips into one video with simple fade in & out transitions between clips in both vertical & horizontal formats.
- **Analyzing BO6 videos in bulk from a Twitch channel**: Downloads and analyzes CoD BO6 streams from a Twitch channel performing bulk analysis of gameplay videos.
- **Events Timestamping & CSV Output**: Timestamps detected events and dumps into a CSV file with 2 columns [Timestamp, Event] for further gameplay data analysis and inspections.

---

### **Advanced Tool Features**

- **Accurate Event Clipping**: Using RapidOCR to avoid counting frames of KILLCAMS and SPECTATING.
- **Custom Montage Lengths**: Allowing for creating compilations of any length.

---

### **Installation**

To get started with **NiceShot_AI**, you need to download & install ffmpeg from the official website: https://www.ffmpeg.org/download.html first and add it to your PATH.

#### **Second: Install the Dependencies**
Create a Python virtual environment (optional, but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

Install torch cuda. What worked for me is cuda 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### **Run the tool**
```
from niceshot_ai import NiceShot_AI

niceshot = NiceShot_AI('yolov8n-cod_bo6.pt', # YOLO Model Path
                        'ffmpeg-win-x86_64-v7.1.exe', # FFMPEG Path (You can find it where you installed FFMPEG. Can also be 'ffmpeg.exe')
                        "https://www.twitch.tv/your_channel", # CoD BO6 gameplay video or Twitch Channel Link
                        seconds_before_kill=2, # Seconds before kill to include when a kill event is detected (for clipping)
                        seconds_before_death=1, # Seconds before death to include when a death event is detected (for clipping)
                        total_hours=0.5, # Total hours to analyze of the video (each video in case analyzing a Twitch Channel)
                        save_clips=True, # Save clips locally
                        add_to_csv=True, # Add events and timestamps to a CSV file.
                        output_dir='outputs', # Output directory where all clips, highlights and CSV file are saved
                        csv_file='timestamps.csv', # CSV file path
                        frames_to_skip=8, # Frames to skip during analysis (The more, the faster the analysis is finished)
                        frame_idx_start=0, # Starting frame
                        create_montage=True, # Create a highlight reel of kills
                        max_workers=4, # Default
                        max_videos=3, # Only useful if passing a Twitch channel as it gets the most recent specified number of videos
                        montage_length_sec=50) # Total duration of the generated highlight reel in seconds

niceshot.detect_events()
```