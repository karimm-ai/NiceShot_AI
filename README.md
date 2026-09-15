## **NiceShot_AI: The analytics layer games do not add**
### Lightweight tool - Runs locally - No cloud queues - No sign ups - Footage stays private

NiceShot AI is a Python tool powered by computer vision to analyze gameplay videos. With the integration of cutting-edge tools like YOLO, OpenCV, FFmpeg, and matplotlib, NiceShot AI is designed to automatically detect, track and clip key gameplay events, create visual report for session stats as well as analyzes negative events (ex.Deaths) providing lightweight scene understanding & coaching to the player.

Simple demo showcasing tool results: (https://youtu.be/op1GDREXiOg)

<br>

<p align="center">
</p>

<p align="center">
  <a href="https://niceshot-ai.itch.io/niceshot-ai">
    <img src="https://img.shields.io/badge/Download%20Tool-itch.io-FA5C5C?style=for-the-badge&logo=itchdotio&logoColor=yellow" alt="Download Tool">
  </a>
</p>

<br>

### **Supported Games**

- **Call of Duty: Black Ops 7 (2025)**

| Key events |                        Description                            |          Limitations        |    Extra Context     |
|------------|---------------------------------------------------------------|-----------------------------|------------------------------|
|    Kill    |  Gun kills   |  Only face-to-face gun kills|  Weapon Name + Kill location in map (Still in testing) |
|    Medal   |  When a medal earned by the player pops up during gameplay    | Medal type not detected, only count | - |
|    Death   |  When player gets eliminated during gameplay  |  - | - |

- **Call of Duty: Black Ops 6 (2024)**

| Key events |                        Description                            |          Limitations        |    Extra Context     |
|------------|---------------------------------------------------------------|-----------------------------|------------------------------|
|    Kill    |  Gun kills   |  Only face-to-face gun kills|  Weapon Name + Kill location in map (Still in testing) |
|    Medal   |  When a medal earned by the player pops up during gameplay    | Medal type not detected, only count | - |
|    Death   |  When player gets eliminated during gameplay  |  - | - |

- **Call of Duty: Modern Warfare II (2022) --> Still in testing**

| Key events |                        Description                            |          Limitations        |
|------------|---------------------------------------------------------------|-----------------------------
|    Kill    |  Gun kills                                  |  -  |
|    Death   |  When player gets eliminated during gameplay                |                -              |

---

### **Models Description**

**Event Detector**: YOLOv8n & YOLO11n by [Ultralytics](https://github.com/ultralytics/ultralytics)
. Fine-tuned on a custom-collected & annotated dataset of gameplay videos under a CC license.

**Event Scene Understanding & Coaching**: Qwen2.5-VL-3B-Instruct by [Qwen](https://github.com/QwenLM-corp/Qwen2.5-VL)
. Used as the vision-language model (VLM) for analyzing short gameplay clips and generating coaching insights based on detected events and visual context.

---

### **Tool Features**

- **Robust to Scalability**: Uses configurable variables enabling it to adapt to different game models and events without massive changes. (In testing)
- **Accurate Event Confirmation**: Using EasyOCR to prevent counting events occurring in special game scenes (ex.KILLCAMS and SPECTATING).
- **Special Events Detection**: (Ex. Kill Streaks occurring from the combination of multiple consecutive kills within a time threshold).
- **Events Timestamping & CSV Output**: Timestamps detected events and dumps into a CSV file with 2 columns [Timestamp, Event] for further gameplay data analysis and inspections.
- **Session Analysis**: Creates a summary report consisting of multiple charts providing a post-session stats analysis.

![Report Screenshot](sample_report.png)

- **Lightweight Coaching**: Analyzes as much short negative clips (ex.Deaths) as needed, understanding what happened and providing coaching and better play suggestions for each single clip. Outputs to "coaching.jsonl" file.

|Type|% of negative clips analyzed|
|-------|------------------------|
|'basic'|25%|
|'short|50%|
|'long'|75%|
|'very long'| 100%|

---

### **Extra Features**

- **Event Auto-Clipping**: Clipping detected events using event's start time and end time.
- **Clips Export in 16:9 & TikTok formats**
- **Creating Highlight Compilations**: Concatenating all clips within a folder into one compilation video with simple fade in & out transition edits between clips in both vertical & horizontal formats.
- **Custom Compilation Lengths**: Allowing for creating compilations of any length from the extracted clips.
- **Analyzing videos in bulk from a Twitch channel**: Downloads and analyzes desired game streams from a Twitch channel performing bulk analysis of gameplay videos. (In testing)
- **Ranking Special Clips**: Ex. (Hot Kill Clips where multiple medals pop up during the event).

---

### **Limitations**

- **Event detection is not perfect**: From my testing, an event can get detected more than once or not detected at all.

---

### **Installation**

To get started with **NiceShot_AI**, clone the repository first.

#### **Second: Install the Dependencies**
Create a Python virtual environment (optional, but recommended). My Python version is 3.10.11

```bash
python -m venv venv
venv\Scripts\activate
```

Install torch cuda. I used cuda 12.1 for GTX1650 4GB. Currently, I am using nightly cuda 12.8 for RTX5070 8GB:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### **Run the tool**
```
from detector import EventDetector

detector = EventDetector("Call of Duty: Black Ops 7", # Name of the game
            "A:/2.mp4", # Gameplay video path
            total_hours=0.5, # Total hours to analyze of the video
            save_clips=False, # Save clips locally
            add_to_csv=True, # Add events and timestamps to a CSV file
            output_dir='test17', # Output directory where all clips, highlights and CSV file are saved
            frames_to_skip=8, # Frames to skip during analysis (The more, the faster the analysis is finished)
            frame_idx_start=0, # Starting frame
            create_montage=False, # Create a highlight reel for clipped events
            max_workers=2, # Default for extracting clips
            max_videos=3, # Only useful if passing a Twitch channel as it gets the most recent specified number of videos
            montage_length_sec=120, # Total duration of the generated highlight reel in seconds
            vertical_format=False, # Auto-clip in vertical format
            advanced_detection=True, # Use OCR to confirm some events & context surrounding an event 
            session_analysis=True, # Create stats summary charts in a report
            coaching='basic' # Analyzes short negative clips using VLM to understand what happened & provide coaching tips
            )

detector.detect_events()
```

---

### **Processing Speed**
(Note: Coaching processing speed results is still not included)

Tested on laptop_1 with the following specs:
- **CPU**: core i9 14th gen
- **GPU**: RTX5070 8GB
- **RAM**: 32GB

Tested on laptop_2 with the following specs:
- **CPU**: core i7 10th gen
- **GPU**: GTX1650 4GB
- **RAM**: 16GB


|    Device   | Frame Inference |
|-------------|-----------------|
| laptop_1  |  Up to 170 FPS with frames_to_skip = 5|
| laptop_2  |  Up to 60 FPS with frames_to_skip = 5|


#### **Advanced Detection with OCR**

This is run only to:

- Confirm an event after it's detected. Not through the whole video frames. It can cause the processing speed to fall down from 170 FPS to 30 FPS (laptop_2) temporarily until event is confirmed. It can definitely be turned off, however this will cause a kill event during "SPECTATING" to be counted.

- Grab context surrounding important events (ex. Weapon used in a Kill event).

---

### **Minimum System Requirements**

- **Operating System**: Windows 10/11
- **GPU**: NVidia GTX1650 4GB VRAM - No support yet for AMD or Intel GPUs
- **RAM**: 16GB
- **Storage**: 8GB HDD

Note: The minimum specifications are what I've tested on. Performance depends on game resolution, system specifications and count of events in a gameplay video.

---

### **GUI**
(Coaching is still not a part of the GUI)

- The tool provides a simple graphical user interface found in "src/niceshot_ai/NiceShot AI.exe".

---