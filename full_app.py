#from gui import HighlightApp
#import tkinter as tk
from niceshot_ai import NiceShot_AI

#root = tk.Tk()
#root.geometry("720x240")
#root.config(bg='black')
#app = HighlightApp(root)
#root.mainloop()

niceshot = NiceShot_AI('yolov8n-cod_bo6.pt', # YOLO Model Path
                        'ffmpeg-win-x86_64-v7.1.exe', # FFMPEG Path
                        "D:/Clients/Clients/Downloads/1.mp4", # CoD BO6 gameplay video or Twitch Channel Link
                        seconds_before_kill=2, # Seconds before kill to include when a kill event is detected (for clipping)
                        seconds_before_death=1, # Seconds before death to include when a death event is detected (for clipping)
                        total_hours=0.5, # Total hours to analyze of the video (each video in case analyzing a Twitch Channel)
                        save_clips=True, # Save clips locally
                        vertical_format = True, # Export clips in vertical format
                        add_to_csv=True, # Add events and timestamps to a CSV file.
                        output_dir='Clients', # Output directory where all clips, highlights and CSV file are saved
                        csv_file='client1_0.csv', # CSV file path
                        frames_to_skip=8, # Frames to skip during analysis (The more, the faster the analysis is finished)
                        frame_idx_start=28000, # Starting frame
                        create_montage=True, # Create a highlight reel of kills
                        max_workers=4, # Default
                        max_videos=3, # Only useful if passing a Twitch channel as it gets the recent num videos
                        montage_length_sec=50) # Total duration of the generated highlight reel

niceshot.detect_events()