#from gui import HighlightApp
#import tkinter as tk
from niceshot_ai import NiceShot_AI

#root = tk.Tk()
#root.geometry("720x240")
#root.config(bg='black')
#app = HighlightApp(root)
#root.mainloop()

obj = NiceShot_AI('yolov8n-cod_bo6.pt',
                  'ffmpeg-win-x86_64-v7.1.exe',
                  "https://www.twitch.tv/dizzocod",
                  seconds_before_kill=2,
                  seconds_before_death=1,
                  total_hours=0.5,
                  save_clips=False,
                  add_to_csv=True,
                  output_dir='Clients',
                  csv_file='client1_0.csv',
                  frames_to_skip=8,
                  frame_idx_start=0,
                  create_montage=True,
                  max_workers=4,
                  max_videos=3,
                  montage_length_sec=50)

obj.detect_events()