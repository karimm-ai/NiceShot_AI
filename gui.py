import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from niceshot_ai import NiceShot_AI

class HighlightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CoD Clips Extractor")

        style = ttk.Style(self.root)
        style.theme_use('default')
        style.configure("orange.Horizontal.TProgressbar", troughcolor='white', background='orange')

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        tk.Label(root, bg="Black", fg='Orange', text="Select Gameplay Video:",  font=("Times New Roman", 16)).grid(row=2, column=0, padx=5, pady=5)
        tk.Entry(root, textvariable=self.video_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(root, bg="Black", fg="Orange", text="Browse", command=self.browse_video, font=("Times New Roman", 11)).grid(row=2, column=2, padx=5, pady=5)

        tk.Label(root, bg="Black", fg='Orange', text="Select Output Folder:",  font=("Times New Roman", 16)).grid(row=4, column=0, padx=5, pady=5)
        tk.Entry(root, textvariable=self.output_dir, width=50).grid(row=4, column=1, padx=5, pady=5)
        tk.Button(root, bg="Black", fg="Orange", text="Browse", command=self.browse_output, font=("Times New Roman", 11)).grid(row=4, column=2, padx=5, pady=5)

        self.generate_button = tk.Button(root, bg="Black", fg="Orange", text="Extract Clips", command=self.generate, font=("Times New Roman", 11))
        self.generate_button.grid(row=6, column=1, padx=5, pady=5)


    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov, *.webm, *.mkv")])
        if path:
            self.video_path.set(path)


    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)


    def generate(self):
        if not self.video_path.get() or not self.output_dir.get():
            messagebox.showwarning("Missing Info", "Please select both video and output folder.")
            return
        
        tk.Label(self.root, bg="Black", fg='Orange', text="Progress",  font=("Times New Roman", 11)).grid(row=20, column=0, padx=5, pady=5)
        self.progress = ttk.Progressbar(self.root, style="orange.Horizontal.TProgressbar", orient='horizontal', length=300, mode='determinate')
        self.progress.grid(row=20, column=1, pady=10)
        self.generate_button.config(state="disabled")
        self.root.config(cursor="wait")
        self.root.update()
        self.progress['value'] = 0

        niceshot = NiceShot_AI(frame_idx_start=71850, model_path='yolov8n-cod_bo6.pt',
                               ffmpeg_path='ffmpeg-win-x86_64-v7.1.exe',
                               video_path=self.video_path.get(),
                               total_hours=0.33333333333333333, save_clips=False, add_to_csv=False, count_deaths=False, create_montage=False)
        
        niceshot.detect_kills_and_generate_clips(progress_bar=self.progress)
        messagebox.showinfo("Done", "Highlights Extracted!")

        self.generate_button.config(state="normal")
        self.root.config(cursor="")