import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import os, json, sys
from tkinter import font, ttk
import subprocess
from pathlib import Path

from updater import check_and_update


class GUI:
    def __init__(self, root):
        should_restart = check_and_update()
        if should_restart:
            subprocess.Popen([
                sys.executable,
                os.path.join(Path(sys.executable).resolve().parent, "NiceShot AI.exe")
            ])

            sys.exit(0)

        self.root = root
        self.root.title("NiceShot AI")
        self.root.geometry("500x450")
        self.root.configure(bg="#dcb561")
        icon_path = Path(sys.executable).resolve().parent / "icon.ico"
        self.root.iconbitmap(icon_path)
        
        bold_font = font.Font(size=9, family='anonymous pro')
        style = ttk.Style()

        style.configure(
            "Custom.TCombobox",
            fieldbackground="yellow",  # background inside box
            background="yellow",             # button area
            foreground="black"
        )

        games = ["Call of Duty: Black Ops 6", "Call of Duty: Black Ops 7"]

        game_frame = tk.Frame(root, bg="#dcb561")
        game_frame.pack(fill="x", padx=20, anchor="w", pady=20)
        tk.Label(game_frame, text="Game:", bg="#dcb561", font=bold_font, fg="black").pack(side="left", padx=(0,10))
        self.combo = ttk.Combobox(game_frame, values=games, state="readonly", width=40, style="Custom.TCombobox")
        self.combo.current(0)
        self.combo.pack(side="left")
        
        tk.Label(root, text="Input Video:", bg="#dcb561", font=bold_font, fg="black").pack(anchor="w", padx=20, pady=(10,0))
        input_frame = tk.Frame(root, bg="#dcb561")
        input_frame.pack(fill="x", padx=20, anchor="w")
        self.input_entry = tk.Entry(input_frame)
        self.input_entry.pack(side="left", fill="x", expand=True)
        tk.Button(input_frame, text="Browse", command=self.browse_input, bg="#dcb561", fg="black").pack(side="left", padx=5)
        
        # --- Output Folder ---
        tk.Label(root, text="Output Folder:", bg="#dcb561", font=bold_font, fg="black").pack(anchor="w", padx=20, pady=(10,0))
        output_frame = tk.Frame(root, bg="#dcb561")
        output_frame.pack(fill="x", padx=20, anchor="w")
        self.output_entry = tk.Entry(output_frame)
        self.output_entry.pack(side="left", fill="x", expand=True)
        tk.Button(output_frame, text="Browse", command=self.browse_output, bg="#dcb561", fg="black").pack(side="left", padx=5)

        self.save_clips = tk.BooleanVar()
        self.create_compilation = tk.BooleanVar()
        self.vertical_format = tk.BooleanVar()
        self.analysis = tk.BooleanVar()
        bg_color = "#dcb561"   # deep violet
        fg_color = "black"

        # Container frame (keeps everything aligned nicely)
        frame = tk.Frame(root, bg="#dcb561", padx=20, pady=10)
        frame.pack(fill="x", anchor="w")  # align whole block to the left

        # Make columns expand evenly
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Checkbox style
        cb_options = {
            "bg": bg_color,
            "fg": fg_color,
            "activebackground": bg_color,
            "activeforeground": fg_color,
            "selectcolor": "white",
            "justify": "left"
        }

        # Checkboxes placed in grid (2 per row)
        cb1 = tk.Checkbutton(frame, text="Save Clips", variable=self.save_clips, **cb_options, font=bold_font)
        cb2 = tk.Checkbutton(frame, text="Create Compilation", variable=self.create_compilation, **cb_options, font=bold_font)
        cb3 = tk.Checkbutton(frame, text="Save clips in vertical format", variable=self.vertical_format, **cb_options, font=bold_font)
        cb4 = tk.Checkbutton(frame, text="Create Session Analysis Report", variable=self.analysis, **cb_options, font=bold_font)

        cb1.grid(row=0, column=0, padx=0, pady=3, sticky="w")
        cb2.grid(row=0, column=1, padx=0, pady=3, sticky="w")
        cb3.grid(row=1, column=0, padx=0, pady=3, sticky="w")
        cb4.grid(row=1, column=1, padx=0, pady=3, sticky="w")

        comp_frame = tk.Frame(root, bg="#dcb561")
        comp_frame.pack(fill="x", padx=20, anchor="w")
        tk.Label(comp_frame, text="Length of compilation (minutes):", bg="#dcb561", font=bold_font, fg="black").pack(side="left", padx=(0,10))

        def validate(value):
            return value.isdigit() or value == ""
        vcmd = (self.root.register(validate), "%P")
        self.spin = tk.Spinbox(
            comp_frame,
            from_=0,
            to=100,
            validate="key",
            validatecommand=vcmd,
        )
        self.spin.pack(anchor="w", padx=20, pady=25)

        # --- Analyze Button ---
        self.analyze_btn = tk.Button(root, text="Analyze gameplay", command=self.analyze_video, bg="#dcb561", fg="black", font=bold_font)
        self.analyze_btn.pack(pady=10)
        # --- Progress Bar ---
        progress_bar_style = ttk.Style()
        progress_bar_style.theme_use('default')
        progress_bar_style.configure(
            "gray.Horizontal.TProgressbar",
            troughcolor='white',   # background/trough
            background='#4e4e4e'     # progress fill
        )

        self.progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate', style="gray.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=0)
        self.progress_text = tk.StringVar()
        self.progress_text.set("Waiting...")
        self.progress_label = tk.Label(root, textvariable=self.progress_text, font=bold_font)
        self.progress_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    def on_close(self):
        if hasattr(self, "process") and self.process is not None:
            if self.process.poll() is None:  # still running
                print("Stopping subprocess...")

                self.process.terminate()  # try graceful stop

                try:
                    self.process.wait(timeout=3)
                except:
                    self.process.kill()  # force kill if needed

        self.root.destroy()


    # --- Methods ---
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select gameplay video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")]
        )
        if filename:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, filename)
    
    def browse_output(self):
        foldername = filedialog.askdirectory(title="Select output folder")
        if foldername:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, foldername)
    

    def update_progress(self):
        progress_file = f"{self.output_entry.get()}/progress.json"

        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)

                    progress = data.get("PROGRESS", 0)
                    msg = data.get("MSG", "")

                    self.progress_bar["value"] = progress
                    self.progress_text.set(f"{msg} {progress}%")

            except Exception as e:
                print("Error reading progress:", e)

        # keep updating while process runs
        if self.process.poll() is None:
            self.root.after(500, self.update_progress)
        else:
            self.progress_bar["value"] = 100
            self.progress_text.set("Done ✔")
            messagebox.showinfo(
                "Done",
                f"Analysis complete! Results saved to:\n{self.output_entry.get()}"
            )
            self.analyze_btn.config(state="normal")
            
            temp_files = ("status.json", "progress.json", "events_temp.json", "events_temp_2.json", "video1.csv", "timestamp_sorted.csv")
            for file in temp_files:
                try:
                    os.remove(f"{self.output_entry.get()}/{file}")
                except:
                    pass

    def analyze_video(self):
        self.analyze_btn.config(state="disabled")
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()
        save_clips = self.save_clips.get()
        create_compilation = self.create_compilation.get()
        vert = self.vertical_format.get()
        analysis = self.analysis.get()
        choosen_game = self.combo.get()
        montage_len_seconds = int(self.spin.get())*60

        print(save_clips, create_compilation, vert, analysis, choosen_game, montage_len_seconds)
        
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid input video")
            self.analyze_btn.config(state="normal")
            return
        if not output_path or not os.path.isdir(output_path):
            messagebox.showerror("Error", "Please select a valid output folder")
            self.analyze_btn.config(state="normal")
            return 
        if not save_clips and not create_compilation and not analysis:
            messagebox.showerror("Error", "Please tick a relevant checkbox")
            self.analyze_btn.config(state="normal")
            return
        
        # Simulate progress; replace with your actual YOLO + analysis code
        self.progress_bar["value"] = 0
        self.root.update_idletasks()
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path(__file__).resolve().parent

        # Go up levels like your original logic
        root_dir = base_dir.parent.parent.parent

        python_file = root_dir / ".venv" / "Scripts" / "python.exe"
        cli_file = base_dir / "niceshot_ai.py"
        #messagebox.showinfo("HI", f"{python_file}, {cli_file}")

        args = [
            python_file,
            cli_file,
            "--game", choosen_game,
            "--input", input_path,
            "--output", output_path,
            "--comp_len", str(montage_len_seconds)
        ]

        if vert:
            args.append("--vertical_format")

        if analysis:
            args.append("--session_analysis")

        if save_clips:
            args.append("--save_clips")

        if create_compilation:
            args.append("--compilation")

        self.process = subprocess.Popen(args, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)

        self.update_progress()
        

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()