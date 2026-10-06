from utils import get_duration, move_clips_to_folder, is_mp4_valid, report_progress

import os, subprocess
from kill_events_process import KillEventsProcessor


class Montage:
    """Compiles all clips within a folder into 1 clip with simple edit and converts a video from horizontal aspect to vertical"""

    def __init__(self, output_dir, montage_len, events_config, ffmpeg_path, vertical_format):
        self.output_dir = output_dir
        self.montage_length_sec = montage_len
        self.events_config = events_config
        self.ffmpeg_path = ffmpeg_path
        self.vertical_format = vertical_format
        
        self.needed_montages = []
        for event, val in self.events_config.items():
            if val.get("clip_eligible"):
                self.needed_montages.append(event)
        
        self.percentages = set(range(1, 101, 1))

        
    def make_compilation(
        self,
        input_folder: str,
        output_file: str,
        fade_duration: float = 0.5
    ):
        print("Creating Montage...\n")

        clips = sorted([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(".mp4")
        ])

        if not clips:
            print("No clips found.")
            return

        chunk_size = 15
        temp_outputs = []

        for idx in range(0, len(clips), chunk_size):
            chunk = clips[idx:idx + chunk_size]

            print(f"Processing chunk {idx // chunk_size + 1}...")

            input_args = []
            filter_parts = []
            pairs = ""
            valid_count = 0

            for clip in chunk:
                path = os.path.join(input_folder, clip)

                # Check clip before adding it to FFmpeg
                if not is_mp4_valid(self.ffmpeg_path, path):
                    print(f"Skipping corrupted/invalid clip: {clip}")
                    continue

                i = valid_count
                valid_count += 1

                duration = max(0.1, get_duration(path))
                fade_out = max(0, duration - fade_duration)

                input_args += ["-i", path]

                filter_parts.append(
                    f"[{i}:v]"
                    f"fade=t=in:st=0:d={fade_duration},"
                    f"fade=t=out:st={fade_out}:d={fade_duration}"
                    f"[v{i}]"
                )

                filter_parts.append(
                    f"[{i}:a]"
                    f"afade=t=in:st=0:d={fade_duration},"
                    f"afade=t=out:st={fade_out}:d={fade_duration}"
                    f"[a{i}]"
                )

                pairs += f"[v{i}][a{i}]"

            # Nothing valid in this chunk
            if valid_count == 0:
                print("No valid clips in this chunk. Skipping...")
                continue

            # Concatenate only valid clips
            filter_parts.append(
                f"{pairs}concat=n={valid_count}:v=1:a=1[v][a]"
            )

            filter_complex = ";".join(filter_parts)

            temp_output = os.path.join(
                input_folder,
                f"_temp_{idx}.mp4"
            )

            temp_outputs.append(temp_output)

            cmd = [
                self.ffmpeg_path,
                *input_args,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                temp_output
            ]

            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                print("\n========== FFMPEG ERROR ==========")
                print(result.stderr)
                print("==================================\n")

                raise RuntimeError(
                    f"FFmpeg failed with code {result.returncode}"
                )

        # No valid chunks were created
        if not temp_outputs:
            print("No valid MP4 clips found.")
            return

        # Merge
        print("Merging...")

        if len(temp_outputs) == 1:
            # Only one chunk → move it to final output
            os.replace(temp_outputs[0], output_file)

            print(
                f"Only one chunk, moved to final output: "
                f"{output_file}"
            )

        else:
            # Normal concat merge
            list_file = os.path.join(input_folder, "merge.txt")

            with open(list_file, "w", encoding="utf-8") as f:
                for t in temp_outputs:
                    abs_path = os.path.abspath(t).replace("\\", "/")
                    f.write(f"file '{abs_path}'\n")

            result = subprocess.run(
                [
                    self.ffmpeg_path,
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-y",
                    output_file
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                print("\n========== FFMPEG MERGE ERROR ==========")
                print(result.stderr)
                print("========================================\n")

                raise RuntimeError(
                    f"FFmpeg merge failed with code {result.returncode}"
                )

        print(f"Done: {output_file}")


    def make_tiktok(self, video_path: str, output_path: str):
        # Crop width and height for center vertical slice
        crop_width = 608
        crop_height = 1080

        # Calculate x and y offsets (expressed as FFmpeg expressions)
        x_offset = "(in_w - {0})/2".format(crop_width)
        y_offset = "(in_h - {0})/2".format(crop_height)

        # FFmpeg command with crop and scale
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-filter:v",
            f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale=1080:1920,setsar=1",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-y",  # Overwrite output if exists
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created vertical TikTok video: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")


    def _create_montage(self):
        total_montages_finished = 0
        
        if "Kill" in self.events_config:
            self.kill_proc = KillEventsProcessor(self.output_dir)
            best_kill_clips = self.kill_proc.find_best_kills()
            new_folder = ''.join((self.output_dir, '/best_kill_clips'))
            os.makedirs(new_folder, exist_ok=True)
            move_clips_to_folder(best_kill_clips, self.montage_length_sec, self.output_dir, new_folder)

        for dir in os.listdir(self.output_dir):
            if os.path.isdir(os.path.join(self.output_dir, dir)) and dir not in ("Kill","KillStreak"):
                clips = []
                for clip in os.listdir(os.path.join(self.output_dir, dir)):
                    if clip.endswith("mp4"):
                        clips.append(clip)
                new_folder = ''.join((self.output_dir, f"/{dir}_compilation_clips"))
                os.makedirs(new_folder, exist_ok=True)
                move_clips_to_folder(clips, self.montage_length_sec, self.output_dir, new_folder)
                self.make_compilation(os.path.join(self.output_dir, f"{dir}_compilation_clips"),
                                            os.path.join(self.output_dir, f"{dir}_highlight_reel.mp4"))
                total_montages_finished+=1
                report_progress(self.output_dir, total_montages_finished, len(self.needed_montages), self.percentages, "Creating Montages ...")

                if not self.vertical_format:
                    self.make_tiktok(os.path.join(self.output_dir, f"{dir}_highlight_reel.mp4"),
                                        os.path.join(self.output_dir, f"{dir}_highlight_reel_tiktok.mp4"))
