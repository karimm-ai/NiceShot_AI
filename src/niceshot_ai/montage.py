from .utils import get_duration

import os, subprocess


class Montage:
    """Compiles all clips within a folder into 1 clip with simple edit and converts a video from horizontal aspect to vertical"""

    def __init__(self,):
        pass
   

    def make_compilation(self, input_folder: str, output_file: str, fade_duration: float = 0.5):
        print("🎬 Creating Montage...\n")

        clips = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp4")])
        if not clips:
            print("❌ No clips found.")
            return

        chunk_size = 15
        temp_outputs = []

        for idx in range(0, len(clips), chunk_size):
            chunk = clips[idx:idx + chunk_size]
            print(f"⚙️ Processing chunk {idx // chunk_size + 1}...")

            input_args = []
            filter_parts = []
            pairs = ""

            for i, clip in enumerate(chunk):
                path = os.path.join(input_folder, clip)
                duration = max(0.1, get_duration(path))
                fade_out = max(0, duration - fade_duration)

                input_args += ["-i", path]

                filter_parts.append(
                    f"[{i}:v]fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out}:d={fade_duration}[v{i}]"
                )
                filter_parts.append(
                    f"[{i}:a]afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_out}:d={fade_duration}[a{i}]"
                )

                pairs += f"[v{i}][a{i}]"

            filter_parts.append(f"{pairs}concat=n={len(chunk)}:v=1:a=1[v][a]")

            filter_complex = ";".join(filter_parts)

            temp_output = os.path.join(input_folder, f"_temp_{idx}.mp4")
            temp_outputs.append(temp_output)

            cmd = [
                "ffmpeg",
                *input_args,
                "-filter_complex", filter_complex,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                temp_output
            ]

            subprocess.run(cmd, check=True)

        # Merge
        print("🔗 Merging...")

        if len(temp_outputs) == 1:
            # Only one chunk → just rename it to the final output
            os.replace(temp_outputs[0], output_file)
            print(f"✅ Only one chunk, moved to final output: {output_file}")
        else:
            # Normal concat merge
            list_file = os.path.join(input_folder, "merge.txt")
            with open(list_file, "w") as f:
                for t in temp_outputs:
                    f.write(f"file '{t}'\n")

            subprocess.run([
                "ffmpeg",
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
            ], check=True)

        print(f"✅ Done: {output_file}")


    def make_tiktok(self, video_path: str, output_path: str):
        # Crop width and height for center vertical slice
        crop_width = 608
        crop_height = 1080

        # Calculate x and y offsets (expressed as FFmpeg expressions)
        x_offset = "(in_w - {0})/2".format(crop_width)
        y_offset = "(in_h - {0})/2".format(crop_height)

        # FFmpeg command with crop and scale
        cmd = [
            "ffmpeg",
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
            print(f"✅ Successfully created vertical TikTok video: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
