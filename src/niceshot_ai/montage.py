from .utils import get_duration
import os, subprocess


class Montage:
    """Compiles all clips within a folder into 1 clip with simple edit and converts a video from horizontal aspect to vertical"""

    def __init__(self,):
        pass
   

    def make_compilation(self, input_folder, output_file, fade_duration=0.5):
        print("Creating Montage...\n")
        clips = sorted([f for f in os.listdir(input_folder) if f.endswith('.mp4')])
        if not clips:
            print("❌ No clips found.")
            return

        input_args = []
        durations = []
        for clip in clips:
            path = os.path.join(input_folder, clip)
            input_args.extend(["-i", path])
            durations.append(get_duration(path))

        filter_parts = []
        v_streams = []
        a_streams = []

        for i, duration in enumerate(durations):
            fade_out_start = max(0, duration - fade_duration)
            filter_parts.append(
                f"[{i}:v]fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration},setpts=PTS-STARTPTS[v{i}];"
            )
            filter_parts.append(
                f"[{i}:a]afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration},asetpts=PTS-STARTPTS[a{i}];"
            )
            v_streams.append(f"[v{i}]")
            a_streams.append(f"[a{i}]")

        filter_parts.append(f"{''.join(v_streams)}concat=n={len(clips)}:v=1:a=0[v];")
        filter_parts.append(f"{''.join(a_streams)}concat=n={len(clips)}:v=0:a=1[a]")

        filter_complex = "".join(filter_parts)

        cmd = ["ffmpeg"]
        cmd.extend(input_args)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-vsync", "2",
            "-async", "1",
            "-y",
            output_file
        ])

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Montage with fade transitions created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")


    def make_tiktok(self, video_path, output_path):
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
