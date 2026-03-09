import subprocess, os


class Clipper:
    """Clipper class for clipping segments from a video path"""

    def __init__(self, ffmpeg_path: str, vertical_format: bool):
        self.ffmpeg_path = ffmpeg_path
        self.vertical_format = vertical_format
        self.crop_width = 608
        self.crop_height = 1080
        self.x_offset = "(in_w - {0})/2".format(self.crop_width)
        self.y_offset = "(in_h - {0})/2".format(self.crop_height)


    def clip_event(self, output_dir: str, event: dict, video_path: str):
        output_path = os.path.join(output_dir, event['desc'])

        if not self.vertical_format:
            subprocess.run([
            self.ffmpeg_path,
            "-ss", str(event['timestart']),
            "-i", video_path,
            "-to", str(event['timeend'] - event['timestart']),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-loglevel", "error",
            "-y",

            output_path
            ])
        
        else:
            cmd = [
                self.ffmpeg_path,
                "-ss", str(event['timestart']),
                "-i", video_path,
                "-t", str(event['timeend'] - event['timestart']),
                "-filter:v",
                f"crop={self.crop_width}:{self.crop_height}:{self.x_offset}:{self.y_offset},scale=1080:1920,setsar=1",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-movflags", "+faststart",
                "-y",
                output_path
            ]


            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                #print(f"✅ Successfully extracted vertical TikTok video: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg error: {e.stderr.decode()}")