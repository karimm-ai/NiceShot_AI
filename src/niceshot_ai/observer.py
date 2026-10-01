from utils import is_mp4_valid


import cv2
from qwen_vl_utils import process_vision_info
import os, json
import random
from difflib import SequenceMatcher
import re
from PIL import Image
import time
import unicodedata
from pathlib import Path



class Observer:
    def __init__(self, output_dir: str, type: str | None = None):
        self.model, self.processor = self.load_model()
        self.output_dir = output_dir
        self.type = type
        self.sample_size = self.specify_sample()
        self.obs_summary = Observer_Summary()

        
    def load_model(self):
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            device_map="auto",
            quantization_config=bnb,
        )

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        return model, processor


    def infer(self, messages: list, model, processor):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        generated = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        response = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response


    def pre_process_gameplay(self, video_path: str, sample_every: int = 8):
        output_video = f"{self.output_dir}/sampled_clip.mp4"

        target_width = 720
        target_height = 540

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Original FPS: {fps}")
        print(f"Original frames: {total_frames}")

        output_fps = fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_video,
            fourcc,
            output_fps,
            (target_width, target_height)
        )

        frame_id = 0
        written = 0

        last_frame_to_analyze = total_frames - 140

        while True:
            ret, frame = cap.read()

            if not ret or frame_id >= last_frame_to_analyze:
                break

            # Sample frames
            if frame_id % sample_every == 0:
                resized = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA
                )

                out.write(resized)
                written += 1

            frame_id += 1


        cap.release()
        out.release()

        print(f"Done. Written frames: {written}")
        print(f"Saved: {output_video}")


    def analyze_session(self, prompt, folder):
        output_file = f"{folder}.jsonl"
        clips = []

        for clip in os.listdir(folder):
            clips.append(clip)

        k = max(1, int(len(clips) * self.sample_size))
        sample = random.sample(clips, k=k)
        ffmpeg_path = Path(__file__).resolve().parent / "ffmpeg.exe"

        with open(output_file, "a", encoding="utf-8") as f:
            for clip in sample:
                if is_mp4_valid(str(ffmpeg_path), f"{folder}/{clip}"):
                    print(f"Analyzing {clip} ...")
                    self.pre_process_gameplay(f"{folder}/{clip}")
                    time.sleep(1)
                    obs = self.analyze_video(f"{self.output_dir}/sampled_clip.mp4", prompt)
                    video_observations = ""
                    for idx in obs:
                        moment_obs = f"[{idx['timestamp']:.2f}s] {idx['description']}"
                        video_observations = video_observations + moment_obs + "\n"
                    
                    final_observation, _ = self.obs_summary.aggregate_observations(video_observations)

                    record = {
                        "clip": clip,
                        "msg": final_observation
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def specify_sample(self):
        if self.type == 'quick':
            return 0.1

        elif self.type == 'basic':
            return 0.25

        elif self.type == 'long':
            return 0.5

        elif self.type == 'full':
            return 1
        
        else:
            return 0


    def analyze_frame(self, frame, prompt):
        """
        frame: OpenCV BGR frame

        returns:
            VLM description as a string
        """

        # OpenCV -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # RGB numpy -> PIL
        image = Image.fromarray(frame_rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        output = self.infer(messages, self.model, self.processor)
        return output.strip()


    def analyze_video(self, video_path, prompt, sample_fps=1):
        """
        Analyze the video at sample_fps.

        Example:
            sample_fps=2
            -> analyze approximately 2 frames per second
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = total_frames / video_fps
        sample_fps = total_frames

        print(f"Total Frames: {total_frames}")
        print(f"Video FPS: {video_fps}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sampling FPS: {sample_fps}")

        # Number of frames to skip between observations
        frame_interval = max(1, int(video_fps / sample_fps))

        observations = []

        frame_index = 0
        frame_to_stop_at = total_frames

        while True:

            ret, frame = cap.read()

            if not ret or frame_index >= frame_to_stop_at:
                break

            # Only analyze selected frames
            if frame_index % frame_interval == 0:

                timestamp = frame_index / video_fps

                print(f"\nAnalyzing {timestamp:.2f}s...")

                description = self.analyze_frame(frame, prompt)

                observation = {
                    "timestamp": round(timestamp, 2),
                    "description": description
                }

                observations.append(observation)

            frame_index += 1

        cap.release()

        return observations



class Observer_Summary:
    def __init__(self, similarity_threshold: float = 0.8, max_words: int = 900):
        self.similarity_threshold = similarity_threshold
        self.max_words = max_words


    def normalize_text(self, text):
        """
        Normalize VLM output before comparing observations.
        Keeps ASCII letters, numbers, and whitespace.
        Removes Unicode symbols, emojis, control characters,
        corrupted characters, and punctuation.
        """

        if not isinstance(text, str):
            return ""

        # Remove Unicode control/format characters
        text = "".join(
            char for char in text
            if unicodedata.category(char) not in ("Cc", "Cf", "Cs", "Co", "Cn")
        )

        # Normalize Unicode representation
        text = unicodedata.normalize("NFKD", text)

        # Convert accented characters to ASCII where possible
        text = text.encode("ascii", "ignore").decode("ascii")

        # Keep only letters, numbers and whitespace
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def token_similarity(self, text1, text2):
        """
        Jaccard similarity between the words in two observations.

        Example:

            "player is aiming at enemy"
            "player is aiming at enemy near building"

        will have high similarity.
        """

        words1 = set(self.normalize_text(text1).split())
        words2 = set(self.normalize_text(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


    def sequence_similarity(self, text1, text2):
        """
        Character-level similarity.

        Useful for detecting descriptions that are almost identical
        but have a few changed words.
        """

        text1 = self.normalize_text(text1)
        text2 = self.normalize_text(text2)

        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(
            None,
            text1,
            text2,
        ).ratio()


    def observation_similarity(self, text1, text2):
        """
        Combine token and sequence similarity.

        Token similarity is more useful for descriptions where
        a few words are added/removed.

        Sequence similarity is useful when the descriptions are
        almost identical.
        """

        token_score = self.token_similarity(
            text1,
            text2,
        )

        sequence_score = self.sequence_similarity(
            text1,
            text2,
        )

        # Weighted combination
        return (
            0.65 * token_score
            + 0.35 * sequence_score
        )


    def parse_observations(self, raw_text):
        """
        Parse:

            Analyzing 0.00s...
            description...

            Analyzing 0.02s...
            description...
        """

        pattern = re.compile(
        r"""
        \[(\d+(?:\.\d+)?)s\]
        \s*
        (.*?)
        (?=
            \n\s*
            \[\d+(?:\.\d+)?s\]
            |
            \Z
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

        matches = pattern.findall(raw_text)

        observations = []

        for timestamp, text in matches:

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if not text:
                continue

            observations.append({
                "time": float(timestamp),
                "text": text,
            })

        return observations


    def merge_observations(self, obs1, obs2):
        """
        Merge two consecutive observations.

        We keep the more informative/longer description and
        combine the timestamps.
        """

        # Usually the longer description contains more information.
        if len(obs2["text"]) > len(obs1["text"]):
            text = obs2["text"]
        else:
            text = obs1["text"]

        return {
            "start_time": obs1.get(
                "start_time",
                obs1["time"],
            ),
            "end_time": obs2.get(
                "end_time",
                obs2["time"],
            ),
            "text": text,
        }


    def aggregate_observations(
        self,
        raw_text):
        """
        Compare each observation with the next observation.

        If similarity is above the threshold, merge them.

        This is intentionally sequential:

            A -> B
                B -> C
                    C -> D

        rather than comparing every observation with every other
        observation.

        Parameters
        ----------
        raw_text:
            Raw timestamped observations.

        similarity_threshold:
            Recommended starting value: 0.80-0.85.

            0.75 = aggressive merging
            0.80 = moderate merging
            0.85 = conservative merging
            0.90 = very conservative

        Returns
        -------
        str
            Compact timestamped timeline.
        """

        current_threshold = self.similarity_threshold + 0.05
        word_count = 100000

        while word_count > self.max_words:
            current_threshold -= 0.05
            if current_threshold <= 0:
                break

            observations = self.parse_observations(
                raw_text
            )

            if not observations:
                return ""

            current = {
                "start_time": observations[0]["time"],
                "end_time": observations[0]["time"],
                "text": observations[0]["text"],
            }

            groups = []

            for i in range(1, len(observations)):

                next_obs = observations[i]

                similarity = self.observation_similarity(
                    current["text"],
                    next_obs["text"],
                )

                if similarity >= current_threshold:

                    # Merge
                    current["end_time"] = next_obs["time"]

                    # Keep the longer description because it often
                    # contains slightly more information.
                    if len(next_obs["text"]) > len(current["text"]):
                        current["text"] = next_obs["text"]

                else:

                    # Similarity is low -> this represents a change.
                    groups.append(current)

                    current = {
                        "start_time": next_obs["time"],
                        "end_time": next_obs["time"],
                        "text": next_obs["text"],
                    }

            # Add final group
            groups.append(current)

            output = []

            for group in groups:

                start = group["start_time"]
                end = group["end_time"]

                if start == end:
                    timestamp = f"{start:.2f}s"
                else:
                    timestamp = (
                        f"{start:.2f}-{end:.2f}s"
                    )

                output.append(
                    f"{timestamp}: {group['text']}"
                )

            output = "\n".join(output)
            output = self.normalize_text(output)
            
            word_count = len(output.split())

        return output, word_count