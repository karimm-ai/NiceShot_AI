import cv2
from qwen_vl_utils import process_vision_info
import os, json
import random


class Coach:
    def __init__(self, output_dir: str, type: str | None = None):
        self.model, self.processor = self.load_model()
        self.output_dir = output_dir
        self.type = type
        self.sample_size = self.specify_sample()
        
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
            max_new_tokens=80,
        )

        response = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response


    def pre_process_gameplay(self, video_path: str, sample_every: int = 6):
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

        last_frame_to_analyze = total_frames - 110

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


    def analyze_session(self, messages, folder):
        output_file = f"{self.output_dir}/coaching.jsonl"
        clips = []

        for clip in os.listdir(folder):
            clips.append(clip)

        k = max(1, int(len(clips) * self.sample_size))
        sample = random.sample(clips, k=k)

        with open(output_file, "a", encoding="utf-8") as f:
            for clip in sample:
                self.pre_process_gameplay(f"{folder}/{clip}")
                msg = self.infer(messages, self.model, self.processor)                
                record = {
                    "clip": clip,
                    "msg": msg
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def specify_sample(self):
        if self.type == 'basic':
            return 0.25

        elif self.type == 'short':
            return 0.5

        elif self.type == 'long':
            return 0.75

        elif self.type == 'very long':
            return 1
        
        else:
            return 0