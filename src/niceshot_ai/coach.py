import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig)
import json


class Coach:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.model, self.tokenizer = self.load_model()


    def load_model(self):
        MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True)

        config = AutoConfig.from_pretrained(
            MODEL_ID,
            trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16)

        return model, tokenizer


    def infer(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt")

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                #temperature=0.2,
                do_sample=False,)
                #top_p=0.9)

        input_length = inputs["input_ids"].shape[1]

        answer = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True)

        return answer


    def analyze_session(self, observations_file_path, prompt):
        suggestions = []
        with open(observations_file_path, "r", encoding="utf-8") as f:
            for line in f:
                llm_prompt = prompt
                record = json.loads(line)
                clip = record["clip"]
                msg = record["msg"]

                llm_prompt += msg
                print(llm_prompt)
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful FPS gameplay analyst."
                    },
                    {
                        "role": "user",
                        "content": llm_prompt
                    }
                ]

                response = self.infer(messages)

                suggestion = {'clip': clip, 'coaching': response}
                suggestions.append(suggestion)

        return suggestions