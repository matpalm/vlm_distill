import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


from sentence_transformers import SentenceTransformer, util
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from util import timer


class VLM(object):
    def __init__(self):
        with timer("load model"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                dtype="auto",
                device_map="auto",
            )
        with timer("load processor"):
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct"
            )

    def prompt(self, prompt: str, img_path: str):

        # TODD: should be able to run prompt and cache before providing img

        with timer("process message"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {"type": "image", "image": img_path},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        with timer("generate result"):
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return output_text[0]
