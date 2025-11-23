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
from PIL import Image
from functools import lru_cache

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


class Clip(object):

    def __init__(self):
        # Load CLIP model
        with timer("load model"):
            self.model = SentenceTransformer("clip-ViT-B-16")

    @lru_cache(128)
    def encode_img_fname(self, fname: str):
        with timer("embed img"):
            return self.model.encode(Image.open(fname))

    @lru_cache(128)
    def encode_text(self, text_or_list: str):
        with timer("embed text"):
            return self.model.encode(text_or_list)

    def embedding_dim(self):
        return 512
