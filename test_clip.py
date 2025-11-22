import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


from sentence_transformers import SentenceTransformer, util
import numpy as np
from util import timer
from PIL import Image

# Load CLIP model
with timer("load model"):
    model = SentenceTransformer("clip-ViT-B-16")

# Encode an image:
with timer("embed img"):
    img_emb = model.encode(Image.open("imgs/Selection_073.png"))

# Encode text descriptions
with timer("embed text"):
    text_emb = model.encode(
        [
            "Two dogs in the snow",
            "A modular synth",
            "A cat on a table",
            "A picture of London at night",
        ]
    )

# Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
