from sentence_transformers import SentenceTransformer, util
import numpy as np
from PIL import Image

from pretrained_models import VLM
from util import timer

# create CLM And clip wrappers
with timer("load VLM"):
    vlm = VLM()
with timer("load clip"):
    clip = SentenceTransformer("clip-ViT-B-16")

img_path = "imgs/Selection_073.png"

# describe image with VLM
with timer("describe img"):
    vlm_description = vlm.prompt(
        prompt="Describe the following image using a short paragraph.",
        img_path=img_path,
    )
    print(vlm_description)

# embed image directly with clip
with timer("clip embed img"):
    img_emb = clip.encode(Image.open(img_path))
    np.save("img.npy", img_emb)

# embed the VLM description also
with timer("clip embed desc"):
    desc_emb = clip.encode(vlm_description)
    np.save("desc.npy", desc_emb)

cos_scores = util.cos_sim(img_emb, desc_emb)
print("cos_scores", cos_scores)
