import numpy as np

from models import Clip

clip = Clip()


def l2_norm(e):
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


img_embeddings = l2_norm(
    np.stack(
        [
            clip.encode_img_fname("PetImages/Cat/0.jpg"),
            clip.encode_img_fname("PetImages/Dog/0.jpg"),
        ]
    )
)

desc_embeddings = l2_norm(
    np.stack(
        [
            clip.encode_text("an image of a cat"),
            clip.encode_text("an image of a dog"),
        ]
    )
)

# sims
print(np.dot(img_embeddings, desc_embeddings.T))
