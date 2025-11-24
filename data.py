import numpy as np
import tensorflow as tf
from PIL import Image
from random import Random

from util import parse_manifest, ensure_dir_exists_for_file


CLASSES = ["cat", "dog"]
CLASS_TO_LABEL = dict(zip(CLASSES, range(len(CLASSES))))

def load_embeddings_x_y(split: str, embedding_fname: str):
    x = np.load(f"data/{split}/{embedding_fname}")
    y_true = np.load(f"data/{split}/y_true.npy")
    return x, y_true


def create_img_label_ds(split: str, img_hw: int, seed: int = None):

    manifest = parse_manifest(f"data/{split}/manifest.txt")
    labels = np.load(f"data/{split}/y_true.npy")
    assert len(manifest) == len(labels)

    def _generator():
        idxs = list(range(len(manifest)))
        if seed is not None:
            Random(seed).shuffle(idxs)
        for i in idxs:
            pil_img = Image.open(manifest[i]).convert("RGB").resize((img_hw, img_hw))
            x = np.array(pil_img, dtype=float) / 255.0
            y = labels[i]
            yield x, y

    return tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(img_hw, img_hw, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.uint8),
        ),
    )


def create_img_embedding_ds(
    split: str, img_hw: int, embedding_type: str, seed: int = 234, cache: bool = False
):

    manifest = parse_manifest(f"data/{split}/manifest.txt")
    embeddings = np.load(f"data/{split}/{embedding_type}")

    embedding_dim = embeddings.shape[-1]

    assert len(manifest) == len(embeddings)

    def _generator():
        idxs = list(range(len(manifest)))
        Random(seed).shuffle(idxs)
        for i in idxs:
            pil_img = Image.open(manifest[i]).convert("RGB").resize((img_hw, img_hw))
            x = np.array(pil_img)  # uint8
            y = embeddings[i]
            yield x, y

    def convert_dtype(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.0
        return x, y

    ds = tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            tf.TensorSpec(shape=(img_hw, img_hw, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(embedding_dim,), dtype=tf.float32),
        ),
    )
    if cache:
        cache_file = f"cache/{split}/{embedding_type}/cache_"
        ensure_dir_exists_for_file(cache_file)
        ds = ds.cache(cache_file)
    ds = ds.map(convert_dtype)

    return ds, len(manifest)
