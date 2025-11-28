import numpy as np
import tensorflow as tf
from PIL import Image
from random import Random
from tensorflow.keras.layers import RandomFlip
import json
from PIL import Image

from util import parse_manifest, ensure_dir_exists_for_file


# CLASSES = ["cat", "dog"]
# CLASS_TO_LABEL = dict(zip(CLASSES, range(len(CLASSES))))

def load_embeddings_x_y(split: str, embedding_fname: str):
    x = np.load(f"data/{split}/{embedding_fname}")
    y_true = np.load(f"data/{split}/y_true.npy")
    return x, y_true


def create_ds(
    split: str,
    img_hw: int,
    embedding_type: str,  # if None, don't emit embedding
    include_labels: bool,
    seed: int = 123,
    cache: bool = False,
):
    if (embedding_type is None) and (not include_labels):
        raise Exception("need to include one of embeddings or y_labels in output")

    manifest = parse_manifest(f"data/{split}/manifest.txt")
    if embedding_type is not None:
        embeddings = np.load(f"data/{split}/{embedding_type}")
        assert len(manifest) == len(embeddings)
        embedding_dim = embeddings.shape[-1]
    if include_labels:
        labels = np.load(f"data/{split}/y_true.npy")
        assert len(manifest) == len(labels)

    def _generator():
        idxs = list(range(len(manifest)))
        if seed is not None:
            Random(seed).shuffle(idxs)
        for i in idxs:
            pil_img = Image.open(manifest[i]).convert("RGB").resize((img_hw, img_hw))
            output = [np.array(pil_img)]  # uint8
            if embedding_type is not None:
                output.append(embeddings[i])
            if include_labels:
                # ensure batched labeled as (B, 1) and not just (B, )
                output.append(np.expand_dims(labels[i], axis=-1))
            yield tuple(output)

    output_signature = [tf.TensorSpec(shape=(img_hw, img_hw, 3), dtype=tf.uint8)]
    if embedding_type is not None:
        output_signature.append(tf.TensorSpec(shape=(embedding_dim,), dtype=tf.float32))
    if include_labels:
        output_signature.append(tf.TensorSpec(shape=(1), dtype=tf.uint8))

    ds = tf.data.Dataset.from_generator(
        _generator, output_signature=tuple(output_signature)
    )

    if cache:
        cache_file = f"cache/{split}/{img_hw}/"
        if embedding_type is not None:
            cache_file += f"e_{embedding_type}/"
        cache_file += f"labels_{include_labels}"
        ensure_dir_exists_for_file(cache_file)
        ds = ds.cache(cache_file)

    def convert_dtype(*xy):
        # xy might be x, y or x, y, y
        xy = list(xy)
        xy[0] = tf.cast(xy[0], dtype=tf.float32) / 255.0
        return tuple(xy)

    ds = ds.map(convert_dtype)

    try:
        with open(f"data/{split}/class_to_labels.json", "r") as f:
            classes_to_labels = json.load(f)
    except FileNotFoundError:
        print(f"no classes_to_labels for split=[{split}]")
        classes_to_labels = None

    return ds, len(manifest), classes_to_labels


def ds_post_processing(
    ds,
    batch_size: int,
    shuffle: bool = False,
    lr_flip: bool = False,
    drop_remainder: bool = True,
):
    if shuffle:
        ds = ds.shuffle(batch_size * 10)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if lr_flip:
        random_flip = RandomFlip(mode="horizontal")

        def flip_image(*xy):
            xy = list(xy)
            xy[0] = random_flip(xy[0], training=True)
            return tuple(xy)

        ds = ds.map(flip_image)

    return ds


def to_pil(a):
    if a.dtype != float:
        raise Exception("expected float data")
    a = np.array(a * 255, dtype=np.uint8)
    return Image.fromarray(a)
