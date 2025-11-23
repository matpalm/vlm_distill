import numpy as np
import tensorflow as tf

CLASSES = ["cat", "dog"]
CLASS_TO_LABEL = dict(zip(CLASSES, range(len(CLASSES))))

def load_embeddings_x_y(split: str, embedding_fname: str):
    x = np.load(f"data/{split}/{embedding_fname}")
    y_true = np.load(f"data/{split}/y_true.npy")
    return x, y_true
