import numpy as np

LABELS = ["cat", "dog"]
LABEL_TO_IDX = dict(zip(LABELS, range(len(LABELS))))


def load_embeddings_x_y(split: str, embedding_fname: str):
    all_embeddings = []
    y = []
    for label in ["cat", "dog"]:
        embeddings = np.load(f"data/{split}/{label}/{embedding_fname}")
        all_embeddings.append(embeddings)
        y.append(np.array([LABEL_TO_IDX[label]] * len(embeddings)))
    x = np.concatenate(all_embeddings)
    y = np.concatenate(y)
    return x, y
