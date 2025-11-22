import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--embedding-npy", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

LABELS = ["cat", "dog"]
LABEL_TO_IDX = dict(zip(LABELS, range(len(LABELS))))


def load_x_y(split: str):
    all_embeddings = []
    y = []
    for label in ["cat", "dog"]:
        embeddings = np.load(f"data/{split}/{label}/{opts.embedding_npy}")
        all_embeddings.append(embeddings)
        y += [LABEL_TO_IDX[label]] * len(embeddings)
    x = np.concatenate(all_embeddings)
    return x, y


x_train, y_train = load_x_y("train")
x_test, y_test = load_x_y("test")

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred, target_names=LABELS))
