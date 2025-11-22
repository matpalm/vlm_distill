import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from data import LABELS, LABEL_TO_IDX, load_embeddings_x_y

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--embedding-npy", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

x_train, y_train = load_embeddings_x_y(opts.train, opts.embedding_npy)
x_test, y_test = load_embeddings_x_y(opts.test, opts.embedding_npy)

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred, target_names=LABELS))
