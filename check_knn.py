import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

from data import load_embeddings_x_y, CLASSES


def check(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--embedding-npy", type=str, required=True)
    opts = parser.parse_args()

    x_train, y_train = load_embeddings_x_y(opts.train, opts.embedding_npy)
    x_test, y_test = load_embeddings_x_y(opts.test, opts.embedding_npy)

    print(check(x_train, y_train, x_test, y_test))
