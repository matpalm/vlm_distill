import os, glob
import numpy as np

from util import ensure_dir_exists
from data import CLASS_TO_LABEL

# collect all cat, dog images
data = {
    "cat": iter(sorted(glob.glob("data/pet_images/cat/*jpg"))),
    "dog": iter(sorted(glob.glob("data/pet_images/dog/*jpg"))),
}


def write_manifest_and_labels(split: str, total_count: int):
    ensure_dir_exists(f"data/{split}")
    labels = []
    with open(f"data/{split}/manifest.txt", "w") as f:
        for c in ["cat", "dog"]:
            for _ in range(total_count // 2):
                print(next(data[c]), file=f)
                labels.append(CLASS_TO_LABEL[c])
    np.save(f"data/{split}/y_true.npy", np.array(labels))


write_manifest_and_labels("knn/train", 100)
write_manifest_and_labels("knn/test", 100)
write_manifest_and_labels("cat_dog_1k/train", 1_000)
write_manifest_and_labels("cat_dog_1k/validate", 100)
write_manifest_and_labels("cat_dog_1k/test", 100)
write_manifest_and_labels("cat_dog_10k/train", 10_000)
write_manifest_and_labels("cat_dog_10k/validate", 100)
write_manifest_and_labels("cat_dog_10k/test", 100)
