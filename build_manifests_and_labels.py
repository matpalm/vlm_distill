import os, glob
import numpy as np
import json
from typing import List

from util import ensure_dir_exists
# from data import CLASS_TO_LABEL

# collect all cat, dog images
data = {
    "cat": iter(sorted(glob.glob("data/pet_images/cat/*jpg"))),
    "dog": iter(sorted(glob.glob("data/pet_images/dog/*jpg"))),
}


def write_manifest_and_labels(split: str, total_count: int):
    classes = data.keys()
    CLASS_TO_LABEL = dict(zip(classes, range(len(classes))))
    ensure_dir_exists(f"data/{split}")
    labels = []
    with open(f"data/{split}/manifest.txt", "w") as f:
        for c in ["cat", "dog"]:
            for _ in range(total_count // 2):
                print(next(data[c]), file=f)
                labels.append(CLASS_TO_LABEL[c])
    # TODO: write class_to_label.json
    np.save(f"data/{split}/y_true.npy", np.array(labels))


def write_rnd_objs_manifest_and_labels(splits: List[str], total_counts: List[int]):
    # write_manifest_and_labels is specific to cat/dog, and we've
    # already written a stack of data for it, so fork a hacky
    # version for rnd_objs and come back to unify this ( if required  )later.

    assert len(splits) == len(total_counts)

    def files_for_idx(idx: str):
        base_dir = f"/data2/zero_shot_detection/data_v4/train/reference_patches"
        return iter(sorted(glob.glob(f"{base_dir}/{idx}/*png")))

    data = {
        "red": files_for_idx("003"),
        "green": files_for_idx("004"),
        "blue": files_for_idx("000"),
    }
    classes = data.keys()
    class_to_labels = dict(zip(classes, range(len(classes))))

    for split, total_count in zip(splits, total_counts):
        count_per_class = total_count // len(classes)
        ensure_dir_exists(f"data/{split}")
        with open(f"data/{split}/class_to_labels.json", "w") as f:
            json.dump(class_to_labels, fp=f)
        labels = []
        with open(f"data/{split}/manifest.txt", "w") as f:
            for c in data.keys():
                for _ in range(count_per_class):
                    print(next(data[c]), file=f)
                    labels.append(class_to_labels[c])
        np.save(f"data/{split}/y_true.npy", np.array(labels))


# deprecated old experiment re: zero shot replication
# write_manifest_and_labels("knn/train", 100)
# write_manifest_and_labels("knn/test", 100)

# write_manifest_and_labels("cat_dog_1k/train", total_count=1_000)
# write_manifest_and_labels("cat_dog_1k/validate", total_count=100)
# write_manifest_and_labels("cat_dog_1k/test", total_count=100)
# write_manifest_and_labels("cat_dog_10k/train", total_count=10_000)
# write_manifest_and_labels("cat_dog_10k/validate", total_count=100)
# write_manifest_and_labels("cat_dog_10k/test", total_count=100)

write_rnd_objs_manifest_and_labels(
    splits=["rnd_objs/train", "rnd_objs/test"], total_counts=[100, 100]
)
