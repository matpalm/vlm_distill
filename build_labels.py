# given a manifest of files generate a npy array with the 0, 1 labels

import argparse
import numpy as np

from util import parse_manifest, ensure_dir_exists_for_file
import data

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--labels-npy", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

ensure_dir_exists_for_file(opts.labels_npy)

fnames = parse_manifest(opts.manifest)
labels = []
for fname in fnames:
    clazz = fname.split("/")[-2]  # o_O
    assert clazz in data.CLASSES
    labels.append(data.CLASS_TO_LABEL[clazz])
np.save(opts.labels_npy, np.array(labels))
