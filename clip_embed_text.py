# given a manifest of files generate a npy array with clip embeddings of the imgs

import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--text", type=str, required=True)
parser.add_argument("--npy-output", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

if os.path.exists(opts.npy_output):
    print("--npy-output already exists")
    exit()

import tqdm
import numpy as np

from pretrained_models import Clip
from util import parse_manifest, ensure_dir_exists_for_file

ensure_dir_exists_for_file(opts.npy_output)

clip = Clip()
embeddings = []
lines = parse_manifest(opts.text)
for line in tqdm.tqdm(lines, desc="clip embed text"):
    embeddings.append(clip.encode_text(line))
np.save(opts.npy_output, np.stack(embeddings))
