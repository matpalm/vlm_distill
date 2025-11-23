# given a manifest of files generate a npy array with clip embeddings of the imgs

import argparse
import tqdm
import numpy as np

from pretrained_models import Clip
from util import parse_manifest, ensure_dir_exists_for_file

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--npy-output", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

ensure_dir_exists_for_file(opts.npy_output)

clip = Clip()
fnames = parse_manifest(opts.manifest)

# pre alloc to ensure we have mem
embeddings = np.empty((len(fnames), clip.embedding_dim()), dtype=np.float32)

for i, fname in enumerate(tqdm.tqdm(fnames)):
    embeddings[i] = clip.encode_img_fname(fname)
np.save(opts.npy_output, embeddings)
