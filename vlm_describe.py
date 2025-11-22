# given a manifest of files generate a npy array with clip embeddings of the imgs

import argparse
import tqdm
import numpy as np

from models import VLM

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--txt-output", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

vlm = VLM()

with open(opts.txt_output, "w") as f:
    fnames = [f.split("\t")[0] for f in open(opts.manifest, "r").readlines()]
    for fname in tqdm.tqdm(fnames):
        result = vlm.prompt(
            prompt=opts.prompt,
            img_path=fname,
        )
        print(result, file=f, flush=True)
