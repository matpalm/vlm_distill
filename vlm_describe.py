import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--txt-output", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

if os.path.exists(opts.txt_output):
    print(f"skip; --txt-output [{opts.txt_output}] already exists")
    exit()

import tqdm
import numpy as np

from pretrained_models import VLM
from util import parse_manifest, ensure_dir_exists_for_file

ensure_dir_exists_for_file(opts.txt_output)

vlm = VLM()

with open(opts.txt_output, "w") as f:
    fnames = parse_manifest(opts.manifest)
    for fname in tqdm.tqdm(fnames, desc="vlm desc imgs"):
        result = vlm.prompt(prompt=opts.prompt, img_path=fname)
        result = result.replace("\n", " ")  # o_O
        print(result, file=f, flush=True)
