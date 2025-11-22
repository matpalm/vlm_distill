import os
from PIL import Image
from multiprocessing import Pool
import tqdm

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--hw", type=int, default=640)
opts = parser.parse_args()
print("opts", opts)


def resize_image(source_path):
    try:
        output_path = os.path.join(opts.output_dir, os.path.basename(source_path))
        (
            Image.open(source_path)
            .convert("RGB")
            .resize((opts.hw, opts.hw), Image.Resampling.LANCZOS)
            .save(output_path)
        )
    except Exception as e:
        print("failed to convert", source_path, str(e))


if __name__ == "__main__":
    os.makedirs(opts.output_dir, exist_ok=True)
    fnames = [f.strip() for f in open(opts.manifest, "r").readlines()]
    with Pool() as pool:
        _ = list(
            tqdm.tqdm(pool.imap_unordered(resize_image, fnames), total=len(fnames))
        )
