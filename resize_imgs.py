import os
from PIL import Image
from multiprocessing import Pool
import tqdm

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--hw", type=int, default=640)
parser.add_argument(
    "--squash",
    action="store_true",
    help="whether to squash or keep aspect with padding",
)
opts = parser.parse_args()
print("opts", opts)


def resize_squash(pil_img):
    return pil_img.resize((opts.hw, opts.hw), Image.Resampling.LANCZOS)


def resize_with_padding(pil_img):
    img_w, img_h = pil_img.size
    rescale = opts.hw / max(img_w, img_h)
    resized_img_w = int(img_w * rescale)
    resized_img_h = int(img_h * rescale)
    pil_img = pil_img.resize((resized_img_w, resized_img_h), Image.Resampling.LANCZOS)
    px = (opts.hw - resized_img_w) // 2
    py = (opts.hw - resized_img_h) // 2
    canvas = Image.new("RGB", (opts.hw, opts.hw))
    canvas.paste(pil_img, (px, py))
    return canvas


def resize_image(source_path):
    try:
        output_path = os.path.join(opts.output_dir, os.path.basename(source_path))
        pil_img = Image.open(source_path).convert("RGB")
        if opts.squash:
            pil_img = resize_squash(pil_img)
        else:
            pil_img = resize_with_padding(pil_img)
        pil_img.save(output_path)
    except Exception as e:
        print("failed to convert", source_path, str(e))


if __name__ == "__main__":
    os.makedirs(opts.output_dir, exist_ok=True)
    fnames = [f.strip() for f in open(opts.manifest, "r").readlines()]
    with Pool() as pool:
        _ = list(
            tqdm.tqdm(
                pool.imap_unordered(resize_image, fnames),
                total=len(fnames),
                desc="resizing imgs",
            )
        )
