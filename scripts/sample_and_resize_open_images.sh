#!/usr/bin/env bash

source jax

# sample resized images from open images; 1k, 10k and 100k
# ( each from a different shard

find /data/open_images/original/train_0 -type f -name \*jpg | head -n1000 > /tmp/manifest.$$
python3 resize_imgs.py --manifest /tmp/manifest.$$ --output-dir data/imgs/open_images/1k --hw 640

find /data/open_images/original/train_1 -type f -name \*jpg | head -n10000 > /tmp/manifest.$$
python3 resize_imgs.py --manifest /tmp/manifest.$$ --output-dir data/imgs/open_images/10k --hw 640

find /data/open_images/original/train_2 -type f -name \*jpg | head -n100000 > /tmp/manifest.$$
python3 resize_imgs.py --manifest /tmp/manifest.$$ --output-dir data/imgs/open_images/100k --hw 640

# set up manifests

for D in 1k 10k 100k; do
 mkdir -p data/open_images/$D/train
 find data/imgs/open_images/$D -type f -name *jpg > data/open_images/$D/train/manifest.txt
done