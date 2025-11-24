#!/usr/bin/env bash
source jax
find /data/open_images/original/train_0 -type f -name \*jpg | head -n100000 > /tmp/manifest.$$
time python3 resize_imgs.py --manifest /tmp/manifest.$$ --output-dir data/open_images_img/ --hw 640
rm /tmp/manifest.$$
