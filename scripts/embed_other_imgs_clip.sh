#!/usr/bin/env bash

# run images directly through clip
source hf
# for D in cat_dog_1k cat_dog_10k; do
#  for S in train validate test; do
#   python3 clip_embed_img.py \
#    --manifest data/$D/$S/manifest.txt \
#    --npy-output data/$D/$S/clip_embed_img.npy
#  done
# done

python3 clip_embed_img.py \
 --manifest data/open_images_100k/manifest.txt \
 --npy-output data/open_images_100k/clip_embed_img.npy