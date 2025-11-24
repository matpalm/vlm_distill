#!/usr/bin/env bash

# run images directly through clip
source hf

for D in 1k 10k; do
 for S in train validate test; do
  python3 clip_embed_img.py \
   --manifest data/cat_dog/$D/$S/manifest.txt \
   --npy-output data/cat_dog/$D/$S/clip_embed_img.npy
 done
done

for D in 1k 10k 100k; do
 python3 clip_embed_img.py \
  --manifest data/open_images/$D/manifest.txt \
  --npy-output data/open_images/$D/clip_embed_img.npy
done

