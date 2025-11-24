#!/usr/bin/env bash

# run images directly through clip
source hf
for S in knn/train knn/test; do
 python3 clip_embed_img.py \
  --manifest data/$S/manifest.txt \
  --npy-output data/$S/clip_embed_img.npy
done

# check knn zero shot performance
source jax
python3 check_knn.py \
 --train knn/train \
 --test knn/test \
 --embedding-npy clip_embed_img.npy
