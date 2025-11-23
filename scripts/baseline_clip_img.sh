#!/usr/bin/env bash

# run images directly through clip
source hf
for S in train_knn test_knn; do
 python3 clip_embed_img.py \
  --manifest data/$S/manifest.tsv \
  --npy-output data/$S/clip_embed_img.npy
done

# check knn zero shot performance
source jax
python3 check_knn.py \
 --train train_knn \
 --test test_knn \
 --embedding-npy clip_embed_img.npy