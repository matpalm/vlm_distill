#!/usr/bin/env bash

source hf
for S in train_knn test_knn; do
 for L in cat dog; do
  python3 vlm_describe.py \
   --manifest data/$S/$L/manifest.tsv \
   --prompt 'describe this image in a sentence' \
   --txt-output data/$S/$L/p1/descriptions.txt
  python3 clip_embed_text.py \
   --text data/$S/$L/p1/descriptions.txt \
   --npy-output data/$S/$L/p1/clip_embed_text.npy
 done
done

# knn performance
source jax
python3 check_knn.py \
 --train train_knn \
 --test test_knn \
 --embedding-npy p1/clip_embed_text.npy
