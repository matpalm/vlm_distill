#!/usr/bin/env bash

# run VLM for descriptions, then run those descripions through clip
source hf
for S in train_knn test_knn; do
  python3 vlm_describe.py \
   --manifest data/$S/manifest.tsv \
   --prompt 'describe this image in a sentence' \
   --txt-output data/$S/p1/descriptions.txt
  python3 clip_embed_text.py \
   --text data/$S/p1/descriptions.txt \
   --npy-output data/$S/p1/clip_embed_text.npy
done

# check knn zero shot performance
source jax
python3 check_knn.py \
 --train train_knn \
 --test test_knn \
 --embedding-npy p1/clip_embed_text.npy
