#!/usr/bin/env bash

# run VLM for descriptions, then run those descripions through clip
source hf
#for S in cat_dog_1k cat_dog_10k; do
for D in cat_dog_1k; do
 for S in train validate test; do
   python3 vlm_describe.py \
    --manifest data/$D/$S/manifest.txt \
    --prompt 'describe this image in a sentence' \
    --txt-output data/$D/$S/p1/descriptions.txt
   python3 clip_embed_text.py \
    --text data/$D/$S/p1/descriptions.txt \
    --npy-output data/$D/$S/p1/clip_embed_text.npy
 done
done
