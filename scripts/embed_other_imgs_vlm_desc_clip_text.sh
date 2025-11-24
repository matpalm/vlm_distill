#!/usr/bin/env bash

# run VLM for descriptions, then run those descripions through clip
source hf

#for D in cat_dog_1k; do
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

#for D in cat_dog_1k; do
for D in cat_dog_1k; do
 for S in train validate test; do
   python3 vlm_describe.py \
    --manifest data/$D/$S/manifest.txt \
    --prompt 'describe the primary features of this image, in a single sentence, with respect to classifying the image as a cat, or a dog, or neither.' \
    --txt-output data/$D/$S/p2/descriptions.txt
   python3 clip_embed_text.py \
    --text data/$D/$S/p2/descriptions.txt \
    --npy-output data/$D/$S/p2/clip_embed_text.npy
 done
done
