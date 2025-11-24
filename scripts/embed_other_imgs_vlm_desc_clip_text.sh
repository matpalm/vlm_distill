#!/usr/bin/env bash

# run VLM for descriptions, then run those descripions through clip
source hf

for D in 1k; do
 for S in train validate test; do
   python3 vlm_describe.py \
    --manifest data/cat_dog/$D/$S/manifest.txt \
    --prompt 'describe this image in a sentence' \
    --txt-output data/cat_dog/$D/$S/p1/descriptions.txt
   python3 clip_embed_text.py \
    --text data/cat_dog/$D/$S/p1/descriptions.txt \
    --npy-output data/cat_dog/$D/$S/p1/clip_embed_text.npy
 done
done

for D in 1k; do
 for S in train validate test; do
   python3 vlm_describe.py \
    --manifest data/cat_dog/$D/$S/manifest.txt \
    --prompt 'describe the primary features of this image, in a single sentence, with respect to classifying the image as a cat, or a dog, or neither.' \
    --txt-output data/cat_dog/$D/$S/p2/descriptions.txt
   python3 clip_embed_text.py \
    --text data/cat_dog/$D/$S/p2/descriptions.txt \
    --npy-output data/cat_dog/$D/$S/p2/clip_embed_text.npy
 done
done

python3 vlm_describe.py \
 --manifest data/open_images/1k/train/manifest.txt \
 --prompt 'describe this image in a sentence' \
 --txt-output data/open_images/1k/train/p1/descriptions.txt

python3 vlm_describe.py \
 --manifest data/open_images/1k/train/manifest.txt \
 --prompt 'describe the primary features of this image, in a single sentence, with respect to classifying the image as a cat, or a dog, or neither.' \
 --txt-output data/open_images/1k/train/p2/descriptions.txt

for P in p1 p2; do
 python3 clip_embed_text.py \
  --text data/open_images/1k/train/$P/descriptions.txt \
  --npy-output data/open_images/1k/train/$P/clip_embed_text.npy
done
