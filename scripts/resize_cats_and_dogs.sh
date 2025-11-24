#!/usr/bin/env bash
source jax
for L in Cat Dog; do
 find /data/kaggle_cats_and_dogs/PetImages/$L/ -type f > /tmp/manifest
 python3 resize_imgs.py --manifest /tmp/manifest --output-dir data/pet_images/$L/ --hw 640
done
mv data/pet_images/Cat data/pet_images/cat
mv data/pet_images/Dog data/pet_images/dog
