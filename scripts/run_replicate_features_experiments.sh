#!/usr/bin/env bash

source jax

#python train_and_test_student.py --dataset cat_dog/1k --embedding-type clip_embed_img.npy
#python train_and_test_student.py --dataset cat_dog/10k --embedding-type clip_embed_img.npy

#python train_and_test_student.py --dataset cat_dog/1k --embedding-type p1/clip_embed_text.npy
#python train_and_test_student.py --dataset cat_dog/1k --embedding-type p2/clip_embed_text.npy

python train_and_test_student.py --dataset open_images/1k --embedding-type clip_embed_img.npy --mse-loss-weight 0.5 --sim-loss-weight 1.0
python train_and_test_student.py --dataset open_images/10k --embedding-type clip_embed_img.npy --mse-loss-weight 0.5 --sim-loss-weight 1.0
python train_and_test_student.py --dataset open_images/100k --embedding-type clip_embed_img.npy --mse-loss-weight 0.5 --sim-loss-weight 1.0
