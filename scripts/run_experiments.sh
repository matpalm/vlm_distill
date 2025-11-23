python train_and_test_student.py --dataset cat_dog_1k --embedding-type clip_embed_img.npy --sim-loss-weight 0.5 --mse-loss-weight 0.5
python train_and_test_student.py --dataset cat_dog_10k --embedding-type clip_embed_img.npy --sim-loss-weight 0.5 --mse-loss-weight 0.5

python train_and_test_student.py --dataset cat_dog_1k --embedding-type p1/clip_embed_text.npy --sim-loss-weight 0.5 --mse-loss-weight 0.5
# don't have vlm descriptions for cat_dog_10k yet

python train_and_test_student.py --dataset cat_dog_1k --embedding-type p2/clip_embed_text.npy --sim-loss-weight 0.5 --mse-loss-weight 0.5
# don't have vlm descriptions for cat_dog_10k yet

