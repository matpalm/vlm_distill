import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import MSE
import json, pickle, tqdm

from data import create_img_embedding_ds, create_img_label_ds
from models import create_embedding_model
from check_knn import check
from util import DTS, ensure_dir_exists

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--img-hw", type=int, default=640)
parser.add_argument(
    "--embedding-type",
    type=str,
    required=True,
    help="which pretrained embedding to use as target",
)
opts = parser.parse_args()
print(opts)

train_ds, num_train = create_img_embedding_ds(
    split=f"{opts.dataset}/{opts.split}",
    img_hw=opts.img_hw,
    embedding_type=opts.embedding_type,
    cache=True,
)
train_ds.prefetch(1)

for _ in tqdm.tqdm(train_ds, total=num_train, desc="priming cache"):
    pass
