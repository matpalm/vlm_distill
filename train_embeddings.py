import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
import json, pickle
from tensorflow.keras.losses import MSE, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *

from data import create_ds, ds_post_processing
from models import create_backbone, create_adapter
from callbacks import EvalCallback
from util import DTS, ensure_dir_exists

run = DTS()
print("run", run)
ensure_dir_exists(f"runs/{run}")

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--img-hw", type=int, default=640)
parser.add_argument("--base-num-filters", type=int, default=8)
parser.add_argument("--max-num-filters", type=int, default=None)
parser.add_argument("--embedding-type", type=str, default="clip_embed_img.npy")
parser.add_argument("--learning-rate", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--include-vit-blocks", action="store_true")
parser.add_argument("--include-squeeze-excite", action="store_true")
opts = parser.parse_args()
print(opts)

json.dump(vars(opts), open(f"runs/{run}/opts.json", "w"))

train_ds, num_train, classes_to_labels = create_ds(
    split=f"{opts.dataset}/train",
    img_hw=opts.img_hw,
    embedding_type=opts.embedding_type,
    include_labels=False,
    cache=True,
)
train_ds = ds_post_processing(
    train_ds, batch_size=opts.batch_size, shuffle=True, lr_flip=True
)
print("|train_ds|", num_train, "classes_to_labels", classes_to_labels)

val_ds, num_val, classes_to_labels = create_ds(
    split=f"{opts.dataset}/test",
    img_hw=opts.img_hw,
    embedding_type=opts.embedding_type,
    include_labels=False,
    cache=True,
)
val_ds = ds_post_processing(val_ds, batch_size=10, shuffle=False, lr_flip=False)
print("|val_ds|", num_val, classes_to_labels)

backbone = create_backbone(
    img_hw=opts.img_hw,
    base_num_filters=opts.base_num_filters,
    max_num_filters=opts.max_num_filters,
    depth=6,
    act_fn="silu",
    include_vit_blocks=opts.include_vit_blocks,
    include_squeeze_excite=opts.include_squeeze_excite,
)
print(backbone.summary())

feature_dim = backbone.output.shape[-1]
adapter = create_adapter(feature_dim, embedding_dim=512)
print(adapter.summary())

inp = backbone.input
y = adapter(backbone(inp))
model = Model(inp, y)
print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=opts.learning_rate),
    loss="mse",
)

callbacks = [
    ModelCheckpoint(filepath=f"runs/{run}/ckpts/" + "e{epoch:03d}_{loss:0.5f}.keras")
]
model.fit(train_ds, validation_data=val_ds, epochs=opts.epochs, callbacks=callbacks)
