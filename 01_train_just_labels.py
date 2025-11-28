import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
import json, pickle
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from data import create_ds, ds_post_processing
from models import create_backbone, create_classifier_head
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
parser.add_argument("--learning-rate", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--include-vit-blocks", action="store_true")
parser.add_argument("--include-squeeze-excite", action="store_true")
opts = parser.parse_args()
print(opts)

json.dump(vars(opts), open(f"runs/{run}/opts.json", "w"))

# create datasets for training the keras model
# (X, y) are (img, embeddings_from_whatever_pretrained_model )

train_ds, num_train, classes_to_labels = create_ds(
    split=f"{opts.dataset}/train",
    img_hw=opts.img_hw,
    embedding_type=None,
    include_labels=True,
    cache=True,
)
train_ds = ds_post_processing(
    train_ds, batch_size=opts.batch_size
)  # , shuffle=True, lr_flip=True)
print("|train_ds|", num_train, "classes_to_labels", classes_to_labels)

val_ds, num_val, classes_to_labels = create_ds(
    split=f"{opts.dataset}/test",
    img_hw=opts.img_hw,
    embedding_type=None,
    include_labels=True,
    cache=True,
)
val_ds = ds_post_processing(val_ds, batch_size=10, shuffle=False, lr_flip=False)
print("|val_ds|", num_val, classes_to_labels)

use_dummy_model = False

if use_dummy_model:
    inp = Input((opts.img_hw, opts.img_hw, 3))
    y = inp
    y = Dense(units=16, activation="relu")(y)
    y = GlobalAveragePooling2D()(y)
    y = Dense(units=len(classes_to_labels), activation=None)(y)
    model = Model(inp, y)
else:
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
    classifier = create_classifier_head(feature_dim, num_classes=len(classes_to_labels))
    print(classifier.summary())
    inp = backbone.input
    y = backbone(inp)
    y = classifier(y)
    model = Model(inp, y)
    print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=opts.learning_rate),
    loss=SparseCategoricalCrossentropy(from_logits=True),
)

class_names = classes_to_labels.keys()
callbacks = [
    EvalCallback("train", train_ds, class_names, cb_freq=10),
    EvalCallback("val", val_ds, class_names, cb_freq=10),
]

model.fit(train_ds, validation_data=val_ds, epochs=opts.epochs, callbacks=callbacks)
