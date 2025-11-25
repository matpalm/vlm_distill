import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import Callback
import json, pickle

from data import create_img_embedding_ds, create_img_label_ds
from models import create_embedding_model
from check_knn import check
from util import DTS, ensure_dir_exists

run = DTS()
print("run", run)
ensure_dir_exists(f"runs/{run}")


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--img-hw", type=int, default=640)
parser.add_argument("--base-num-filters", type=int, default=8)
parser.add_argument(
    "--embedding-type",
    type=str,
    required=True,
    help="which pretrained embedding to use as target",
)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)  # note: early stopping
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--target-embedding-dim", type=int, default=512)
parser.add_argument("--mse-loss-weight", type=float, default=0.5)
parser.add_argument("--sim-loss-weight", type=float, default=0.5)

opts = parser.parse_args()
print(opts)


json.dump(vars(opts), open(f"runs/{run}/opts.json", "w"))

# create datasets for training the keras model
# (X, y) are (img, embeddings_from_whatever_pretrained_model )

train_ds, _num_train = create_img_embedding_ds(
    split=f"{opts.dataset}/train",
    img_hw=opts.img_hw,
    embedding_type=opts.embedding_type,
    cache=True,
)
train_ds = train_ds.shuffle(opts.batch_size * 10)
train_ds = train_ds.batch(opts.batch_size)

try:
    validate_ds, _num_validate = create_img_embedding_ds(
        split=f"{opts.dataset}/validate",
        img_hw=opts.img_hw,
        embedding_type=opts.embedding_type,
        cache=True,
    )
    validate_ds = validate_ds.batch(16)
except FileNotFoundError:
    # not all datasets, e.g. open_images, even have a validation set
    validate_ds = None

# build and train keras model
# note: not bothering with projection for now

model = create_embedding_model(
    img_hw=opts.img_hw,
    num_filters=opts.base_num_filters,
    depth=6,
    act_fn="silu",
    embedding_dim=opts.target_embedding_dim,
    projection_dim=None,
    include_vit_blocks=True,
    include_squeeze_excite=False,
)


def _to_file(s):
    with open(f"runs/{run}/model_summary", "w") as f:
        print(s, file=f)


model.summary(print_fn=_to_file)

def cosine_sim_loss(y_true, y_pred):
    y_true = y_true / jnp.linalg.norm(y_true, axis=-1, keepdims=True)
    y_pred = y_pred / jnp.linalg.norm(y_pred, axis=-1, keepdims=True)
    sims = 1 - jnp.einsum("BE,BE->B", y_true, y_pred)
    return jnp.mean(sims)


def combined_loss(y_true, y_pred):
    loss = 0
    if opts.mse_loss_weight > 0:
        loss += MSE(y_true, y_pred) * opts.mse_loss_weight
    if opts.sim_loss_weight > 0:
        loss += cosine_sim_loss(y_true, y_pred) * opts.sim_loss_weight
    return loss


model.compile(optimizer=Adam(learning_rate=opts.learning_rate), loss=combined_loss)

callbacks = []

# if validate_ds is not None:
#     callbacks.append(
#         EarlyStopping(
#             monitor="val_loss",
#             patience=5,
#             min_delta=1e-4,
#             verbose=0,
#             mode="auto",
#             restore_best_weights=True,
#         )
#     )


def generate_embeddings_from_model(model, ds):
    embeddings = []
    ys = []
    for x, y in ds:
        embeddings.append(model(x))
        ys.append(y)
    return np.concatenate(embeddings), np.concatenate(ys)


class ZeroShotKNNTestCallback(Callback):

    def __init__(self, cb_freq: int = 1, log_fname: str = None):
        self.knn_train = create_img_label_ds(split="knn/train", img_hw=640).batch(16)
        self.knn_test = create_img_label_ds(split="knn/test", img_hw=640).batch(16)
        self.cb_freq = cb_freq
        self.log = None if log_fname is None else open(log_fname, "w")

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.cb_freq == 0:
            x_train, y_train = generate_embeddings_from_model(
                self.model, self.knn_train
            )
            x_test, y_test = generate_embeddings_from_model(self.model, self.knn_test)
            report = check(x_train, y_train, x_test, y_test)
            for k in ["accuracy", "macro avg", "weighted avg"]:
                del report[k]
            logs["mean_f1"] = (
                report["cat"]["f1-score"] + report["dog"]["f1-score"]
            ) / 2
            report["epoch"] = epoch
            if self.log:
                print(json.dumps(report), file=self.log, flush=True)
        return logs


callbacks.append(
    ZeroShotKNNTestCallback(cb_freq=2, log_fname=f"runs/{run}/knn_metrics.jsonl")
)

# last to pick up f1 info from knn test
callbacks.append(
    TensorBoard(
        log_dir=f"tb/{run}",
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
    )
)

history = model.fit(
    train_ds, validation_data=validate_ds, epochs=opts.epochs, callbacks=callbacks
)
pickle.dump(history, open(f"runs/{run}/history.pkl", "wb"))

# use keras model to generate embeddings for knn train and test datasets
# (X, y) are ( img, true labels )

def generate_embeddings_from_model(split: str):
    embeddings = []
    ys = []
    for x, y in create_img_label_ds(split=split, img_hw=opts.img_hw).batch(16):
        embeddings.append(model(x))
        ys.append(y)
    return np.concatenate(embeddings), np.concatenate(ys)


# recall; y_ are the true values
x_train, y_train = generate_embeddings_from_model("knn/train")
x_test, y_test = generate_embeddings_from_model("knn/test")
report = check(x_train, y_train, x_test, y_test)
print(report)
json.dump(report, open(f"runs/{run}/report", "w"))

print(run)
