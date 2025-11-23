import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.optimizers import Adam

from data import create_img_embedding_ds, create_img_label_ds
from models import create_embedding_model
from check_knn import check

# create datasets for training the keras model
# (X, y) are (img, embeddings_from_whatever_pretrained_model )

train_ds = create_img_embedding_ds(
    split="cat_dog_1k/train", img_hw=640, embedding_type="clip_embed_img.npy"
)
train_ds = train_ds.cache()
train_ds = train_ds.batch(16)
validate_ds = create_img_embedding_ds(
    split="cat_dog_1k/validate", img_hw=640, embedding_type="clip_embed_img.npy"
)
validate_ds = validate_ds.cache()
validate_ds = validate_ds.batch(16)

# build and train keras model

model = create_embedding_model(
    img_hw=640,
    num_filters=8,
    depth=6,
    act_fn="silu",
    embedding_dim=512,
    projection_dim=None,
    include_vit_blocks=False,
    include_squeeze_excite=False,
)
print(model.summary())
model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
model.fit(train_ds, validation_data=validate_ds, epochs=10)

# use keras model to generate embeddings for knn train and test datasets
# (X, y) are ( img, true labels )


def generate_embeddings_from_model(split: str):
    embeddings = []
    ys = []
    for x, y in create_img_label_ds(split=split, img_hw=640).batch(16):
        embeddings.append(model(x))
        ys.append(y)
    return np.concatenate(embeddings), np.concatenate(ys)


# recall; y_ are the true values
x_train, y_train = generate_embeddings_from_model("knn/train")
x_test, y_test = generate_embeddings_from_model("knn/test")
check(x_train, y_train, x_test, y_test)
