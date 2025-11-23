from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def projection_mlp(y, prj_dim: int, name: str):
    num_units = y.shape[-1]
    y = Dense(
        units=num_units,
        kernel_initializer="he_normal",
        activation="silu",
        name=f"{name}_prj1",
    )(y)
    y = Dense(
        units=prj_dim,
        kernel_initializer="glorot_uniform",
        activation=None,
        name=f"{name}_prj2",
    )(y)
    return y


def projection_conv(filters, name):
    return Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        kernel_initializer="glorot_uniform",
        activation=None,
        padding="same",
        name=name,
    )


def mobile_vit_block(y, prj_dim: int, name: str):
    inp = y
    feature_dim = y.shape[-1]
    y = projection_conv(filters=prj_dim, name=f"{name}_prj1")(y)
    _b, h, w, f = y.shape
    y = Reshape((h * w, f), name=f"{name}_fold")(y)
    y = MultiHeadAttention(key_dim=32, num_heads=4, name=f"{name}_mha")(y, y)
    y = Reshape((h, w, f), name=f"{name}_unfold")(y)
    y = projection_conv(filters=feature_dim, name=f"{name}_prj2")(y)
    return Add(name=f"{name}_residual")([inp, y])


def squeeze_excite_block(y, name: str, ratio=16):
    inp = y
    feature_dim = y.shape[-1]
    y = GlobalAveragePooling2D(name=f"{name}_squeeze")(y)
    y = Dense(
        feature_dim // ratio,
        kernel_initializer="he_normal",
        activation="silu",
        use_bias=True,
        name=f"{name}_excite",
    )(y)
    y = Dense(
        feature_dim,
        kernel_initializer="he_normal",
        activation="sigmoid",
        use_bias=False,
        name=f"{name}_gating",
    )(y)
    y = Multiply(name=f"{name}_scale")([inp, y])
    return Add(name=f"{name}_residual")([inp, y])


def create_embedding_model(
    img_hw: int,
    num_filters: int,
    depth: int,
    act_fn: str,
    embedding_dim: int,
    projection_dim: int,
    include_vit_blocks: bool,
    include_squeeze_excite: bool,
):
    inps = Input(shape=(img_hw, img_hw, 3))
    y = inps
    for b in range(depth):
        y = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=2,
            activation=act_fn,
            padding="same",
            name=f"b{b}_conv",
        )(y)
        if include_squeeze_excite:
            y = squeeze_excite_block(y, name=f"b{b}_se")
        num_filters *= 2
    if include_vit_blocks:
        y = mobile_vit_block(y, prj_dim=256, name="vit")
    y = GlobalAveragePooling2D()(y)
    embeddings = Dense(
        units=embedding_dim,
        kernel_initializer="glorot_uniform",
        activation=None,
        name="embedding",
    )(y)
    if projection_dim is not None:
        embeddings = projection_mlp(
            embeddings, prj_dim=projection_dim, name="emb_projection"
        )

    return Model(inps, embeddings)
