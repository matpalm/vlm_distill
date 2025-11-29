from typing import Tuple

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model


def conv_batchnorm_act_block(
    y, num_filters: int, act_fn: str, use_seperable: bool, name: str
):
    if use_seperable:
        y = SeparableConv2D(
            filters=num_filters,
            kernel_size=3,
            strides=2,
            activation=None,
            use_bias=False,
            padding="same",
            depth_multiplier=1,
            name=f"{name}_conv",
        )(y)
    else:
        y = Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=2,
            activation=None,
            use_bias=False,
            padding="same",
            name=f"{name}_conv",
        )(y)
    y = BatchNormalization(name=f"{name}_bn")(y)
    y = Activation(act_fn, name=f"{name}_act")(y)
    return y


def mlp(y, dims: int, name: str):
    """simple MLP with no activation or batchnorm on last layer"""
    last_layer_idx = len(dims) - 1
    for i in range(len(dims)):
        if i != last_layer_idx:
            y = Dense(
                units=dims[i],
                kernel_initializer="he_normal",
                name=f"{name}_prj{i}",
            )(y)
            y = BatchNormalization(name=f"{name}_bn{i}")(y)
            y = Activation("silu", name=f"{name}_act{i}")(y)
            y = Dropout(0.1, name=f"{name}_dropout{i}")(y)
        else:
            y = Dense(
                units=dims[i],
                kernel_initializer="glorot_uniform",
                activation=None,
                name=f"{name}_prj{i}",
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
    y0 = LayerNormalization()(y)
    y = MultiHeadAttention(key_dim=32, num_heads=4, name=f"{name}_mha")(y0, y0)
    y = Add(name=f"{name}_mha_residual")([y0, y])
    y = Reshape((h, w, f), name=f"{name}_unfold")(y)
    y0 = LayerNormalization()(y)
    y = projection_conv(filters=feature_dim, name=f"{name}_prj2")(y)
    y = Add(name=f"{name}_prj_residual")([y0, y])
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


def create_backbone(
    img_hw: int,
    base_num_filters: int,
    depth: int,
    act_fn: str,
    include_vit_blocks: bool,
    include_squeeze_excite: bool,
    max_num_filters: int = None,
):
    num_filters = base_num_filters

    inps = Input(shape=(img_hw, img_hw, 3))
    y = inps
    # main conv stack
    for b in range(depth - 1):
        use_seperable = b >= 3
        y = conv_batchnorm_act_block(
            y, num_filters, act_fn, use_seperable, name=f"b{b}"
        )
        if include_squeeze_excite:
            y = squeeze_excite_block(y, name=f"b{b}_se")
        num_filters *= 2
        if max_num_filters is not None:
            num_filters = min(max_num_filters, num_filters)

    # ViT mixer
    if include_vit_blocks:
        y = mobile_vit_block(y, prj_dim=num_filters // 2, name="vit")

    # final ( post mixer ) conv
    y = conv_batchnorm_act_block(
        y, num_filters, act_fn, use_seperable=True, name=f"b{b+1}"
    )
    y = GlobalAveragePooling2D(name="features")(y)
    return Model(inps, y, name="backbone")


def create_classifier(feature_dim: int, num_classes: int):
    inps = Input(shape=(feature_dim,))
    logits = mlp(
        inps,
        dims=[feature_dim, num_classes],
        name="logits",
    )
    return Model(inps, logits, name="classifier")


def create_adapter(feature_dim: int, embedding_dim: int):
    inps = Input(shape=(feature_dim,))
    adapter = mlp(
        inps,
        dims=[feature_dim, embedding_dim],
        name="adapter",
    )
    return Model(inps, adapter, name="adapter")


def load_backbone_from_pretrained_model(ckpt: str):
    """ load the specified model and extract the 'backbone' """
    model = load_model(ckpt)
    for l in model.layers:
        if l.name == 'backbone':
            return l
    raise Exception(f"no component named backbone found? ( found [{model.layers}] )")