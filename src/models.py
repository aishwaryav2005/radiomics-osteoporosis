"""Model construction for the hybrid feature-fusion architectures.

Two equivalent views of the same architecture are provided:

``build_fusion_head``
    The trainable part only, taking a pre-fused feature vector. This is what
    the cross-validation actually trains, on top of the cached frozen-backbone
    features (see ``src/features.py``).

``build_end_to_end_model``
    The complete graph: image -> N frozen backbones -> GAP -> concatenate ->
    same fusion head -> sigmoid. Used for parameter counts, FLOPs, model size,
    inference timing and Grad-CAM, and available for optional fine-tuning.

Both share ``_fusion_layers`` so the head trained on cached features is exactly
the head inside the end-to-end model; weights transfer directly.
"""
from __future__ import annotations

from typing import Sequence

from . import config


def _fusion_layers(x, dropout: Sequence[float] = config.HEAD_DROPOUT,
                   units: Sequence[int] = config.HEAD_DENSE_UNITS,
                   name_prefix: str = "fusion"):
    """Dense -> activation -> dropout stack ending in a single sigmoid unit."""
    from tensorflow.keras import layers

    for i, (n, p) in enumerate(zip(units, dropout), start=1):
        x = layers.Dense(n, activation=config.HEAD_ACTIVATION,
                         name=f"{name_prefix}_dense_{i}")(x)
        x = layers.Dropout(p, name=f"{name_prefix}_dropout_{i}")(x)
    return layers.Dense(1, activation=config.OUTPUT_ACTIVATION,
                        name="prediction")(x)


def build_fusion_head(input_dim: int, learning_rate: float = config.LEARNING_RATE):
    """Trainable classifier over an already-fused feature vector."""
    from tensorflow.keras import Input, Model, optimizers

    inp = Input(shape=(input_dim,), name="fused_features")
    out = _fusion_layers(inp)
    model = Model(inp, out, name="fusion_head")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=config.LOSS,
        metrics=["accuracy"],
    )
    return model


def build_end_to_end_model(backbones: Sequence[str],
                           learning_rate: float = config.LEARNING_RATE,
                           trainable_backbones: bool = False,
                           expose_conv_maps: bool = False,
                           feature_scaler: tuple | None = None):
    """Full image-to-prediction graph for the given backbone combination.

    The model takes a **raw RGB image in [0, 255]** and applies each backbone's
    own ``preprocess_input`` inside the graph. This matters: VGG19 expects
    Caffe-style BGR mean subtraction while InceptionResNetV2 and MobileNetV2
    expect [-1, 1]. Feeding one shared tensor to all three would put two of the
    three backbones out of distribution.

    With ``expose_conv_maps=True`` the model also returns each branch's final
    convolutional activation, which is what Grad-CAM differentiates against.
    """
    from tensorflow.keras import Input, Model, layers, optimizers

    from .features import build_backbone
    from .preprocessing import get_preprocess_fn

    shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    image_in = Input(shape=shape, name="image")

    branch_outputs, conv_maps = [], []
    for name in backbones:
        base = build_backbone(name, trainable=trainable_backbones)
        base._name = f"backbone_{name}"

        pre = layers.Lambda(get_preprocess_fn(name), name=f"preprocess_{name}")(image_in)

        if expose_conv_maps:
            from .gradcam import FINAL_CONV

            conv_layer = base.get_layer(FINAL_CONV[name])
            branch = Model(base.input, [conv_layer.output, base.output],
                           name=f"branch_{name}")
            conv_out, pooled = branch(pre)
            conv_maps.append(layers.Identity(name=f"convmap_{name}")(conv_out))
            branch_outputs.append(pooled)
        else:
            branch_outputs.append(base(pre))

    fused = (branch_outputs[0] if len(branch_outputs) == 1
             else layers.Concatenate(name="feature_fusion")(branch_outputs))

    # The fusion head is trained on standardised features (see src/train.py), so
    # the end-to-end graph must apply the same transform or its predictions
    # would not match the trained head.
    if feature_scaler is not None:
        import numpy as np

        mean, std = (np.asarray(a, dtype="float32").ravel() for a in feature_scaler)
        norm = layers.Normalization(axis=-1, mean=mean, variance=std ** 2,
                                    name="feature_standardisation")
        fused = norm(fused)

    out = _fusion_layers(fused)

    outputs = [out, *conv_maps] if expose_conv_maps else out
    model = Model(image_in, outputs, name=f"hybrid_{'_'.join(backbones)}")
    if not expose_conv_maps:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=config.LOSS,
            metrics=["accuracy"],
        )
    return model


def count_parameters(model) -> dict:
    """Total / trainable / non-trainable parameter counts (calculated exactly)."""
    import numpy as np

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def describe_freezing(backbones: Sequence[str], trainable_backbones: bool = False) -> dict:
    """Explicit record of which layers were frozen, for the report."""
    return {
        "strategy": ("stage 1 only: all ImageNet backbone layers frozen, "
                     "fusion head trained from scratch"),
        "backbones_frozen": list(backbones) if not trainable_backbones else [],
        "backbones_trainable": [] if not trainable_backbones else list(backbones),
        "trainable_components": [
            f"fusion_dense_{i}" for i in range(1, len(config.HEAD_DENSE_UNITS) + 1)
        ] + ["prediction"],
        "backbone_weights": "ImageNet (keras.applications pretrained), not re-trained",
        "stage_2_finetuning": (
            "implemented in build_end_to_end_model(trainable_backbones=True) but "
            "NOT executed in the reported results - see results/final_report.md"
        ),
    }
