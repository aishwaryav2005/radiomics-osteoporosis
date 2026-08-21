"""Image loading, augmentation and per-backbone preprocessing.

Two things matter here and are deliberately kept separate:

1. **Augmentation** operates in raw 8-bit RGB space and is applied to TRAINING
   images only. It is deterministic: variant *k* of image *i* always yields the
   same pixels, derived from ``AUGMENTATION_SEED``. That makes the whole
   experiment reproducible and lets augmented views be cached.

2. **Backbone preprocessing** is applied *after* augmentation and is different
   for each network. VGG19 expects Caffe-style BGR mean subtraction, while
   InceptionResNetV2 and MobileNetV2 expect inputs scaled to [-1, 1]. Applying
   one shared [0,1] rescale to all three - as a naive shared pipeline would -
   silently feeds two of the three backbones out-of-distribution inputs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

from . import config


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_image(path: str, size: tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """Load an image as an RGB float32 array in [0, 255] at ``size``."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32)


# --------------------------------------------------------------------------
# Deterministic offline augmentation
# --------------------------------------------------------------------------
def _rng_for(image_index: int, variant: int) -> np.random.Generator:
    seed = (config.AUGMENTATION_SEED * 1_000_003
            + image_index * 7919 + variant * 104_729) % (2**32)
    return np.random.default_rng(seed)


def augment_image(path: str, image_index: int, variant: int,
                  size: tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """Return augmented variant ``variant`` of the image at ``path``.

    ``variant == 0`` is the identity transform (the un-augmented image), so
    validation and test data can reuse this function without ever being
    augmented.
    """
    if variant == 0:
        return load_image(path, size)

    rng = _rng_for(image_index, variant)

    with Image.open(path) as im:
        img = im.convert("RGB")

        # Work at a slightly larger canvas so rotation/zoom do not bake in
        # black corners at the target resolution.
        work = img.resize((int(size[0] * 1.25), int(size[1] * 1.25)),
                          Image.Resampling.BILINEAR)

        if config.AUG_HORIZONTAL_FLIP and rng.random() < 0.5:
            work = work.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        angle = float(rng.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG))
        work = work.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

        zoom = float(rng.uniform(*config.AUG_ZOOM_RANGE))
        w, h = work.size
        cw, ch = int(w / zoom), int(h / zoom)
        cw, ch = min(cw, w), min(ch, h)
        left = (w - cw) // 2
        top = (h - ch) // 2
        work = work.crop((left, top, left + cw, top + ch))

        work = work.resize(size, Image.Resampling.BILINEAR)

        brightness = float(rng.uniform(*config.AUG_BRIGHTNESS_RANGE))
        work = ImageEnhance.Brightness(work).enhance(brightness)

        contrast = float(rng.uniform(*config.AUG_CONTRAST_RANGE))
        work = ImageEnhance.Contrast(work).enhance(contrast)

        return np.asarray(work, dtype=np.float32)


# --------------------------------------------------------------------------
# Backbone-specific preprocessing
# --------------------------------------------------------------------------
def get_preprocess_fn(backbone: str):
    """Return the official Keras ``preprocess_input`` for a backbone."""
    from tensorflow.keras.applications import (
        inception_resnet_v2 as _irv2,
        mobilenet_v2 as _mnv2,
        vgg19 as _vgg19,
    )

    table = {
        "vgg19": _vgg19.preprocess_input,
        "inceptionresnetv2": _irv2.preprocess_input,
        "mobilenetv2": _mnv2.preprocess_input,
    }
    if backbone not in table:
        raise KeyError(f"Unknown backbone: {backbone}")
    return table[backbone]


def preprocessing_description() -> dict:
    """Machine-readable record of exactly what was applied, for the report."""
    return {
        "resize": f"{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]} bilinear",
        "colour": "RGB (grayscale radiographs replicated to 3 channels)",
        "backbone_preprocessing": {
            "vgg19": "keras.applications.vgg19.preprocess_input (Caffe: RGB->BGR, ImageNet mean subtraction)",
            "inceptionresnetv2": "keras.applications.inception_resnet_v2.preprocess_input (scale to [-1, 1])",
            "mobilenetv2": "keras.applications.mobilenet_v2.preprocess_input (scale to [-1, 1])",
        },
        "augmentation": {
            "applied_to": "training split only",
            "scheme": "deterministic offline augmentation",
            "variants_per_training_image": config.N_AUG_VARIANTS,
            "variant_0": "identity (un-augmented original)",
            "rotation_degrees": f"+/-{config.AUG_ROTATION_DEG}",
            "horizontal_flip": config.AUG_HORIZONTAL_FLIP,
            "zoom_range": list(config.AUG_ZOOM_RANGE),
            "brightness_range": list(config.AUG_BRIGHTNESS_RANGE),
            "contrast_range": list(config.AUG_CONTRAST_RANGE),
            "seed": config.AUGMENTATION_SEED,
        },
        "validation_and_test": "variant 0 only; never augmented",
    }
