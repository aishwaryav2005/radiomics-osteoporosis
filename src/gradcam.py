"""Grad-CAM visualisation for the hybrid model.

Scope note, stated deliberately: Grad-CAM shows which image regions most
influenced this model's output. It is a *model interpretability / visual
explanation* technique. It is not evidence that the model reasons like a
clinician, and it does not constitute clinical validation of the features it
highlights. The figures produced here are captioned accordingly.

For the multi-branch hybrid, one map is produced per backbone branch using that
branch's final convolutional layer, plus a combined view. Using a single
"final conv layer" for a three-branch fusion model would be wrong - each branch
has its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .preprocessing import load_image
from .utils import banner, get_logger, read_json

log = get_logger("gradcam")

# Final convolutional / activation layer of each backbone.
FINAL_CONV = {
    "vgg19": "block5_conv4",
    "inceptionresnetv2": "conv_7b_ac",
    "mobilenetv2": "out_relu",
}


def _load_trained_head(arch: str, fold: int):
    from tensorflow.keras.models import load_model

    from .train import run_dir

    p = run_dir(arch, fold) / "best_model.keras"
    if not p.exists():
        raise FileNotFoundError(f"No trained head at {p}")
    return load_model(p)


def build_gradcam_model(arch: str, fold: int):
    """End-to-end model exposing each branch's conv maps, with the trained head.

    ``expose_conv_maps=True`` puts the final convolutional activation of every
    backbone into the *outer* graph, so a single gradient tape can differentiate
    the prediction with respect to each of them. Without that, the conv tensors
    live inside a nested sub-model and are not reachable from the outer input.
    """
    from .models import build_end_to_end_model
    from .train import run_dir

    cfg = config.ABLATION_CONFIGS[arch]

    scaler_path = run_dir(arch, fold) / "feature_scaler.npz"
    scaler = None
    if scaler_path.exists():
        z = np.load(scaler_path)
        scaler = (z["mean"], z["std"])
    else:
        log.warning("No feature scaler at %s; Grad-CAM predictions may not match "
                    "the trained head.", scaler_path)

    model = build_end_to_end_model(cfg["backbones"], expose_conv_maps=True,
                                   feature_scaler=scaler)
    head = _load_trained_head(arch, fold)

    transferred = 0
    for layer in model.layers:
        if layer.name.startswith("fusion_") or layer.name == "prediction":
            layer.set_weights(head.get_layer(layer.name).get_weights())
            transferred += 1
    log.info("%s fold %d: transferred %d trained layers into the end-to-end model",
             arch, fold, transferred)
    return model


def gradcam_for_prediction(model, image_path: str, backbones: list[str]) -> dict:
    """Class-discriminative Grad-CAM for every branch, plus the prediction.

    The gradient is taken of the *osteoporosis logit* with respect to each
    branch's final conv activation, which is what makes the map class-specific.
    A generic activation map would not be Grad-CAM.
    """
    import tensorflow as tf
    from PIL import Image

    raw = load_image(image_path)                      # raw RGB in [0, 255]
    batch = tf.convert_to_tensor(raw[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        outputs = model(batch, training=False)
        prediction, conv_maps = outputs[0], list(outputs[1:])
        for cm in conv_maps:
            tape.watch(cm)
        # Re-run so the watched tensors are on the tape's path.
        score = prediction[:, 0]
    grads = tape.gradient(score, conv_maps)

    maps: dict[str, np.ndarray] = {}
    for b, conv, grad in zip(backbones, conv_maps, grads):
        if grad is None:
            log.warning("No gradient for branch %s on %s", b, Path(image_path).name)
            maps[b] = np.zeros(config.IMAGE_SIZE, dtype=np.float32)
            continue
        weights = tf.reduce_mean(grad, axis=(1, 2))                  # channel weights
        cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)
        cam = tf.nn.relu(cam).numpy()[0]
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            config.IMAGE_SIZE, Image.Resampling.BILINEAR)
        maps[b] = np.asarray(cam_img, dtype=np.float32) / 255.0

    return {
        "image": raw / 255.0,
        "maps": maps,
        "probability": float(np.asarray(prediction).ravel()[0]),
    }


def generate_gradcam(arch: str = config.FULL_MODEL, n_per_class: int = 3) -> dict:
    banner("STAGE: GRAD-CAM INTERPRETABILITY")
    config.ensure_dirs()

    import matplotlib.pyplot as plt

    from .figures import apply_style, save

    apply_style()

    fold_metrics = pd.read_csv(config.CV_DIR / "fold_metrics.csv")
    sub = fold_metrics[fold_metrics["architecture"] == arch]
    if sub.empty:
        log.warning("No results for %s; skipping Grad-CAM.", arch)
        return {"generated": 0}
    fold = int(sub.loc[sub["accuracy"].idxmax(), "fold"])
    cfg = config.ABLATION_CONFIGS[arch]

    try:
        model = build_gradcam_model(arch, fold)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not build Grad-CAM model: %s", exc)
        return {"generated": 0, "error": str(exc)}

    from .train import run_dir

    preds = pd.read_csv(run_dir(arch, fold) / "test_predictions.csv")
    manifest = pd.read_csv(config.METADATA_DIR / "dataset_manifest.csv")
    path_of = dict(zip(manifest["relpath"], manifest["filepath"]))

    chosen = []
    for label, cname in ((0, "Normal"), (1, "Osteoporosis")):
        s = preds[(preds["y_true"] == label) & (preds["y_pred"] == label)]
        s = s.reindex(s["y_prob"].sub(label).abs().sort_values().index)
        chosen += [(r["relpath"], cname, r["y_prob"]) for _, r in s.head(n_per_class).iterrows()]

    backbones = cfg["backbones"]
    ncols = len(backbones) + 1
    fig, axes = plt.subplots(len(chosen), ncols, figsize=(3.5 * ncols, 3.4 * len(chosen)))
    axes = np.atleast_2d(axes)
    generated = 0

    for r, (relpath, cname, prob) in enumerate(chosen):
        fp = path_of.get(relpath)
        if fp is None:
            continue
        res = gradcam_for_prediction(model, fp, backbones)
        axes[r, 0].imshow(res["image"])
        axes[r, 0].set_ylabel(f"{cname}\np(osteoporosis)={prob:.3f}", fontsize=10)
        axes[r, 0].set_title("Input radiograph" if r == 0 else "", fontsize=11)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        for c, b in enumerate(backbones, start=1):
            axes[r, c].imshow(res["image"])
            axes[r, c].imshow(res["maps"][b], cmap="jet", alpha=0.45)
            axes[r, c].set_title(config.BACKBONE_DISPLAY[b] if r == 0 else "", fontsize=11)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        generated += 1

    fig.suptitle(
        f"Grad-CAM visual explanations — {arch} ({cfg['display']}), fold {fold}\n"
        "Model interpretability only; not clinical validation of the highlighted regions",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, config.GRADCAM_DIR / f"gradcam_{arch}_fold{fold}")

    log.info("Grad-CAM generated for %d images", generated)
    log.warning("Grad-CAM shows model attribution, not clinical reasoning. "
                "Do not describe it as clinical interpretability validation.")
    return {"generated": generated, "architecture": arch, "fold": fold,
            "final_conv_layers": {b: FINAL_CONV[b] for b in backbones},
            "caveat": ("Grad-CAM is a visual explanation of model attribution; "
                       "it is not evidence of clinically valid reasoning.")}


if __name__ == "__main__":
    generate_gradcam()
