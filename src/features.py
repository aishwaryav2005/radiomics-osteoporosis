"""Pre-computed frozen-backbone feature bank.

Why this exists
---------------
This machine has no CUDA GPU (Intel UHD integrated graphics only), so every
forward pass runs on a 15 W mobile CPU. Training 5 architectures x 5 folds
end-to-end through VGG19 + InceptionResNetV2 + MobileNetV2 is not tractable
here.

The pipeline therefore uses the *frozen-backbone transfer-learning* protocol
(stage 1 of the two-stage strategy): the ImageNet backbones are used purely as
fixed feature extractors and only the fusion head is trained. Because the
backbones are frozen and the augmentation set is fixed and deterministic, the
backbone output for a given (image, augmentation-variant) pair is a constant.
Computing it once and reusing it is *mathematically identical* to running the
frozen backbone inside the training loop every epoch - it is a caching
optimisation, not an approximation.

This is recorded honestly in the report: results correspond to frozen-backbone
feature fusion, not to end-to-end fine-tuning. ``src/models.py`` can still build
the full end-to-end graph (used for parameter counts, FLOPs, Grad-CAM and
optional fine-tuning).

Layout on disk
--------------
``cache/feature_index.csv``            row order (relpath, label, group, ...)
``cache/feat_{backbone}_v{k}.npy``     float32 [n_images, feature_dim]
``cache/feature_bank_meta.json``       shapes, dims, timings, versions
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .preprocessing import augment_image, get_preprocess_fn
from .utils import banner, get_logger, read_json, write_json

log = get_logger("features")


def feature_path(backbone: str, variant: int) -> Path:
    return config.CACHE_DIR / f"feat_{backbone}_v{variant}.npy"


INDEX_PATH = config.CACHE_DIR / "feature_index.csv"
META_PATH = config.CACHE_DIR / "feature_bank_meta.json"


# --------------------------------------------------------------------------
# Backbone construction
# --------------------------------------------------------------------------
def build_backbone(backbone: str, trainable: bool = False):
    """Return a frozen ``include_top=False`` backbone with global average pooling."""
    from tensorflow.keras.applications import (
        InceptionResNetV2, MobileNetV2, VGG19,
    )

    shape = (*config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    ctor = {
        "vgg19": VGG19,
        "inceptionresnetv2": InceptionResNetV2,
        "mobilenetv2": MobileNetV2,
    }[backbone]

    model = ctor(include_top=False, weights="imagenet", input_shape=shape, pooling="avg")
    model.trainable = trainable
    return model


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
def build_index() -> pd.DataFrame:
    """Row order for the feature bank: every unique image in the binary task."""
    manifest = pd.read_csv(config.METADATA_DIR / "dataset_manifest.csv")
    df = manifest.loc[manifest["usable_binary"] == True].copy()  # noqa: E712
    df = df.sort_values("relpath").reset_index(drop=True)
    df["row"] = np.arange(len(df))
    df["label"] = df["class"].map(config.BINARY_LABELS).astype(int)
    cols = ["row", "relpath", "filepath", "dataset_source", "class", "label",
            "patient_id", "group_id", "group_kind"]
    out = df[cols]
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(INDEX_PATH, index=False, encoding="utf-8")
    return out


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        return build_index()
    return pd.read_csv(INDEX_PATH)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _extract_one(model, preprocess, paths: list[str], row_ids: list[int],
                 variant: int, batch_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    total = len(paths)
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        chunk_paths = paths[start:start + batch_size]
        chunk_rows = row_ids[start:start + batch_size]
        batch = np.stack([
            augment_image(p, int(r), variant)
            for p, r in zip(chunk_paths, chunk_rows)
        ])
        batch = preprocess(batch.copy())
        # Direct call rather than .predict(): avoids per-batch retracing when the
        # final batch is short, and skips the Trainer wrapper overhead.
        feats.append(np.asarray(model(batch, training=False)))
        done = min(start + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            log.info("    %4d/%4d  (%.1f img/s, eta %.0fs)", done, total, rate, eta)
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_feature_bank(force: bool = False,
                         backbones: list[str] | None = None,
                         n_variants: int | None = None) -> dict:
    """Compute and cache backbone features for every image and variant."""
    banner("STAGE: FEATURE BANK EXTRACTION (frozen ImageNet backbones)")
    config.ensure_dirs()

    backbones = backbones or config.BACKBONES
    n_variants = config.N_AUG_VARIANTS if n_variants is None else n_variants

    index = build_index()
    paths = index["filepath"].tolist()
    rows = index["row"].tolist()
    log.info("Feature bank covers %d unique images x %d variants (0..%d)",
             len(index), n_variants + 1, n_variants)

    meta = read_json(META_PATH, default={}) or {}
    meta.setdefault("backbones", {})
    meta["n_images"] = len(index)
    meta["n_variants"] = n_variants + 1
    meta["image_size"] = list(config.IMAGE_SIZE)
    meta["augmentation_seed"] = config.AUGMENTATION_SEED

    for backbone in backbones:
        needed = [v for v in range(n_variants + 1)
                  if force or not feature_path(backbone, v).exists()]
        if not needed:
            log.info("%s: all %d variants already cached, skipping.",
                     backbone, n_variants + 1)
            continue

        log.info("Building %s ...", config.BACKBONE_DISPLAY[backbone])
        model = build_backbone(backbone)
        preprocess = get_preprocess_fn(backbone)
        dim = int(model.output_shape[-1])
        log.info("%s feature dimension: %d (config expected %d)",
                 config.BACKBONE_DISPLAY[backbone], dim,
                 config.BACKBONE_FEATURE_DIMS[backbone])

        entry = meta["backbones"].setdefault(backbone, {})
        entry["feature_dim"] = dim
        entry["expected_dim"] = config.BACKBONE_FEATURE_DIMS[backbone]
        entry["dim_matches_paper"] = dim == config.BACKBONE_FEATURE_DIMS[backbone]
        entry["params"] = int(model.count_params())
        entry.setdefault("extraction_seconds", {})

        for variant in needed:
            log.info("  %s variant %d/%d ...", backbone, variant, n_variants)
            t0 = time.perf_counter()
            feats = _extract_one(model, preprocess, paths, rows, variant,
                                 config.FEATURE_EXTRACTION_BATCH_SIZE)
            elapsed = time.perf_counter() - t0
            np.save(feature_path(backbone, variant), feats)
            entry["extraction_seconds"][str(variant)] = round(elapsed, 2)
            log.info("  saved %s  shape=%s  (%.1fs)",
                     feature_path(backbone, variant).name, feats.shape, elapsed)
            write_json(META_PATH, meta)

        del model
        import gc

        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()

    total_dim = sum(meta["backbones"][b]["feature_dim"]
                    for b in config.BACKBONES if b in meta["backbones"])
    meta["fused_feature_dim_all_three"] = total_dim
    write_json(META_PATH, meta)
    log.info("Fused feature dimension (all three backbones): %d", total_dim)
    return meta


# --------------------------------------------------------------------------
# Consumption
# --------------------------------------------------------------------------
class FeatureBank:
    """Random-access view over the cached features."""

    def __init__(self, backbones: list[str], n_variants: int | None = None):
        self.backbones = backbones
        self.index = load_index()
        self.row_of = {r: i for i, r in enumerate(self.index["relpath"])}
        self.n_variants = (config.N_AUG_VARIANTS if n_variants is None else n_variants) + 1
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self.dims = {}
        for b in backbones:
            arr = self._get(b, 0)
            self.dims[b] = arr.shape[1]
        self.dim = sum(self.dims[b] for b in backbones)

    def _get(self, backbone: str, variant: int) -> np.ndarray:
        key = (backbone, variant)
        if key not in self._cache:
            path = feature_path(backbone, variant)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing cached features: {path}. Run `--stage features`."
                )
            self._cache[key] = np.load(path, mmap_mode="r")
        return self._cache[key]

    def vectors(self, relpaths: list[str], variant: int = 0) -> np.ndarray:
        rows = np.array([self.row_of[p] for p in relpaths], dtype=np.int64)
        parts = [np.asarray(self._get(b, variant)[rows]) for b in self.backbones]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def training_matrix(self, relpaths: list[str], labels: np.ndarray,
                        use_augmentation: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Stack the original plus every cached augmented variant."""
        n_var = self.n_variants if use_augmentation else 1
        xs, ys = [], []
        for v in range(n_var):
            xs.append(self.vectors(relpaths, variant=v))
            ys.append(labels)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


if __name__ == "__main__":
    extract_feature_bank()
