"""Five-fold training of every ablation configuration.

25 runs = 5 architectures (A1..A5) x 5 folds, all reading the *same* fixed folds
from ``splits/`` so that model comparisons are paired.

Robustness
----------
* Each run records its state in ``results/status.json``; completed runs are
  skipped on restart, so a crash never costs more than the run in flight.
* Failures are caught per run, written to the status file and to
  ``results/experiment_status.csv``, and the sweep continues. Nothing is
  silently swallowed - a failed run stays visible as ``failed`` with its
  traceback.
* NaN losses terminate the run via ``TerminateOnNaN`` and are reported.

What is trained
---------------
The ImageNet backbones are frozen; only the fusion head is trained, on the
cached feature bank (see ``src/features.py`` for why). Training data uses the
original plus the cached deterministic augmented variants; validation and test
use un-augmented features only.
"""
from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import FeatureBank
from .metrics import compute_metrics, verify_against_confusion_matrix
from .models import build_fusion_head, count_parameters, describe_freezing
from .utils import (
    banner, get_logger, is_completed, process_peak_memory_mb, read_json,
    set_all_seeds, set_status, write_json,
)

log = get_logger("train")


def run_key(arch: str, fold: int) -> str:
    return f"{arch}_fold{fold}"


def run_dir(arch: str, fold: int) -> Path:
    return config.MODELS_DIR / arch / f"fold_{fold}"


def load_fold(fold: int) -> dict[str, pd.DataFrame]:
    base = config.SPLITS_DIR / f"fold_{fold}"
    return {
        "train": pd.read_csv(base / "train.csv"),
        "validation": pd.read_csv(base / "validation.csv"),
        "test": pd.read_csv(base / "test.csv"),
    }


def train_one(arch: str, fold: int, bank: FeatureBank,
              max_epochs: int | None = None,
              use_augmentation: bool = True,
              force: bool = False) -> dict:
    """Train and evaluate one (architecture, fold) run."""
    import tensorflow as tf
    from tensorflow.keras import callbacks

    key = run_key(arch, fold)
    out = run_dir(arch, fold)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"

    if not force and is_completed(key) and result_path.exists():
        log.info("%s already completed - skipping.", key)
        return read_json(result_path)

    set_all_seeds(config.GLOBAL_SEED + fold)
    set_status(key, "running")
    cfg = config.ABLATION_CONFIGS[arch]
    max_epochs = max_epochs or config.MAX_EPOCHS

    splits = load_fold(fold)
    y = {k: v["label"].to_numpy().astype(int) for k, v in splits.items()}
    paths = {k: v["relpath"].tolist() for k, v in splits.items()}

    x_train, y_train = bank.training_matrix(
        paths["train"], y["train"], use_augmentation=use_augmentation)
    x_val = bank.vectors(paths["validation"], variant=0)
    x_test = bank.vectors(paths["test"], variant=0)

    # Standardise the fused feature vector. The three backbones emit GAP
    # activations on quite different scales, so raw concatenation would let the
    # largest-magnitude branch dominate the fusion - which would confound the
    # very thing the ablation measures. Statistics are fitted on TRAINING
    # features only and then applied to validation/test, so no information
    # crosses the split.
    feat_mean = x_train.mean(axis=0, keepdims=True)
    feat_std = x_train.std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0
    x_train = (x_train - feat_mean) / feat_std
    x_val = (x_val - feat_mean) / feat_std
    x_test = (x_test - feat_mean) / feat_std
    np.savez_compressed(out / "feature_scaler.npz", mean=feat_mean, std=feat_std)

    log.info("%s | features %d-d | train %d (x%d aug) | val %d | test %d",
             key, bank.dim, len(paths["train"]),
             bank.n_variants if use_augmentation else 1,
             len(paths["validation"]), len(paths["test"]))

    model = build_fusion_head(bank.dim)
    params = count_parameters(model)

    ckpt = out / "best_model.keras"
    cbs = [
        callbacks.ModelCheckpoint(str(ckpt), monitor=config.MONITOR_METRIC,
                                  save_best_only=True, mode="min", verbose=0),
        callbacks.EarlyStopping(monitor=config.MONITOR_METRIC,
                                patience=config.EARLY_STOPPING_PATIENCE,
                                restore_best_weights=True, mode="min", verbose=0),
        callbacks.ReduceLROnPlateau(monitor=config.MONITOR_METRIC,
                                    factor=config.REDUCE_LR_FACTOR,
                                    patience=config.REDUCE_LR_PATIENCE,
                                    min_lr=config.MIN_LEARNING_RATE, verbose=0),
        callbacks.TerminateOnNaN(),
        callbacks.CSVLogger(str(out / "training_log.csv")),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y["validation"]),
        epochs=max_epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=cbs,
        verbose=0,
        shuffle=True,
    )
    train_seconds = time.perf_counter() - t0

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    if any(not np.isfinite(v).all() for v in hist.values()):
        raise FloatingPointError(f"{key}: non-finite values in training history")
    write_json(out / "history.json", hist)

    # -- inference timing on the head (measured) --------------------------
    _ = model.predict(x_test[:8], verbose=0)
    t0 = time.perf_counter()
    prob_test = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=0).ravel()
    head_infer_seconds = time.perf_counter() - t0

    prob_val = model.predict(x_val, batch_size=config.BATCH_SIZE, verbose=0).ravel()

    test_metrics = compute_metrics(y["test"], prob_test)
    val_metrics = compute_metrics(y["validation"], prob_val)
    verification = verify_against_confusion_matrix(y["test"], prob_test)
    if not verification["agrees"]:
        raise AssertionError(f"{key}: metric verification failed: {verification}")

    preds = pd.DataFrame({
        "relpath": paths["test"],
        "dataset_source": splits["test"]["dataset_source"],
        "group_id": splits["test"]["group_id"],
        "y_true": y["test"],
        "y_prob": prob_test,
        "y_pred": (prob_test >= config.DECISION_THRESHOLD).astype(int),
        "fold": fold,
        "architecture": arch,
    })
    preds.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame({
        "relpath": paths["validation"], "y_true": y["validation"],
        "y_prob": prob_val, "fold": fold, "architecture": arch,
    }).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8")

    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    result = {
        "architecture": arch,
        "architecture_display": cfg["display"],
        "backbones": cfg["backbones"],
        "fold": fold,
        "feature_dim": int(bank.dim),
        "feature_dims_per_backbone": {b: int(bank.dims[b]) for b in cfg["backbones"]},
        "head_parameters": params,
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(hist["val_loss"])),
        "best_val_accuracy": float(max(hist.get("val_accuracy", [float("nan")]))),
        "final_train_loss": float(hist["loss"][-1]),
        "final_train_accuracy": float(hist.get("accuracy", [float("nan")])[-1]),
        "train_seconds": round(train_seconds, 3),
        "seconds_per_epoch": round(train_seconds / max(1, len(hist["loss"])), 4),
        "head_inference_seconds_total": round(head_infer_seconds, 5),
        "head_inference_ms_per_image": round(1000 * head_infer_seconds / len(x_test), 4),
        "peak_process_memory_mb": process_peak_memory_mb(),
        "n_train_vectors": int(len(x_train)),
        "n_train_images": int(len(paths["train"])),
        "n_validation": int(len(paths["validation"])),
        "n_test": int(len(paths["test"])),
        "test_metrics": test_metrics,
        "validation_metrics": val_metrics,
        "metric_verification": verification,
        "freezing": describe_freezing(cfg["backbones"]),
        "feature_standardisation": {
            "applied": True,
            "fitted_on": "training features only (including augmented variants)",
            "applied_to": ["train", "validation", "test"],
            "rationale": ("backbone GAP activations differ in scale; without this "
                          "the largest-magnitude branch dominates the fusion"),
            "saved_to": "feature_scaler.npz",
        },
        "hyperparameters": {
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "max_epochs": max_epochs,
            "optimizer": config.OPTIMIZER,
            "loss": config.LOSS,
            "dropout": list(config.HEAD_DROPOUT),
            "dense_units": list(config.HEAD_DENSE_UNITS),
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": config.REDUCE_LR_PATIENCE,
            "reduce_lr_factor": config.REDUCE_LR_FACTOR,
            "monitor": config.MONITOR_METRIC,
            "decision_threshold": config.DECISION_THRESHOLD,
            "augmentation_variants": bank.n_variants if use_augmentation else 1,
            "seed": config.GLOBAL_SEED + fold,
        },
    }
    write_json(result_path, result)
    set_status(key, "completed", {
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_auc": round(test_metrics["roc_auc"], 4),
        "train_seconds": round(train_seconds, 1),
    })

    log.info("%s | acc %.4f | f1 %.4f | auc %.4f | sens %.4f | spec %.4f | %d ep | %.1fs",
             key, test_metrics["accuracy"], test_metrics["f1"], test_metrics["roc_auc"],
             test_metrics["sensitivity"], test_metrics["specificity"],
             result["epochs_run"], train_seconds)

    del model, x_train, x_val, x_test
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def run_all_experiments(architectures: list[str] | None = None,
                        folds: list[int] | None = None,
                        max_epochs: int | None = None,
                        use_augmentation: bool = True,
                        force: bool = False) -> dict:
    """Run the full 5 x 5 sweep, skipping completed runs and surviving failures."""
    banner("STAGE: FIVE-FOLD CROSS-VALIDATION TRAINING")
    config.ensure_dirs()

    architectures = architectures or config.ABLATION_ORDER
    folds = folds or list(range(1, config.N_FOLDS + 1))

    # One bank per distinct backbone set, reused across that architecture's folds.
    banks: dict[tuple[str, ...], FeatureBank] = {}
    records, failures = [], []
    total = len(architectures) * len(folds)
    done = 0
    t_start = time.perf_counter()

    for arch in architectures:
        cfg = config.ABLATION_CONFIGS[arch]
        bbk = tuple(cfg["backbones"])
        if bbk not in banks:
            banks[bbk] = FeatureBank(list(bbk))
        bank = banks[bbk]

        for fold in folds:
            done += 1
            key = run_key(arch, fold)
            log.info("[%2d/%2d] %s  (%s)", done, total, key, cfg["display"])
            try:
                res = train_one(arch, fold, bank, max_epochs=max_epochs,
                                use_augmentation=use_augmentation, force=force)
                records.append(res)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("FAILED %s: %s", key, exc)
                log.debug(tb)
                set_status(key, "failed", {"error": f"{type(exc).__name__}: {exc}"})
                failures.append({"experiment": key, "architecture": arch, "fold": fold,
                                 "error_type": type(exc).__name__, "error": str(exc),
                                 "traceback": tb})
                (config.LOGS_DIR / f"error_{key}.txt").write_text(tb, encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    _write_experiment_status(architectures, folds, failures)

    log.info("-" * 70)
    log.info("Completed %d/%d runs in %.1f min (%d failed)",
             len(records), total, elapsed / 60, len(failures))
    return {"completed": len(records), "failed": len(failures),
            "total": total, "elapsed_seconds": elapsed, "failures": failures}


def _write_experiment_status(architectures, folds, failures) -> None:
    rows = []
    fail_map = {f["experiment"]: f for f in failures}
    for arch in architectures:
        for fold in folds:
            key = run_key(arch, fold)
            rp = run_dir(arch, fold) / "results.json"
            res = read_json(rp) if rp.exists() else None
            rows.append({
                "experiment": key,
                "architecture": arch,
                "architecture_display": config.ABLATION_CONFIGS[arch]["display"],
                "fold": fold,
                "status": "failed" if key in fail_map else (
                    "completed" if res else "pending"),
                "test_accuracy": res["test_metrics"]["accuracy"] if res else None,
                "test_f1": res["test_metrics"]["f1"] if res else None,
                "test_auc": res["test_metrics"]["roc_auc"] if res else None,
                "epochs_run": res["epochs_run"] if res else None,
                "train_seconds": res["train_seconds"] if res else None,
                "error_type": fail_map.get(key, {}).get("error_type"),
                "error": fail_map.get(key, {}).get("error"),
            })
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "experiment_status.csv",
                              index=False, encoding="utf-8")
    if failures:
        write_json(config.RESULTS_DIR / "failed_experiments.json", failures)


if __name__ == "__main__":
    run_all_experiments()
