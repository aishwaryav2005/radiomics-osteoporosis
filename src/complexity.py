"""Computational complexity of each ablation configuration.

Every number carries an explicit provenance tag, because conflating these is a
common reporting error:

``calculated``  derived exactly from the graph - parameter counts (sum of weight
                tensor sizes) and FLOPs (TensorFlow graph profiler over a frozen
                concrete function).
``measured``    timed or observed on this machine - inference latency, training
                time, peak process memory, serialised model size on disk.
``estimated``   not used in this module; no value here is a guess.

FLOPs are reported as the profiler's ``float_operation`` count. That counter
reports multiply-accumulate work as two operations, so the value is comparable
to the "FLOPs" convention used by most papers; it is halved to give MACs where
that convention is wanted. This is stated rather than left ambiguous.

All timings are CPU timings on a machine with no CUDA GPU, and are not
comparable with GPU numbers reported elsewhere.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .models import build_end_to_end_model, count_parameters
from .utils import (
    banner, get_logger, process_peak_memory_mb, read_json, write_json,
)

log = get_logger("complexity")


def compute_flops(model) -> dict:
    """Exact graph FLOPs via the TF profiler on a frozen concrete function."""
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )

        shape = (1, *config.IMAGE_SIZE, config.IMAGE_CHANNELS)

        @tf.function
        def fn(x):
            return model(x, training=False)

        concrete = fn.get_concrete_function(tf.TensorSpec(shape, tf.float32))
        frozen = convert_variables_to_constants_v2(concrete)

        run_meta = tf.compat.v1.RunMetadata()
        opts = (tf.compat.v1.profiler.ProfileOptionBuilder
                .float_operation())
        opts["output"] = "none"
        flops = tf.compat.v1.profiler.profile(
            graph=frozen.graph, run_meta=run_meta, cmd="op", options=opts)
        total = int(flops.total_float_ops) if flops else 0
        return {
            "flops": total,
            "gflops": round(total / 1e9, 4),
            "gmacs": round(total / 2e9, 4),
            "provenance": "calculated",
            "method": ("tf.compat.v1.profiler float_operation over the frozen "
                       "concrete function, batch size 1, 224x224x3"),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("FLOPs profiling unavailable: %s", exc)
        return {"flops": None, "gflops": None, "gmacs": None,
                "provenance": "unavailable", "method": f"failed: {exc}"}


def measure_inference(model, n_warmup: int = 3, n_runs: int = 12) -> dict:
    """Single-image inference latency, measured end-to-end on this CPU."""
    shape = (1, *config.IMAGE_SIZE, config.IMAGE_CHANNELS)
    rng = np.random.default_rng(config.GLOBAL_SEED)
    x = rng.standard_normal(shape).astype(np.float32)

    for _ in range(n_warmup):
        _ = model(x, training=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(x, training=False)
        times.append(time.perf_counter() - t0)
    t = np.asarray(times) * 1000.0

    # Throughput at a realistic batch size.
    xb = rng.standard_normal((config.FEATURE_EXTRACTION_BATCH_SIZE, *shape[1:])).astype(np.float32)
    _ = model(xb, training=False)
    t0 = time.perf_counter()
    _ = model(xb, training=False)
    batch_s = time.perf_counter() - t0

    return {
        "inference_ms_per_image": round(float(t.mean()), 3),
        "inference_ms_std": round(float(t.std(ddof=1)), 3),
        "inference_ms_median": round(float(np.median(t)), 3),
        "batched_ms_per_image": round(
            1000 * batch_s / config.FEATURE_EXTRACTION_BATCH_SIZE, 3),
        "batch_size_for_throughput": config.FEATURE_EXTRACTION_BATCH_SIZE,
        "n_runs": n_runs,
        "provenance": "measured",
        "device": "CPU (no CUDA GPU present)",
    }


def measure_model_size(model, tmp_dir: Path) -> dict:
    """Serialised size on disk, measured."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{model.name}.keras"
    try:
        model.save(path)
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        return {"model_size_mb": round(size / 1024**2, 2), "provenance": "measured",
                "method": "serialised .keras file size on disk"}
    except Exception as exc:  # noqa: BLE001
        params = count_parameters(model)
        return {
            "model_size_mb": round(params["total_params"] * 4 / 1024**2, 2),
            "provenance": "calculated",
            "method": f"params x 4 bytes (float32); serialisation failed: {exc}",
        }


def analyse_complexity(architectures: list[str] | None = None) -> pd.DataFrame:
    banner("STAGE: COMPUTATIONAL COMPLEXITY")
    config.ensure_dirs()

    architectures = architectures or config.ABLATION_ORDER
    fold_path = config.CV_DIR / "fold_metrics.csv"
    fold_metrics = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()

    rows = []
    for arch in architectures:
        cfg = config.ABLATION_CONFIGS[arch]
        log.info("Profiling %s (%s) ...", arch, cfg["display"])

        import tensorflow as tf
        tf.keras.backend.clear_session()

        model = build_end_to_end_model(cfg["backbones"])
        params = count_parameters(model)
        flops = compute_flops(model)
        timing = measure_inference(model)
        size = measure_model_size(model, config.CACHE_DIR / "tmp_models")

        row = {
            "config": arch,
            "architecture": cfg["display"],
            "n_branches": len(cfg["backbones"]),
            "backbones": " + ".join(config.BACKBONE_DISPLAY[b] for b in cfg["backbones"]),
            "fused_feature_dim": sum(config.BACKBONE_FEATURE_DIMS[b]
                                     for b in cfg["backbones"]),
            "total_params": params["total_params"],
            "trainable_params": params["trainable_params"],
            "non_trainable_params": params["non_trainable_params"],
            "total_params_millions": round(params["total_params"] / 1e6, 3),
            "trainable_params_millions": round(params["trainable_params"] / 1e6, 4),
            "params_provenance": "calculated",
            "flops": flops["flops"],
            "gflops": flops["gflops"],
            "gmacs": flops["gmacs"],
            "flops_provenance": flops["provenance"],
            "flops_method": flops["method"],
            "model_size_mb": size["model_size_mb"],
            "model_size_provenance": size["provenance"],
            "inference_ms_per_image": timing["inference_ms_per_image"],
            "inference_ms_std": timing["inference_ms_std"],
            "batched_ms_per_image": timing["batched_ms_per_image"],
            "inference_provenance": "measured",
            "peak_process_memory_mb": process_peak_memory_mb(),
            "memory_provenance": "measured (process RSS, CPU; no GPU memory to report)",
            "device": "CPU",
        }

        if not fold_metrics.empty:
            sub = fold_metrics[fold_metrics["architecture"] == arch]
            if not sub.empty:
                row["head_train_seconds_mean"] = round(float(sub["train_seconds"].mean()), 2)
                row["head_train_seconds_std"] = round(float(sub["train_seconds"].std(ddof=1)), 2)
                row["head_seconds_per_epoch_mean"] = round(
                    float(sub["seconds_per_epoch"].mean()), 4)
                row["head_inference_ms_per_image"] = round(
                    float(sub["head_inference_ms_per_image"].mean()), 4)
                row["training_time_provenance"] = "measured"

        rows.append(row)
        log.info("   params %.2fM | %s GFLOPs | %.1f MB | %.1f ms/img",
                 row["total_params_millions"], row["gflops"],
                 row["model_size_mb"], row["inference_ms_per_image"])

        del model
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df.to_csv(config.COMPLEXITY_DIR / "computational_complexity.csv",
              index=False, encoding="utf-8")
    df.to_csv(config.RESULTS_DIR / "computational_complexity.csv",
              index=False, encoding="utf-8")

    env = read_json(config.RESULTS_DIR / "environment.json", default={}) or {}
    write_json(config.COMPLEXITY_DIR / "complexity.json", {
        "rows": df.to_dict(orient="records"),
        "provenance_legend": {
            "calculated": "derived exactly from the model graph (parameters, FLOPs)",
            "measured": "timed or observed on this machine (latency, size, memory)",
            "estimated": "not used - no value in this table is an estimate",
        },
        "hardware": {
            "device": "CPU only",
            "processor": env.get("processor"),
            "logical_cpus": env.get("cpu_count_logical"),
            "ram_gb": env.get("total_ram_gb"),
            "gpu_available": env.get("gpu_available", False),
        },
        "caveat": ("All latency and training-time figures are CPU measurements on "
                   "a mobile 15 W processor and are not comparable to GPU timings."),
        "training_time_note": (
            "head_train_seconds_* covers training the fusion head on cached frozen "
            "backbone features. It excludes the one-off feature-extraction pass, "
            "which is reported separately in cache/feature_bank_meta.json."
        ),
    })
    return df


if __name__ == "__main__":
    analyse_complexity()
