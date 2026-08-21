"""Environment / hardware verification and recording.

Writes ``results/environment.json`` and ``environment.txt`` and runs a small
throughput benchmark so the report can state honestly how fast this machine is
and whether any GPU acceleration was actually available.
"""
from __future__ import annotations

import subprocess
import sys
import time

from . import config
from .utils import banner, collect_environment, get_logger, write_json

log = get_logger("env")


def benchmark_cpu(n: int = 512, reps: int = 20) -> dict:
    import numpy as np
    import tensorflow as tf

    a = tf.random.normal((n, n))
    b = tf.random.normal((n, n))
    _ = tf.matmul(a, b).numpy()          # warm up
    t0 = time.perf_counter()
    for _ in range(reps):
        c = tf.matmul(a, b)
    _ = c.numpy()
    dt = time.perf_counter() - t0
    gflops = reps * 2 * n ** 3 / dt / 1e9
    return {
        "matmul_size": n,
        "repetitions": reps,
        "seconds": round(dt, 4),
        "effective_gflops": round(gflops, 2),
        "measurement": "measured",
    }


def check(verbose: bool = True) -> dict:
    banner("STAGE: ENVIRONMENT CHECK")
    config.ensure_dirs()

    info = collect_environment()

    try:
        info["cpu_benchmark"] = benchmark_cpu()
    except Exception as exc:  # noqa: BLE001
        info["cpu_benchmark"] = {"error": str(exc)}

    try:
        info["pip_freeze"] = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        info["pip_freeze"] = [f"<unavailable: {exc}>"]

    info["compute_note"] = (
        "No CUDA-capable GPU is present on this machine; TensorFlow runs on CPU. "
        "All timing figures in this project are therefore CPU timings and are "
        "not comparable to GPU timings reported elsewhere."
    ) if not info.get("gpu_available") else "CUDA GPU detected."

    write_json(config.RESULTS_DIR / "environment.json", info)

    lines = [
        "ENVIRONMENT RECORD",
        "=" * 70,
        f"Python            : {info['python_version']}",
        f"Executable        : {info['python_executable']}",
        f"Platform          : {info['platform']}",
        f"Processor         : {info['processor']}",
        f"Logical CPUs      : {info['cpu_count_logical']}",
        f"Physical CPUs     : {info.get('cpu_count_physical')}",
        f"RAM (GB)          : {info.get('total_ram_gb')}",
        f"TensorFlow        : {info.get('tensorflow_version')}",
        f"Keras             : {info.get('keras_version')}",
        f"TF built w/ CUDA  : {info.get('tf_build_with_cuda')}",
        f"GPUs detected     : {info.get('gpus_detected')}",
        f"nvidia-smi        : {info.get('nvidia_smi')}",
        f"CUDA toolkit      : {str(info.get('cuda_toolkit'))[:120]}",
        f"NumPy             : {info.get('numpy_version')}",
        f"pandas            : {info.get('pandas_version')}",
        f"SciPy             : {info.get('scipy_version')}",
        f"scikit-learn      : {info.get('sklearn_version')}",
        f"matplotlib        : {info.get('matplotlib_version')}",
        f"Pillow            : {info.get('PIL_version')}",
        f"CPU benchmark     : {info.get('cpu_benchmark')}",
        f"Seeds             : {info['seeds']}",
        "",
        "NOTE: " + info["compute_note"],
        "",
        "pip freeze",
        "-" * 70,
        *info["pip_freeze"],
    ]
    (config.PROJECT_ROOT / "environment.txt").write_text("\n".join(lines), encoding="utf-8")

    if verbose:
        for line in lines[:24]:
            log.info(line)
    log.info("Wrote results/environment.json and environment.txt")
    return info


if __name__ == "__main__":
    check()
