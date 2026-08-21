"""Shared utilities: logging, seeding, JSON/CSV IO, timing, status tracking."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config

_LOGGER_NAME = "radiomics"
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if not _configured:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.LOGS_DIR / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        _configured = True
    return logger if name is None else logger.getChild(name)


def banner(text: str, char: str = "=", width: int = 78) -> None:
    log = get_logger()
    log.info(char * width)
    log.info(text)
    log.info(char * width)


def set_all_seeds(seed: int = config.GLOBAL_SEED) -> None:
    """Seed every RNG the pipeline can touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


@contextmanager
def timer() -> Iterator[dict]:
    """Wall-clock timer. ``result['seconds']`` is populated on exit."""
    result: dict = {"seconds": None}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - t0


# --------------------------------------------------------------------------
# Experiment status tracking (checkpoint / resume)
# --------------------------------------------------------------------------
STATUS_PATH = config.RESULTS_DIR / "status.json"


def load_status() -> dict:
    return read_json(STATUS_PATH, default={}) or {}


def set_status(key: str, value: str, extra: dict | None = None) -> None:
    status = load_status()
    entry: dict[str, Any] = {"state": value, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        entry.update(extra)
    status[key] = entry
    write_json(STATUS_PATH, status)


def get_status(key: str) -> str | None:
    entry = load_status().get(key)
    if entry is None:
        return None
    if isinstance(entry, str):      # tolerate a hand-edited flat file
        return entry
    return entry.get("state")


def is_completed(key: str) -> bool:
    return get_status(key) == "completed"


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def collect_environment() -> dict:
    """Capture everything needed to reproduce / interpret a run."""
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "seeds": {
            "global": config.GLOBAL_SEED,
            "fold": config.FOLD_SEED,
            "augmentation": config.AUGMENTATION_SEED,
        },
    }

    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        info["cpu_count_physical"] = None
        info["total_ram_gb"] = None

    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "not installed"

    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
        info["keras_version"] = getattr(tf.keras, "__version__", "unknown")
        gpus = tf.config.list_physical_devices("GPU")
        info["gpus_detected"] = [g.name for g in gpus]
        info["gpu_available"] = bool(gpus)
        try:
            info["tf_build_with_cuda"] = bool(tf.test.is_built_with_cuda())
        except Exception:
            info["tf_build_with_cuda"] = None
    except Exception as exc:  # noqa: BLE001
        info["tensorflow_version"] = f"not importable ({exc})"
        info["gpu_available"] = False
        info["gpus_detected"] = []

    info["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"])
    info["cuda_toolkit"] = _run(["nvcc", "--version"])
    return info


def process_peak_memory_mb() -> float | None:
    """Peak resident set size of this process, in MB (measured)."""
    try:
        import psutil

        p = psutil.Process()
        mem = p.memory_info()
        peak = getattr(mem, "peak_wset", None)   # Windows-specific, if present
        if peak is not None:
            return round(peak / 1024**2, 2)
        return round(mem.rss / 1024**2, 2)
    except Exception:
        return None


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"
