"""Aggregate the 25 runs into fold-wise metrics, CV summaries and OOF predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .metrics import METRIC_KEYS, bootstrap_auc_ci, compute_metrics, mean_ci, roc_points
from .train import run_dir
from .utils import banner, get_logger, read_json, write_json

log = get_logger("evaluate")


def collect_fold_results() -> pd.DataFrame:
    """One row per completed (architecture, fold) run."""
    rows = []
    for arch in config.ABLATION_ORDER:
        for fold in range(1, config.N_FOLDS + 1):
            rp = run_dir(arch, fold) / "results.json"
            if not rp.exists():
                continue
            r = read_json(rp)
            m = r["test_metrics"]
            rows.append({
                "architecture": arch,
                "architecture_display": r["architecture_display"],
                "backbones": "+".join(r["backbones"]),
                "n_backbones": len(r["backbones"]),
                "fold": fold,
                "feature_dim": r["feature_dim"],
                **{k: m[k] for k in METRIC_KEYS},
                "balanced_accuracy": m["balanced_accuracy"],
                "npv": m["npv"],
                "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
                "n_test": m["n_samples"],
                "epochs_run": r["epochs_run"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": r["best_val_loss"],
                "train_seconds": r["train_seconds"],
                "seconds_per_epoch": r["seconds_per_epoch"],
                "head_inference_ms_per_image": r["head_inference_ms_per_image"],
                "trainable_params": r["head_parameters"]["trainable_params"],
            })
    if not rows:
        raise RuntimeError("No completed runs found. Run the train stage first.")
    return pd.DataFrame(rows).sort_values(["architecture", "fold"]).reset_index(drop=True)


def collect_oof_predictions() -> pd.DataFrame:
    """Pooled out-of-fold predictions - each image predicted exactly once per model."""
    frames = []
    for arch in config.ABLATION_ORDER:
        for fold in range(1, config.N_FOLDS + 1):
            p = run_dir(arch, fold) / "test_predictions.csv"
            if p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        raise RuntimeError("No prediction files found.")
    oof = pd.concat(frames, ignore_index=True)

    # Every architecture must cover each image exactly once across folds.
    for arch, sub in oof.groupby("architecture"):
        dupes = sub["relpath"].duplicated().sum()
        if dupes:
            raise AssertionError(
                f"{arch}: {dupes} images appear in more than one test fold.")
    return oof


def summarise_cv(fold_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean / SD / 95% CI per architecture per metric."""
    summary_rows, ci_rows = [], []
    for arch in config.ABLATION_ORDER:
        sub = fold_df[fold_df["architecture"] == arch]
        if sub.empty:
            continue
        cfg = config.ABLATION_CONFIGS[arch]
        row = {
            "architecture": arch,
            "architecture_display": cfg["display"],
            "n_folds": int(len(sub)),
            "feature_dim": int(sub["feature_dim"].iloc[0]),
        }
        for metric in METRIC_KEYS:
            s = mean_ci(sub[metric].to_numpy())
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
            row[f"{metric}_ci_lower"] = s["ci_lower"]
            row[f"{metric}_ci_upper"] = s["ci_upper"]
            row[f"{metric}_mean_pm_sd"] = f"{100*s['mean']:.2f} ± {100*s['std']:.2f}"
            ci_rows.append({
                "architecture": arch,
                "architecture_display": cfg["display"],
                "metric": metric,
                "mean": s["mean"], "std": s["std"], "n": s["n"], "sem": s["sem"],
                "ci_lower": s["ci_lower"], "ci_upper": s["ci_upper"],
                "confidence_level": config.CONFIDENCE_LEVEL,
                "method": s["method"],
            })
        row["train_seconds_mean"] = float(sub["train_seconds"].mean())
        row["epochs_run_mean"] = float(sub["epochs_run"].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(ci_rows)


def oof_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Metrics computed on the pooled out-of-fold predictions (the preferred view)."""
    rows, curves = [], {}
    for arch in config.ABLATION_ORDER:
        sub = oof[oof["architecture"] == arch]
        if sub.empty:
            continue
        m = compute_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
        ci = bootstrap_auc_ci(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
        rows.append({
            "architecture": arch,
            "architecture_display": config.ABLATION_CONFIGS[arch]["display"],
            **{k: m[k] for k in METRIC_KEYS},
            "auc_ci_lower": ci["ci_lower"], "auc_ci_upper": ci["ci_upper"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
            "n_samples": m["n_samples"],
        })
        curves[arch] = {
            "roc": roc_points(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy()),
            "auc_ci": ci,
            "confusion_matrix": m["confusion_matrix"],
            "display": config.ABLATION_CONFIGS[arch]["display"],
        }
    return pd.DataFrame(rows), curves


def evaluate() -> dict:
    banner("STAGE: EVALUATION & CROSS-VALIDATION SUMMARY")
    config.ensure_dirs()

    fold_df = collect_fold_results()
    fold_df.to_csv(config.CV_DIR / "fold_metrics.csv", index=False, encoding="utf-8")
    log.info("fold_metrics.csv: %d rows", len(fold_df))

    summary, cis = summarise_cv(fold_df)
    summary.to_csv(config.CV_DIR / "cv_summary.csv", index=False, encoding="utf-8")
    cis.to_csv(config.CV_DIR / "confidence_intervals.csv", index=False, encoding="utf-8")

    oof = collect_oof_predictions()
    oof.to_csv(config.CV_DIR / "out_of_fold_predictions.csv", index=False, encoding="utf-8")
    oof_df, curves = oof_metrics(oof)
    oof_df.to_csv(config.CV_DIR / "out_of_fold_metrics.csv", index=False, encoding="utf-8")
    write_json(config.CV_DIR / "roc_curves.json", curves)

    log.info("-" * 96)
    log.info("%-38s %-16s %-16s %-16s", "Architecture", "Accuracy", "F1", "AUC")
    log.info("-" * 96)
    for _, r in summary.iterrows():
        log.info("%-38s %-16s %-16s %-16s",
                 r["architecture_display"][:37],
                 r["accuracy_mean_pm_sd"], r["f1_mean_pm_sd"], r["roc_auc_mean_pm_sd"])
    log.info("-" * 96)

    payload = {
        "n_runs": int(len(fold_df)),
        "architectures": config.ABLATION_ORDER,
        "cv_summary": summary.to_dict(orient="records"),
        "out_of_fold": oof_df.to_dict(orient="records"),
    }
    write_json(config.CV_DIR / "cv_summary.json", payload)
    return payload


if __name__ == "__main__":
    evaluate()
