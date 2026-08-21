"""Component-wise ablation of the proposed hybrid architecture.

This is a genuine ablation, not a comparison of unrelated networks: every
configuration is the *same* fusion architecture with one or more backbone
branches removed, trained on identical folds with identical hyper-parameters.

    A1  VGG19                                       single branch
    A2  VGG19 + InceptionResNetV2                   MobileNetV2 removed
    A3  VGG19 + MobileNetV2                         InceptionResNetV2 removed
    A4  InceptionResNetV2 + MobileNetV2             VGG19 removed
    A5  VGG19 + InceptionResNetV2 + MobileNetV2     full proposed model

Each reduced variant therefore isolates the contribution of the branch it drops.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .metrics import METRIC_KEYS, mean_ci
from .utils import banner, get_logger, write_json

log = get_logger("ablation")

BRANCH_REMOVED = {
    "A1": "InceptionResNetV2 and MobileNetV2 removed",
    "A2": "MobileNetV2 removed",
    "A3": "InceptionResNetV2 removed",
    "A4": "VGG19 removed",
    "A5": "none (full proposed model)",
}


def build_ablation_results(fold_metrics: pd.DataFrame,
                           complexity: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for arch in config.ABLATION_ORDER:
        sub = fold_metrics[fold_metrics["architecture"] == arch]
        if sub.empty:
            continue
        cfg = config.ABLATION_CONFIGS[arch]
        row = {
            "config": arch,
            "architecture": cfg["display"],
            "backbones": " + ".join(config.BACKBONE_DISPLAY[b] for b in cfg["backbones"]),
            "n_branches": len(cfg["backbones"]),
            "branch_removed": BRANCH_REMOVED[arch],
            "fused_feature_dim": int(sub["feature_dim"].iloc[0]),
            "n_folds": int(len(sub)),
            "is_full_model": arch == config.FULL_MODEL,
        }
        for metric in METRIC_KEYS:
            s = mean_ci(sub[metric].to_numpy())
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
            row[f"{metric}_display"] = f"{100*s['mean']:.2f} ± {100*s['std']:.2f}"
        row["train_seconds_mean"] = float(sub["train_seconds"].mean())
        row["head_trainable_params"] = int(sub["trainable_params"].iloc[0])

        if complexity is not None and not complexity.empty:
            c = complexity[complexity["config"] == arch]
            if not c.empty:
                c = c.iloc[0]
                for k in ("total_params", "trainable_params", "non_trainable_params",
                          "gflops", "model_size_mb", "inference_ms_per_image"):
                    if k in c.index:
                        row[k] = c[k]
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_summary(ablation: pd.DataFrame,
                           reference: str = config.FULL_MODEL) -> pd.DataFrame:
    """Absolute improvement of the full model over each reduced variant."""
    if ablation.empty:
        return pd.DataFrame()
    ref = ablation[ablation["config"] == reference]
    if ref.empty:
        return pd.DataFrame()
    ref = ref.iloc[0]

    rows = []
    for _, r in ablation.iterrows():
        if r["config"] == reference:
            continue
        row = {
            "comparison": f"{reference} vs {r['config']}",
            "full_model": ref["architecture"],
            "reduced_variant": r["architecture"],
            "branch_removed_in_variant": r["branch_removed"],
            "fused_dim_full": ref["fused_feature_dim"],
            "fused_dim_variant": r["fused_feature_dim"],
        }
        for metric in METRIC_KEYS:
            f, v = ref[f"{metric}_mean"], r[f"{metric}_mean"]
            row[f"{metric}_full"] = f
            row[f"{metric}_variant"] = v
            row[f"{metric}_abs_improvement_pp"] = 100.0 * (f - v)
            row[f"{metric}_rel_improvement_pct"] = (
                100.0 * (f - v) / v if v else float("nan"))
        for k in ("total_params", "gflops", "model_size_mb", "inference_ms_per_image"):
            if k in ablation.columns and pd.notna(r.get(k)) and pd.notna(ref.get(k)):
                row[f"{k}_full"] = ref[k]
                row[f"{k}_variant"] = r[k]
                row[f"{k}_delta"] = ref[k] - r[k]
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation() -> dict:
    banner("STAGE: ABLATION ANALYSIS")
    config.ensure_dirs()

    fold_metrics = pd.read_csv(config.CV_DIR / "fold_metrics.csv")
    cpath = config.COMPLEXITY_DIR / "computational_complexity.csv"
    complexity = pd.read_csv(cpath) if cpath.exists() else None

    ablation = build_ablation_results(fold_metrics, complexity)
    ablation.to_csv(config.ABLATION_DIR / "ablation_results.csv",
                    index=False, encoding="utf-8")

    summary = build_ablation_summary(ablation)
    summary.to_csv(config.ABLATION_DIR / "ablation_summary.csv",
                   index=False, encoding="utf-8")

    log.info("-" * 92)
    log.info("%-6s %-38s %-8s %-16s %-16s", "Cfg", "Architecture", "Dim",
             "Accuracy", "F1")
    log.info("-" * 92)
    for _, r in ablation.iterrows():
        log.info("%-6s %-38s %-8d %-16s %-16s", r["config"],
                 r["architecture"][:37], r["fused_feature_dim"],
                 r["accuracy_display"], r["f1_display"])
    log.info("-" * 92)
    if not summary.empty:
        log.info("Absolute accuracy improvement of the full model (percentage points):")
        for _, r in summary.iterrows():
            log.info("   %-12s %+6.2f pp  (%s)", r["comparison"],
                     r["accuracy_abs_improvement_pp"], r["branch_removed_in_variant"])

    payload = {
        "ablation": ablation.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "definition": (
            "Component ablation of a single fusion architecture: each variant is "
            "the full model with one or more backbone branches removed, trained "
            "on identical folds with identical hyper-parameters."
        ),
    }
    write_json(config.ABLATION_DIR / "ablation.json", payload)
    return payload


if __name__ == "__main__":
    run_ablation()
