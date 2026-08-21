"""Manuscript-ready tables, written as CSV and XLSX.

Table 10  Real five-fold cross-validation summary
Table 11  Real ablation study
Table 12  Real computational complexity
Table 13  Statistical significance analysis

These are generated from the experimental outputs only. The illustrative values
in the current manuscript are never copied into them; a separate comparison
table (``table_14_reported_vs_measured``) sets the paper's existing reported
numbers beside the measured ones so the difference is explicit rather than
hidden.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .utils import banner, get_logger, read_json

log = get_logger("tables")

# Values already reported in the manuscript. Recorded here only so they can be
# displayed alongside the new measurements - never merged into a results table.
PAPER_REPORTED = {
    "Custom CNN": {"accuracy": 89.10, "task": "Multiclass"},
    "VGG19": {"accuracy": 94.50, "task": "Binary"},
    "ResNet-50": {"accuracy": 82.16, "task": "Binary"},
    "DenseNet-121": {"accuracy": 92.99, "task": "Binary"},
    "XceptionNet": {"accuracy": 89.20, "task": "Multiclass"},
    "Hybrid Model 2 (DenseNet + EfficientNet)": {"accuracy": 94.80, "task": "Multiclass"},
    "Hybrid Model 1 (VGG19 + InceptionResNetV2 + MobileNetV2)": {
        "accuracy": 97.50, "task": "Binary"},
}


def _write(df: pd.DataFrame, stem: str, sheet: str) -> None:
    csv_path = config.TABLES_DIR / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df.to_excel(config.TABLES_DIR / f"{stem}.xlsx", index=False,
                    sheet_name=sheet[:31])
    except Exception as exc:  # noqa: BLE001
        log.warning("XLSX write failed for %s: %s", stem, exc)
    log.info("  %s.csv / .xlsx  (%d rows)", stem, len(df))


def table_10_cross_validation(cv_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cv_summary.iterrows():
        rows.append({
            "Config": r["architecture"],
            "Model": r["architecture_display"],
            "Fused dim": int(r["feature_dim"]),
            "Folds": int(r["n_folds"]),
            "Accuracy (mean ± SD %)": r["accuracy_mean_pm_sd"],
            "Precision (mean ± SD %)": r["precision_mean_pm_sd"],
            "Recall (mean ± SD %)": r["recall_mean_pm_sd"],
            "F1-score (mean ± SD %)": r["f1_mean_pm_sd"],
            "Sensitivity (mean ± SD %)": r["sensitivity_mean_pm_sd"],
            "Specificity (mean ± SD %)": r["specificity_mean_pm_sd"],
            "ROC-AUC (mean ± SD)": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
            "Accuracy 95% CI (%)": f"[{100*r['accuracy_ci_lower']:.2f}, {100*r['accuracy_ci_upper']:.2f}]",
            "F1 95% CI (%)": f"[{100*r['f1_ci_lower']:.2f}, {100*r['f1_ci_upper']:.2f}]",
            "AUC 95% CI": f"[{r['roc_auc_ci_lower']:.4f}, {r['roc_auc_ci_upper']:.4f}]",
        })
    return pd.DataFrame(rows)


def table_11_ablation(ablation: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ablation.iterrows():
        row = {
            "Config": r["config"],
            "Backbone combination": r["backbones"],
            "Branches": int(r["n_branches"]),
            "Branch removed": r["branch_removed"],
            "Fused dim": int(r["fused_feature_dim"]),
            "Accuracy (%)": r["accuracy_display"],
            "Precision (%)": r["precision_display"],
            "Recall (%)": r["recall_display"],
            "F1-score (%)": r["f1_display"],
            "Sensitivity (%)": r["sensitivity_display"],
            "Specificity (%)": r["specificity_display"],
            "ROC-AUC": r["roc_auc_display"],
        }
        if "total_params" in r.index and pd.notna(r.get("total_params")):
            row["Parameters (M)"] = round(float(r["total_params"]) / 1e6, 2)
        if "gflops" in r.index and pd.notna(r.get("gflops")):
            row["GFLOPs"] = round(float(r["gflops"]), 2)
        if not summary.empty and r["config"] != config.FULL_MODEL:
            m = summary[summary["comparison"] == f"{config.FULL_MODEL} vs {r['config']}"]
            if not m.empty:
                row["Δ Accuracy vs A5 (pp)"] = round(
                    -float(m.iloc[0]["accuracy_abs_improvement_pp"]), 2)
                row["Δ F1 vs A5 (pp)"] = round(
                    -float(m.iloc[0]["f1_abs_improvement_pp"]), 2)
        else:
            row["Δ Accuracy vs A5 (pp)"] = 0.0
            row["Δ F1 vs A5 (pp)"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def table_12_complexity(complexity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in complexity.iterrows():
        row = {
            "Config": r["config"],
            "Model": r["architecture"],
            "Total parameters": int(r["total_params"]),
            "Parameters (M)": r["total_params_millions"],
            "Trainable parameters": int(r["trainable_params"]),
            "Non-trainable parameters": int(r["non_trainable_params"]),
            "GFLOPs (batch 1) [calculated]": r["gflops"],
            "GMACs [calculated]": r["gmacs"],
            "Model size (MB) [measured]": r["model_size_mb"],
            "Inference ms/image [measured, CPU]": r["inference_ms_per_image"],
            "Inference SD (ms)": r["inference_ms_std"],
            "Batched ms/image [measured, CPU]": r["batched_ms_per_image"],
        }
        if "head_train_seconds_mean" in r.index and pd.notna(r.get("head_train_seconds_mean")):
            row["Head training time per fold (s) [measured]"] = r["head_train_seconds_mean"]
            row["Head inference ms/image [measured]"] = r.get("head_inference_ms_per_image")
        row["Peak process memory (MB) [measured]"] = r.get("peak_process_memory_mb")
        row["Device"] = "CPU (no CUDA GPU available)"
        rows.append(row)
    return pd.DataFrame(rows)


def table_13_statistics(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in stats_df.iterrows():
        rows.append({
            "Metric": r["metric"],
            "Comparison": f"{r['reference']} vs {r['comparator']}",
            "Full model": r["reference_display"],
            "Reduced variant": r["comparator_display"],
            "Folds": int(r["n_folds"]),
            "Mean (full)": round(float(r["reference_mean"]), 4),
            "Mean (variant)": round(float(r["comparator_mean"]), 4),
            "Mean difference": round(float(r["mean_difference"]), 4),
            "Cohen's d (paired)": round(float(r["cohens_d_paired"]), 3),
            "Paired t": round(float(r["paired_t_statistic"]), 3),
            "Paired t p": round(float(r["paired_t_p_value"]), 5),
            "Paired t p (Holm)": round(float(r["paired_t_p_holm"]), 5),
            "Significant (t, Holm)": "yes" if r["paired_t_significant_holm"] else "no",
            "Wilcoxon W": round(float(r["wilcoxon_statistic"]), 3),
            "Wilcoxon p": round(float(r["wilcoxon_p_value"]), 5),
            "Wilcoxon p (Holm)": round(float(r["wilcoxon_p_holm"]), 5),
            "Significant (Wilcoxon, Holm)": "yes" if r["wilcoxon_significant_holm"] else "no",
        })
    return pd.DataFrame(rows)


def table_14_reported_vs_measured(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """Set the manuscript's existing reported values beside what was measured.

    The two columns are deliberately kept apart. The reported column is what the
    manuscript already states; the measured column is this pipeline's result on
    a de-duplicated dataset under grouped cross-validation. They answer different
    questions and are not interchangeable.
    """
    rows = []
    for model, info in PAPER_REPORTED.items():
        rows.append({
            "Model (as reported in manuscript)": model,
            "Task (as reported)": info["task"],
            "Reported accuracy (%)": info["accuracy"],
            "Reported evaluation": "single split; illustrative CV/ablation section",
            "Measured here": "not re-run" ,
            "Measured accuracy (mean ± SD %)": "",
            "Comment": "existing reported result, preserved unchanged",
        })

    full = cv_summary[cv_summary["architecture"] == config.FULL_MODEL]
    if not full.empty:
        r = full.iloc[0]
        rows.append({
            "Model (as reported in manuscript)":
                "Hybrid Model 1 (VGG19 + InceptionResNetV2 + MobileNetV2)",
            "Task (as reported)": "Binary",
            "Reported accuracy (%)": 97.50,
            "Reported evaluation": "single split; illustrative five-fold section",
            "Measured here": "A5, real grouped five-fold CV",
            "Measured accuracy (mean ± SD %)": r["accuracy_mean_pm_sd"],
            "Comment": ("measured on de-duplicated data with grouped folds and "
                        "frozen backbones; not directly comparable to the "
                        "reported single-split value"),
        })
    return pd.DataFrame(rows)


def generate_all_tables() -> dict:
    banner("STAGE: TABLE GENERATION")
    config.ensure_dirs()
    made = []

    cv_summary = pd.read_csv(config.CV_DIR / "cv_summary.csv")
    t10 = table_10_cross_validation(cv_summary)
    _write(t10, "table_10_cross_validation", "Table10_CV")
    made.append("table_10_cross_validation")

    ap = config.ABLATION_DIR / "ablation_results.csv"
    if ap.exists():
        ablation = pd.read_csv(ap)
        sp = config.ABLATION_DIR / "ablation_summary.csv"
        summary = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
        _write(table_11_ablation(ablation, summary), "table_11_ablation", "Table11_Ablation")
        made.append("table_11_ablation")

    cp = config.COMPLEXITY_DIR / "computational_complexity.csv"
    if cp.exists():
        _write(table_12_complexity(pd.read_csv(cp)), "table_12_complexity", "Table12_Complexity")
        made.append("table_12_complexity")

    sp = config.STATS_DIR / "statistical_comparison.csv"
    if sp.exists():
        st = pd.read_csv(sp)
        _write(table_13_statistics(st), "table_13_statistical_tests", "Table13_Stats")
        made.append("table_13_statistical_tests")
        acc = st[st["metric"] == "accuracy"]
        if not acc.empty:
            _write(table_13_statistics(acc), "table_13a_statistical_tests_accuracy",
                   "Table13a_Accuracy")
            made.append("table_13a_statistical_tests_accuracy")

    _write(table_14_reported_vs_measured(cv_summary),
           "table_14_reported_vs_measured", "Table14_Comparison")
    made.append("table_14_reported_vs_measured")

    fold = pd.read_csv(config.CV_DIR / "fold_metrics.csv")
    _write(fold, "table_S1_fold_level_metrics", "S1_FoldMetrics")
    made.append("table_S1_fold_level_metrics")

    oof = config.CV_DIR / "out_of_fold_metrics.csv"
    if oof.exists():
        _write(pd.read_csv(oof), "table_S2_out_of_fold_metrics", "S2_OOF")
        made.append("table_S2_out_of_fold_metrics")

    stats = read_json(config.RESULTS_DIR / "dataset_statistics.json", default={}) or {}
    if stats:
        ds_rows = []
        raw = stats["raw_scan"]["class_counts_including_duplicates"]
        uniq = stats["deduplicated"]["class_counts"]
        for cls in ("normal", "osteopenia", "osteoporosis"):
            ds_rows.append({
                "Class": cls.capitalize(),
                "Files on disk (with duplicates)": raw.get(cls, 0),
                "Unique images after de-duplication": uniq.get(cls, 0),
                "Duplicates removed": raw.get(cls, 0) - uniq.get(cls, 0),
            })
        ds_rows.append({
            "Class": "TOTAL",
            "Files on disk (with duplicates)": sum(raw.values()),
            "Unique images after de-duplication": sum(uniq.values()),
            "Duplicates removed": sum(raw.values()) - sum(uniq.values()),
        })
        _write(pd.DataFrame(ds_rows), "table_S3_dataset_composition", "S3_Dataset")
        made.append("table_S3_dataset_composition")

    log.info("Generated %d tables (CSV + XLSX) in %s", len(made), config.TABLES_DIR)
    return {"tables": made, "count": len(made)}


if __name__ == "__main__":
    generate_all_tables()
