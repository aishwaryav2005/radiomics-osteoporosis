"""Final report: everything the experiments actually produced, in one document.

The report states what was measured, how, and what the limitations are. Where a
manuscript claim is not supported by the experiments, that is written plainly
rather than smoothed over.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import banner, get_logger, human_seconds, load_status, read_json

log = get_logger("report")


def _fmt_pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * float(x):.{nd}f}"


def _table(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    if df.empty:
        return ["_(no data)_", ""]
    cols = cols or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if np.isfinite(v) else "n/a")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def generate_report() -> Path:
    banner("STAGE: FINAL REPORT")
    config.ensure_dirs()

    stats = read_json(config.RESULTS_DIR / "dataset_statistics.json", default={}) or {}
    env = read_json(config.RESULTS_DIR / "environment.json", default={}) or {}
    split = read_json(config.SPLITS_DIR / "split_manifest.json", default={}) or {}
    caveats = read_json(config.STATS_DIR / "statistical_caveats.json", default={}) or {}
    mm = read_json(config.MULTIMODAL_DIR / "lumos_feasibility.json", default={}) or {}
    bank = read_json(config.CACHE_DIR / "feature_bank_meta.json", default={}) or {}

    def load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    fold_metrics = load(config.CV_DIR / "fold_metrics.csv")
    cv_summary = load(config.CV_DIR / "cv_summary.csv")
    oof = load(config.CV_DIR / "out_of_fold_metrics.csv")
    ablation = load(config.ABLATION_DIR / "ablation_results.csv")
    ab_summary = load(config.ABLATION_DIR / "ablation_summary.csv")
    complexity = load(config.COMPLEXITY_DIR / "computational_complexity.csv")
    stats_df = load(config.STATS_DIR / "statistical_comparison.csv")
    exp_status = load(config.RESULTS_DIR / "experiment_status.csv")

    L: list[str] = []
    A = L.append

    A("# Real Five-Fold Cross-Validation and Ablation Study")
    A("## Radiomics-Driven Knee X-Ray Analysis for Bone Mineral Density Risk Stratification")
    A("")
    A("This report contains **only measured experimental results**. No value here is")
    A("copied from the manuscript's illustrative sections, and no value is simulated.")
    A("")

    # ---------------------------------------------------------------- summary
    n_done = int((exp_status["status"] == "completed").sum()) if not exp_status.empty else 0
    n_fail = int((exp_status["status"] == "failed").sum()) if not exp_status.empty else 0
    n_total = len(exp_status) if not exp_status.empty else 0

    A("## Executive summary")
    A("")
    A(f"- Experiments completed: **{n_done} / {n_total}** ({n_fail} failed)")
    if not cv_summary.empty:
        best = cv_summary.loc[cv_summary["accuracy_mean"].idxmax()]
        full = cv_summary[cv_summary["architecture"] == config.FULL_MODEL]
        A(f"- Best architecture by mean accuracy: **{best['architecture']} "
          f"({best['architecture_display']})** at **{best['accuracy_mean_pm_sd']}%**")
        if not full.empty:
            f = full.iloc[0]
            A(f"- Full proposed hybrid (A5): **{f['accuracy_mean_pm_sd']}%** accuracy, "
              f"F1 **{f['f1_mean_pm_sd']}%**, AUC "
              f"**{f['roc_auc_mean']:.4f} ± {f['roc_auc_std']:.4f}**")
    if stats:
        b = stats.get("binary_task", {})
        A(f"- Binary task sample size: **{b.get('usable_images')} unique images** "
          f"({b.get('class_counts')})")
        A(f"- Manuscript states 1,947 images; **{stats['raw_scan']['total_files_scanned']} "
          f"files** exist and only **{stats['deduplicated']['unique_images']} are unique**")
    A("")

    # ---------------------------------------------------------------- 1 dataset
    A("## 1. Dataset statistics")
    A("")
    if stats:
        raw, dedup = stats["raw_scan"], stats["deduplicated"]
        A(f"- Files scanned: **{raw['total_files_scanned']}** across "
          f"{len(raw['per_dataset_class_counts_including_duplicates'])} source folders")
        A(f"- Unreadable / unusable files: **{raw['failed_files']}**")
        A(f"- Class counts *including* duplicates: `{raw['class_counts_including_duplicates']}`")
        A(f"- **Unique images after de-duplication: {dedup['unique_images']}**")
        A(f"- Class counts after de-duplication: `{dedup['class_counts']}`")
        A("")
        A("### Per-dataset composition (as originally provided, before de-duplication)")
        A("")
        A("Counts below are each source folder's own file counts, i.e. what a reader "
          "tracing back to the original Kaggle pages would see. They double-count "
          "images that are shared between sources — see the overlap matrix in "
          "section 2 for the de-duplicated relationship between the four folders.")
        A("")
        rows = [{"Dataset": k, **v} for k, v in
                stats["raw_scan"]["per_dataset_class_counts_including_duplicates"].items()]
        L.extend(_table(pd.DataFrame(rows)))
        A("### Comparison with the manuscript")
        A("")
        A("| Quantity | Manuscript | Measured here |")
        A("|---|---|---|")
        cc = raw["class_counts_including_duplicates"]
        A(f"| Total images | 1,947 | {raw['total_files_scanned']} files on disk, "
          f"**{dedup['unique_images']} unique** |")
        A(f"| Normal | 780 | {cc.get('normal', 0)} files / "
          f"**{dedup['class_counts'].get('normal', 0)} unique** |")
        A(f"| Osteoporosis | 793 | {cc.get('osteoporosis', 0)} files / "
          f"**{dedup['class_counts'].get('osteoporosis', 0)} unique** |")
        A(f"| Osteopenia | 374 | {cc.get('osteopenia', 0)} files / "
          f"**{dedup['class_counts'].get('osteopenia', 0)} unique** |")
        A("")
    A("")

    # ---------------------------------------------------------------- 2 quality
    A("## 2. Data quality")
    A("")
    if stats:
        d = stats["duplicates"]
        A(f"- Exact duplicate images removed: **{d['exact_pixel_duplicates_removed']}** "
          f"({100 * d['exact_pixel_duplicates_removed'] / max(1, d['total_files_scanned']):.1f}% "
          "of all files)")
        A(f"- Duplicate filenames observed: **{d['duplicate_filenames']}**")
        A(f"- Near-duplicate candidate pairs: {d['near_duplicate_candidate_pairs']}; "
          f"**verified: {d['near_duplicate_pairs_verified']}** "
          f"(pixel correlation ≥ {d['near_duplicate_correlation_threshold']})")
        A("")
        A("### Source datasets are not independent")
        A("")
        A("Pixel-level overlap between the four source folders (shared unique images):")
        A("")
        ov = d["cross_dataset_pixel_overlap"]
        names = list(ov)
        A("| | " + " | ".join(names) + " |")
        A("|---|" + "|".join("---" for _ in names) + "|")
        for a in names:
            A(f"| **{a}** | " + " | ".join(str(ov[a][b]) for b in names) + " |")
        A("")
        A("This shows the four folders are **not four independent datasets**:")
        A("")
        A("- `dataset3` and `dataset4` are each **entirely contained** in `dataset2`")
        A("- `dataset3 ∩ dataset4 = 0` — they are disjoint halves of `dataset2`")
        A("- `dataset1 ∩ dataset2 = 86` images")
        A("")
        A("Effectively there are **two** distinct image sources, not four. The "
          "manuscript's per-class counts (780 normal / 793 osteoporosis) are exactly "
          "the duplicate-inflated totals, which indicates duplicates were counted as "
          "independent samples.")
        A("")
        lc = stats.get("label_conflicts", {})
        if lc:
            A("### Label conflicts between sources")
            A("")
            A(f"- Verified near-duplicate clusters with contradictory labels: "
              f"**{lc['clusters_with_conflicting_labels']}**")
            A(f"- Images relabelled from DXA T-score evidence: "
              f"**{lc['images_relabelled_from_t_score']}**")
            A(f"- Images excluded as unresolvable: **{lc['images_excluded_as_unresolvable']}**")
            A("")
            A("`dataset1` ships DXA T-scores whose labels are **100% consistent** with "
              "the WHO/ISCD criterion, so it serves as ground truth where the sources "
              "disagree. Concretely, images that `dataset1` records as osteoporotic "
              "(T-scores of −2.78, −2.53, −2.76, −2.52, −2.99) appear in `dataset2` "
              "labelled *Normal*. The public Kaggle sources therefore contain genuine "
              "label errors.")
            A("")
            A(f"Resolution rule applied: {lc['rule']}")
            A("")
    A("")

    # ---------------------------------------------------------------- 3 folds
    A("## 3. Fold structure")
    A("")
    if split:
        A(f"- Splitter: `{split['splitter']}` with seed `{split['fold_seed']}`")
        A(f"- Folds: **{split['n_folds']}**, all architectures share identical folds")
        A(f"- Total images: **{split['total_images']}**, grouping units: "
          f"**{split['total_groups']}**")
        A(f"- Grouping composition: `{split['group_kind_counts']}`")
        A(f"- Out-of-fold coverage verified: **{split['out_of_fold_coverage_verified']}**")
        A("")
        L.extend(_table(pd.DataFrame(split["folds"]),
                        ["fold", "n_train", "n_validation", "n_test",
                         "train_positive_rate", "validation_positive_rate",
                         "test_positive_rate", "n_groups_train", "n_groups_test"]))
        A("### Leakage prevention — and an honest limit")
        A("")
        A("Verified per fold by assertion (the pipeline raises if violated): no "
          "`group_id`, no pixel hash and no file path is shared between train, "
          "validation and test.")
        A("")
        A("**The manuscript's claim that \"patient-wise separation was maintained\" is "
          "not supportable for most of this data.** Patient identifiers exist only for "
          "`dataset1`. Of the "
          f"{stats.get('binary_task', {}).get('usable_images', '?')} binary-task images, "
          f"only **{stats.get('binary_task', {}).get('images_with_real_patient_id', '?')} "
          f"({100 * stats.get('binary_task', {}).get('fraction_with_real_patient_id', 0):.1f}%)** "
          "carry a real patient id. For the remainder, grouping falls back to verified "
          "near-duplicate clusters, which prevents duplicate leakage but is **not** "
          "verified patient-level separation.")
        A("")
        A("A further caveat: `dataset1` files some identical radiographs under two "
          "different patient ids (e.g. `OP101`/`OP135`), so even its patient key is "
          "imperfect. See `docs/near_duplicate_calibration.md`.")
        A("")
    A("")

    # ---------------------------------------------------------------- 4 config
    A("## 4. Training configuration")
    A("")
    A(f"- Task: **binary**, normal (0) vs osteoporosis (1); positive class = "
      f"**{config.POSITIVE_CLASS}**")
    A(f"- Osteopenia excluded from the binary ablation (retained in the manifest)")
    A(f"- Input: {config.IMAGE_SIZE[0]}×{config.IMAGE_SIZE[1]} RGB")
    A(f"- Optimiser: {config.OPTIMIZER}, lr {config.LEARNING_RATE}, "
      f"batch {config.BATCH_SIZE}, loss {config.LOSS}")
    A(f"- Max epochs {config.MAX_EPOCHS}, early stopping patience "
      f"{config.EARLY_STOPPING_PATIENCE} on `{config.MONITOR_METRIC}`")
    A(f"- ReduceLROnPlateau: factor {config.REDUCE_LR_FACTOR}, patience "
      f"{config.REDUCE_LR_PATIENCE}, min lr {config.MIN_LEARNING_RATE}")
    A(f"- Dropout {config.HEAD_DROPOUT}, dense units {config.HEAD_DENSE_UNITS}")
    A(f"- Decision threshold: {config.DECISION_THRESHOLD}")
    A("")
    A("### Transfer-learning protocol — what was and was not done")
    A("")
    A("**Stage 1 only was executed.** The ImageNet backbones are used as *frozen "
      "feature extractors*; only the fusion head is trained. Stage 2 (unfreezing "
      "deeper layers and fine-tuning at a lower learning rate) is implemented "
      "(`build_end_to_end_model(trainable_backbones=True)`) but **was not run**, "
      "because this machine has no CUDA GPU.")
    A("")
    A("Because the backbones are frozen and the augmentation set is fixed and "
      "deterministic, each backbone's output for a given (image, variant) pair is "
      "constant. It is computed once and cached. That is a caching optimisation, "
      "**mathematically identical** to running the frozen backbone inside the "
      "training loop each epoch — not an approximation.")
    A("")
    A("The reported results therefore characterise **frozen-backbone feature "
      "fusion**, not end-to-end fine-tuning. Results from a fine-tuned model would "
      "likely differ.")
    A("")
    A("### Augmentation")
    A("")
    A(f"- Applied to **training data only**; validation and test use un-augmented "
      f"features (variant 0)")
    A(f"- {config.N_AUG_VARIANTS} deterministic augmented variants per training image, "
      f"seed {config.AUGMENTATION_SEED}")
    A(f"- Rotation ±{config.AUG_ROTATION_DEG}°, horizontal flip, "
      f"zoom {config.AUG_ZOOM_RANGE}, brightness {config.AUG_BRIGHTNESS_RANGE}, "
      f"contrast {config.AUG_CONTRAST_RANGE}")
    A("")
    A("### Per-backbone preprocessing")
    A("")
    A("Each backbone receives the preprocessing it was pretrained with, rather than "
      "one shared rescale:")
    A("")
    A("| Backbone | Preprocessing |")
    A("|---|---|")
    A("| VGG19 | Caffe-style: RGB→BGR, ImageNet mean subtraction |")
    A("| InceptionResNetV2 | scale to [−1, 1] |")
    A("| MobileNetV2 | scale to [−1, 1] |")
    A("")

    # ---------------------------------------------------------------- 5 arch
    A("## 5. Model architectures")
    A("")
    if bank.get("backbones"):
        A("Measured feature dimensions (verified against the manuscript's stated values):")
        A("")
        A("| Backbone | Measured dim | Manuscript | Match |")
        A("|---|---|---|---|")
        for b, info in bank["backbones"].items():
            A(f"| {config.BACKBONE_DISPLAY.get(b, b)} | {info['feature_dim']} | "
              f"{info['expected_dim']} | {'yes' if info['dim_matches_paper'] else 'NO'} |")
        A(f"| **Fused (all three)** | **{bank.get('fused_feature_dim_all_three')}** | "
          f"**3328** | "
          f"{'yes' if bank.get('fused_feature_dim_all_three') == 3328 else 'NO'} |")
        A("")
    A("Ablation configurations — each is the same fusion architecture with branches removed:")
    A("")
    A("| Config | Backbones | Branch removed | Fused dim |")
    A("|---|---|---|---|")
    from .ablation import BRANCH_REMOVED

    for a in config.ABLATION_ORDER:
        c = config.ABLATION_CONFIGS[a]
        dim = sum(config.BACKBONE_FEATURE_DIMS[b] for b in c["backbones"])
        A(f"| {a} | {c['display']} | {BRANCH_REMOVED[a]} | {dim} |")
    A("")

    # ---------------------------------------------------------------- 6 folds
    A("## 6. Fold-wise metrics")
    A("")
    if not fold_metrics.empty:
        show = fold_metrics.copy()
        for c in ["accuracy", "precision", "recall", "f1", "sensitivity", "specificity"]:
            show[c] = (100 * show[c]).round(2)
        show["roc_auc"] = show["roc_auc"].round(4)
        L.extend(_table(show, ["architecture", "fold", "accuracy", "precision",
                               "recall", "f1", "sensitivity", "specificity",
                               "roc_auc", "tp", "tn", "fp", "fn", "epochs_run"]))
    A("")

    # ---------------------------------------------------------------- 7-8 cv
    A("## 7. Cross-validation summary (mean ± SD)")
    A("")
    if not cv_summary.empty:
        t = cv_summary[["architecture", "architecture_display", "accuracy_mean_pm_sd",
                        "precision_mean_pm_sd", "recall_mean_pm_sd", "f1_mean_pm_sd",
                        "sensitivity_mean_pm_sd", "specificity_mean_pm_sd"]].copy()
        t.columns = ["Config", "Model", "Accuracy", "Precision", "Recall", "F1",
                     "Sensitivity", "Specificity"]
        L.extend(_table(t))
        A("Sensitivity and specificity are derived directly from confusion-matrix "
          "counts and independently cross-checked against scikit-learn; the pipeline "
          "raises if they disagree. `sensitivity == recall` holds **because** the "
          f"positive class is pinned to `{config.POSITIVE_CLASS}`.")
        A("")

    A("## 8. Confidence intervals")
    A("")
    if not cv_summary.empty:
        rows = []
        for _, r in cv_summary.iterrows():
            for m in ["accuracy", "f1", "roc_auc"]:
                rows.append({
                    "Config": r["architecture"], "Metric": m,
                    "Mean": round(float(r[f"{m}_mean"]), 4),
                    "SD": round(float(r[f"{m}_std"]), 4),
                    "95% CI lower": round(float(r[f"{m}_ci_lower"]), 4),
                    "95% CI upper": round(float(r[f"{m}_ci_upper"]), 4),
                })
        L.extend(_table(pd.DataFrame(rows)))
        A("Intervals use the Student-t distribution with df = 4 (5 folds). A normal "
          "approximation would understate them at this sample size.")
        A("")
    if not oof.empty:
        A("### Pooled out-of-fold performance")
        A("")
        t = oof.copy()
        for c in ["accuracy", "precision", "recall", "f1", "sensitivity", "specificity"]:
            t[c] = (100 * t[c]).round(2)
        L.extend(_table(t, ["architecture_display", "accuracy", "precision", "recall",
                            "f1", "sensitivity", "specificity", "roc_auc",
                            "auc_ci_lower", "auc_ci_upper", "tp", "tn", "fp", "fn"]))
        A("AUC intervals here are stratified bootstrap percentile intervals over "
          "pooled out-of-fold predictions (each image predicted exactly once).")
        A("")

    # ---------------------------------------------------------------- 9 stats
    A("## 9. Statistical tests")
    A("")
    if not stats_df.empty:
        acc = stats_df[stats_df["metric"] == "accuracy"].copy()
        t = acc[["comparison", "mean_difference", "cohens_d_paired",
                 "paired_t_statistic", "paired_t_p_value", "paired_t_p_holm",
                 "paired_t_significant_holm", "wilcoxon_p_value", "wilcoxon_p_holm"]]
        t.columns = ["Comparison", "Mean diff", "Cohen's d", "t", "p", "p (Holm)",
                     "Significant", "Wilcoxon p", "Wilcoxon p (Holm)"]
        L.extend(_table(t))
        n_sig = int(acc["paired_t_significant_holm"].sum())
        A(f"**{n_sig} of {len(acc)}** accuracy comparisons reach significance after "
          f"Holm-Bonferroni correction at α = {config.ALPHA}.")
        A("")
    if caveats:
        A("### Statistical caveats")
        A("")
        A(f"- {caveats['interpretation']}")
        A(f"- {caveats['folds_are_not_independent']}")
        A("")

    # ---------------------------------------------------------------- 10 ablation
    A("## 10. Ablation results")
    A("")
    A("This is a **component ablation**: every configuration is the same fusion "
      "architecture with one or more backbone branches removed, trained on identical "
      "folds with identical hyper-parameters. It is not a comparison of unrelated "
      "networks, and it does not mix binary with multiclass tasks.")
    A("")
    if not ablation.empty:
        t = ablation[["config", "architecture", "fused_feature_dim", "accuracy_display",
                      "precision_display", "recall_display", "f1_display",
                      "sensitivity_display", "specificity_display", "roc_auc_display"]].copy()
        t.columns = ["Config", "Architecture", "Dim", "Accuracy", "Precision", "Recall",
                     "F1", "Sensitivity", "Specificity", "ROC-AUC"]
        L.extend(_table(t))
    if not ab_summary.empty:
        A("### Contribution of the full model over each reduced variant")
        A("")
        t = ab_summary[["comparison", "branch_removed_in_variant",
                        "accuracy_abs_improvement_pp", "f1_abs_improvement_pp",
                        "roc_auc_abs_improvement_pp"]].copy()
        t.columns = ["Comparison", "Branch removed in variant",
                     "Δ Accuracy (pp)", "Δ F1 (pp)", "Δ AUC (pp)"]
        L.extend(_table(t))
    A("")

    # ---------------------------------------------------------------- 11 complexity
    A("## 11. Computational complexity")
    A("")
    if not complexity.empty:
        t = complexity[["config", "architecture", "total_params_millions",
                        "trainable_params", "gflops", "model_size_mb",
                        "inference_ms_per_image"]].copy()
        t.columns = ["Config", "Architecture", "Params (M)", "Trainable params",
                     "GFLOPs", "Size (MB)", "Inference (ms/img)"]
        L.extend(_table(t))
        A("**Provenance:** parameters and GFLOPs are *calculated* exactly from the "
          "model graph (TensorFlow profiler, batch size 1). Model size, latency and "
          "memory are *measured* on this machine. **No value is estimated.**")
        A("")
        A("All timings are **CPU** measurements on a 15 W mobile processor with no "
          "CUDA GPU, and are not comparable to GPU timings reported elsewhere. The "
          "manuscript states training used an RTX 3060; that hardware was not "
          "available here.")
        A("")
        if bank.get("backbones"):
            tot = sum(sum(v.get("extraction_seconds", {}).values())
                      for v in bank["backbones"].values())
            A(f"One-off feature-extraction cost (all backbones, all variants): "
              f"**{human_seconds(tot)}**. Head training time is reported separately.")
            A("")

    # ---------------------------------------------------------------- 12 best
    A("## 12. Best-performing architecture")
    A("")
    if not cv_summary.empty:
        best = cv_summary.loc[cv_summary["accuracy_mean"].idxmax()]
        A(f"**{best['architecture']} — {best['architecture_display']}**")
        A("")
        A(f"- Accuracy: **{best['accuracy_mean_pm_sd']}%** "
          f"(95% CI {_fmt_pct(best['accuracy_ci_lower'])}–{_fmt_pct(best['accuracy_ci_upper'])}%)")
        A(f"- F1: **{best['f1_mean_pm_sd']}%**")
        A(f"- Sensitivity: **{best['sensitivity_mean_pm_sd']}%**, "
          f"Specificity: **{best['specificity_mean_pm_sd']}%**")
        A(f"- ROC-AUC: **{best['roc_auc_mean']:.4f} ± {best['roc_auc_std']:.4f}**")
        A("")
        if best["architecture"] != config.FULL_MODEL:
            full = cv_summary[cv_summary["architecture"] == config.FULL_MODEL].iloc[0]
            A(f"> Note: the best configuration is **not** the full proposed hybrid. "
              f"A5 achieved {full['accuracy_mean_pm_sd']}% versus "
              f"{best['accuracy_mean_pm_sd']}% for {best['architecture']}. "
              f"Whether that difference is statistically meaningful is addressed in "
              f"section 9.")
            A("")

    # ---------------------------------------------------------------- 13 limits
    A("## 13. Limitations")
    A("")
    for lim in [
        "**Frozen backbones.** Only stage-1 transfer learning was executed. No "
        "end-to-end fine-tuning was performed, because no CUDA GPU is available.",
        "**Sample size.** After de-duplication the binary task has "
        f"{stats.get('binary_task', {}).get('usable_images', '?')} unique images, "
        "far fewer than the 1,947 the manuscript describes. Confidence intervals are "
        "correspondingly wide.",
        "**Patient-level separation is only partial.** Real patient identifiers "
        "cover roughly 11% of the binary-task images; the rest rely on "
        "near-duplicate clustering.",
        "**Source datasets are not independent.** Two of the four folders are "
        "subsets of a third, so the corpus is effectively two sources.",
        "**Label noise.** The public sources disagree on labels for images that are "
        "verified near-duplicates, including cases contradicted by DXA T-scores.",
        "**No external validation.** All folds come from the same pooled corpus. "
        "There is no held-out external clinical cohort.",
        "**Five folds is a small sample for hypothesis testing.** Fold metrics are "
        "not independent, so p-values are descriptive rather than confirmatory.",
        "**CPU-only timing.** Latency and training-time figures reflect a mobile CPU.",
        "**Grad-CAM is not clinical validation.** It shows model attribution only.",
    ]:
        A(f"- {lim}")
    A("")

    # ---------------------------------------------------------------- 14 failures
    A("## 14. Failed experiments")
    A("")
    if not exp_status.empty:
        failed = exp_status[exp_status["status"] == "failed"]
        if failed.empty:
            A(f"None. All **{n_done}** experiments completed successfully.")
        else:
            A(f"**{len(failed)}** experiments failed:")
            A("")
            L.extend(_table(failed, ["experiment", "architecture", "fold",
                                     "error_type", "error"]))
            A("Full tracebacks are preserved in `results/logs/error_*.txt`.")
    A("")

    # ---------------------------------------------------------------- 15 repro
    A("## 15. Reproducibility")
    A("")
    A(f"- Python: `{env.get('python_version', 'n/a')}`")
    A(f"- TensorFlow: `{env.get('tensorflow_version')}`, Keras `{env.get('keras_version')}`")
    A(f"- NumPy `{env.get('numpy_version')}`, pandas `{env.get('pandas_version')}`, "
      f"scikit-learn `{env.get('sklearn_version')}`, SciPy `{env.get('scipy_version')}`")
    A(f"- Platform: `{env.get('platform')}`")
    A(f"- CPU: `{env.get('processor')}` "
      f"({env.get('cpu_count_physical')} physical / {env.get('cpu_count_logical')} logical)")
    A(f"- RAM: {env.get('total_ram_gb')} GB")
    A(f"- GPU: **{'none detected' if not env.get('gpu_available') else env.get('gpus_detected')}**; "
      f"TF built with CUDA: `{env.get('tf_build_with_cuda')}`")
    A(f"- Seeds: global `{config.GLOBAL_SEED}`, folds `{config.FOLD_SEED}`, "
      f"augmentation `{config.AUGMENTATION_SEED}`")
    A("")
    A("Full environment in `environment.txt` and `results/environment.json`; "
      "pinned dependencies in `requirements.txt`. Reproduce with `python run_all.py`.")
    A("")

    # ---------------------------------------------------------------- 16 claims
    A("## 16. Manuscript claims — supported or not")
    A("")
    A("| Manuscript claim | Status | Basis |")
    A("|---|---|---|")
    A("| Dataset of 1,947 knee X-ray images | **Not supported** | "
      f"{stats.get('raw_scan', {}).get('total_files_scanned', '?')} files exist; only "
      f"{stats.get('deduplicated', {}).get('unique_images', '?')} are unique images |")
    A("| Four independent public datasets | **Not supported** | dataset3 and dataset4 "
      "are wholly contained in dataset2; effectively two sources |")
    A("| \"Patient-wise separation was maintained\" | **Not supported** | patient ids "
      f"exist for only ~{100 * stats.get('binary_task', {}).get('fraction_with_real_patient_id', 0):.0f}% "
      "of binary-task images |")
    if not cv_summary.empty:
        full = cv_summary[cv_summary["architecture"] == config.FULL_MODEL]
        if not full.empty:
            f = full.iloc[0]
            A(f"| Hybrid Model 1 achieves 97.5% accuracy | **Not reproduced here** | "
              f"measured {f['accuracy_mean_pm_sd']}% under grouped five-fold CV on "
              "de-duplicated data with frozen backbones |")
    A("| Table 10 five-fold CV (SD = 0.158 for all metrics) | **Illustrative, not "
      "experimental** | identical SD across four different metrics is not an "
      "empirical result; replaced by Table 10 in `tables/` |")
    A("| Table 11 \"ablation study\" | **Not an ablation** | compares seven unrelated "
      "networks and mixes binary with multiclass tasks; replaced by a true component "
      "ablation |")
    A("| Table 12 estimated GFLOPs | **Superseded** | replaced with values calculated "
      "from the model graph |")
    A("| Multimodal X-ray+CT 98.1% | **No experimental basis** | "
      f"{mm.get('conclusion', 'LUMOS not available')} |")
    A("| Grad-CAM aids clinical decision-making | **Overstated** | Grad-CAM shows "
      "model attribution, not validated clinical reasoning |")
    A("| Trained on NVIDIA RTX 3060 | **Not the case here** | no CUDA GPU on this "
      "machine; all results are CPU |")
    A("")

    A("## 17. Recommended manuscript updates")
    A("")
    for i, rec in enumerate([
        "Correct the dataset section: report unique image counts after "
        "de-duplication, and state that the four sources overlap substantially.",
        "Remove or qualify the claim of patient-wise separation; state precisely "
        "which subset has patient identifiers.",
        "Replace the illustrative Table 10 and Fig. 22–23 with the measured "
        "five-fold results in `tables/table_10_cross_validation.csv` and "
        "`figures/paper/fig_22*`, `fig_23*`.",
        "Replace Table 11 / Fig. 24 with the true component ablation.",
        "Replace Table 12 / Fig. 26 with calculated FLOPs and measured latency, and "
        "label provenance explicitly.",
        "Withdraw the multimodal performance numbers (97.5 / 93.4 / 98.1) or "
        "relabel section 4.11 unambiguously as a planned design.",
        "Report the label conflicts found between the public sources — this is a "
        "genuine contribution and strengthens the paper.",
        "State the compute environment actually used.",
        "Soften the Grad-CAM interpretability claims.",
    ], start=1):
        A(f"{i}. {rec}")
    A("")

    A("---")
    A("")
    A("### Artefact index")
    A("")
    for label, path in [
        ("Dataset manifest", "data/metadata/dataset_manifest.csv"),
        ("Dataset statistics", "results/dataset_statistics.json"),
        ("Label conflicts", "data/metadata/label_conflicts.csv"),
        ("Near-duplicate pairs", "data/metadata/near_duplicate_pairs.csv"),
        ("Fold definitions", "splits/fold_*/"),
        ("Fold-level metrics", "results/cross_validation/fold_metrics.csv"),
        ("CV summary", "results/cross_validation/cv_summary.csv"),
        ("Confidence intervals", "results/cross_validation/confidence_intervals.csv"),
        ("Out-of-fold predictions", "results/cross_validation/out_of_fold_predictions.csv"),
        ("Statistical tests", "results/statistical_tests/statistical_comparison.csv"),
        ("Ablation", "results/ablation/ablation_results.csv"),
        ("Complexity", "results/computational_complexity.csv"),
        ("Experiment status", "results/experiment_status.csv"),
        ("Paper figures", "figures/paper/"),
        ("Manuscript tables", "tables/"),
        ("LUMOS feasibility", "results/multimodal/lumos_feasibility.md"),
    ]:
        A(f"- **{label}**: `{path}`")
    A("")

    out = config.RESULTS_DIR / "final_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("Final report written: %s (%d lines)", out, len(L))
    return out


if __name__ == "__main__":
    generate_report()
