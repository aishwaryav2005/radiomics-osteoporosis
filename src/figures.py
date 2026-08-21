"""Publication-quality figure generation.

Conventions applied to every figure:

* 300 DPI, saved as both PNG and PDF
* colour-blind-safe categorical palette, validated for CVD separation
  (see ``config.PALETTE``); identity is never carried by colour alone -
  every series is also directly labelled or axis-named
* consistent typography and axis labelling across the whole set
* axes are never truncated to exaggerate differences: metric axes start at 0
  unless a zoomed panel is explicitly labelled as such, in which case the
  full-range panel is shown alongside it
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import banner, get_logger, read_json

log = get_logger("figures")

_STYLE_APPLIED = False


def apply_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": config.FIGURE_DPI,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": config.NEUTRAL_DARK,
        "axes.linewidth": 1.0,
        "grid.color": config.GRID_COLOR,
        "grid.linewidth": 0.8,
        "figure.facecolor": config.SURFACE,
        "axes.facecolor": config.SURFACE,
        "savefig.bbox": "tight",
        "savefig.facecolor": config.SURFACE,
        "pdf.fonttype": 42,       # embed real fonts, not Type 3
        "ps.fonttype": 42,
    })
    _STYLE_APPLIED = True


def save(fig, path_stem: Path, close: bool = True) -> list[Path]:
    """Save a figure as PNG and PDF at publication DPI."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in config.FIGURE_FORMATS:
        p = path_stem.with_suffix(f".{fmt}")
        fig.savefig(p, format=fmt, dpi=config.FIGURE_DPI)
        written.append(p)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    log.info("  figure: %s.{%s}", path_stem.name, ",".join(config.FIGURE_FORMATS))
    return written


def _arch_colour(arch: str) -> str:
    return config.PALETTE.get(arch, config.NEUTRAL_MID)


def _short_label(arch: str) -> str:
    """Two-line label: config id on top, backbone combination beneath."""
    cfg = config.ABLATION_CONFIGS[arch]
    abbrev = {"vgg19": "VGG19", "inceptionresnetv2": "IRv2", "mobilenetv2": "MNv2"}
    return f"{arch}\n{'+'.join(abbrev[b] for b in cfg['backbones'])}"


# ==========================================================================
# Fig 22 - fold-wise cross-validation distribution
# ==========================================================================
def fig_cross_validation(fold_metrics: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["Accuracy", "Precision", "Recall", "F1-score"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    archs = [a for a in config.ABLATION_ORDER
             if a in set(fold_metrics["architecture"])]

    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        data = [100 * fold_metrics.loc[fold_metrics["architecture"] == a, metric].to_numpy()
                for a in archs]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops={"color": config.NEUTRAL_DARK, "linewidth": 2},
                        whiskerprops={"color": config.NEUTRAL_DARK},
                        capprops={"color": config.NEUTRAL_DARK},
                        flierprops={"marker": "o", "markersize": 5,
                                    "markerfacecolor": config.NEUTRAL_MID,
                                    "markeredgecolor": "none", "alpha": 0.7})
        for patch, a in zip(bp["boxes"], archs):
            patch.set_facecolor(_arch_colour(a))
            patch.set_alpha(0.55)
            patch.set_edgecolor(_arch_colour(a))
            patch.set_linewidth(1.6)

        # individual folds, jittered, so n is visible rather than implied
        rng = np.random.default_rng(config.GLOBAL_SEED)
        for i, (a, vals) in enumerate(zip(archs, data), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=26,
                       color=_arch_colour(a), edgecolor="white", linewidth=0.8,
                       zorder=3, alpha=0.95)

        ax.set_xticks(range(1, len(archs) + 1))
        ax.set_xticklabels([_short_label(a) for a in archs])
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

    # Out-of-fold count is per model: sum the folds of a single architecture,
    # not of the whole table (which would multiply by the number of models).
    n_oof = int(fold_metrics.loc[fold_metrics["architecture"] == archs[0], "n_test"].sum())
    fig.suptitle(
        "Five-fold cross-validation: fold-wise performance distribution\n"
        f"Binary task (normal vs osteoporosis), n = {n_oof} "
        "out-of-fold predictions per model",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out)


# ==========================================================================
# Fig 23 - mean with 95% confidence intervals
# ==========================================================================
def fig_confidence_intervals(cv_summary: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    metrics = ["accuracy", "precision", "recall", "f1", "sensitivity",
               "specificity", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "Sensitivity",
              "Specificity", "ROC-AUC"]
    archs = list(cv_summary["architecture"])

    fig, ax = plt.subplots(figsize=(14, 7.2))
    n = len(archs)
    group_w = 0.82
    bar_w = group_w / n

    for i, arch in enumerate(archs):
        row = cv_summary[cv_summary["architecture"] == arch].iloc[0]
        means = np.array([100 * row[f"{m}_mean"] for m in metrics])
        los = np.array([100 * row[f"{m}_ci_lower"] for m in metrics])
        his = np.array([100 * row[f"{m}_ci_upper"] for m in metrics])
        err = np.vstack([np.clip(means - los, 0, None), np.clip(his - means, 0, None)])
        x = np.arange(len(metrics)) - group_w / 2 + bar_w * (i + 0.5)

        ax.bar(x, means, bar_w * 0.9, color=_arch_colour(arch),
               label=f"{arch}  {config.ABLATION_CONFIGS[arch]['display']}",
               edgecolor="white", linewidth=1.2, zorder=2)
        ax.errorbar(x, means, yerr=err, fmt="none", ecolor=config.NEUTRAL_DARK,
                    elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=4)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 109)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Five-fold cross-validation: mean performance with 95% confidence intervals",
                 loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.text(0.01, -0.02,
             "Error bars: 95% Student-t confidence interval across 5 folds. "
             "Axis spans the full 0-100% range; bars are not truncated. "
             "Exact values are listed in Table 10.",
             fontsize=9.5, color=config.NEUTRAL_MID)
    fig.tight_layout()
    save(fig, out)


# ==========================================================================
# Fig 24 - ablation
# ==========================================================================
def fig_ablation(ablation: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    archs = list(ablation["config"])
    means = np.array([100 * v for v in ablation["accuracy_mean"]])
    stds = np.array([100 * v for v in ablation["accuracy_std"]])
    colors = [_arch_colour(a) for a in archs]

    bars = ax1.bar(range(len(archs)), means, 0.62, yerr=stds, color=colors,
                   edgecolor="white", linewidth=1.4, capsize=5,
                   error_kw={"ecolor": config.NEUTRAL_DARK, "elinewidth": 1.3}, zorder=2)
    for b in bars[:-1]:
        b.set_alpha(0.85)
    bars[-1].set_linewidth(2.2)
    bars[-1].set_edgecolor(config.NEUTRAL_DARK)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 1.2, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=config.NEUTRAL_DARK)
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([_short_label(a) for a in archs])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 109)
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid(axis="y", alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("Ablation: mean accuracy ± SD across five folds", loc="left")
    ax1.text(0.5, -0.19, "A5 (outlined) is the full proposed hybrid model",
             transform=ax1.transAxes, ha="center", fontsize=10,
             color=config.NEUTRAL_MID)

    if not summary.empty:
        comps = list(summary["comparison"])
        deltas = np.array(summary["accuracy_abs_improvement_pp"])
        cols = [_arch_colour(c.split(" vs ")[1]) for c in comps]
        y = np.arange(len(comps))
        ax2.barh(y, deltas, 0.55, color=cols, edgecolor="white",
                 linewidth=1.3, zorder=2)
        for yi, d in zip(y, deltas):
            off = 0.06 * max(1e-9, np.abs(deltas).max())
            ax2.text(d + (off if d >= 0 else -off), yi, f"{d:+.2f} pp",
                     va="center", ha="left" if d >= 0 else "right",
                     fontsize=11, fontweight="bold", color=config.NEUTRAL_DARK)
        ax2.set_yticks(y)
        ax2.set_yticklabels([f"vs {c.split(' vs ')[1]}" for c in comps])
        ax2.invert_yaxis()
        ax2.axvline(0, color=config.NEUTRAL_DARK, linewidth=1.2)
        ax2.set_xlabel("Accuracy difference (percentage points)")
        ax2.set_title("Contribution of the full model over each reduced variant",
                      loc="left")
        ax2.grid(axis="x", alpha=0.6)
        ax2.set_axisbelow(True)
        lim = max(1.0, float(np.abs(deltas).max()) * 1.45)
        ax2.set_xlim(-lim, lim)

    fig.tight_layout()
    save(fig, out)


# ==========================================================================
# Fig 26 - computational complexity
# ==========================================================================
def fig_complexity(complexity: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.6))
    archs = list(complexity["config"])
    colors = [_arch_colour(a) for a in archs]
    x = np.arange(len(archs))

    panels = [
        ("total_params_millions", "Parameters (millions)", "Model size in parameters", "calculated"),
        ("gflops", "GFLOPs (batch size 1)", "Computational cost", "calculated"),
        ("inference_ms_per_image", "Inference latency (ms/image)", "CPU inference latency", "measured"),
    ]

    for ax, (col, ylabel, title, prov) in zip(axes, panels):
        if col not in complexity.columns or complexity[col].isna().all():
            ax.set_visible(False)
            continue
        vals = complexity[col].astype(float).to_numpy()
        bars = ax.bar(x, vals, 0.62, color=colors, edgecolor="white",
                      linewidth=1.4, zorder=2)
        bars[-1].set_edgecolor(config.NEUTRAL_DARK)
        bars[-1].set_linewidth(2.2)
        span = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        for xi, v in zip(x, vals):
            ax.text(xi, v + span * 0.025, f"{v:,.4g}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=config.NEUTRAL_DARK)
        # Short codes only on the tick - the two-line backbone labels used
        # elsewhere are too wide for three narrow side-by-side panels and
        # collide with their neighbours; the shared legend below names them.
        ax.set_xticks(x)
        ax.set_xticklabels(archs, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  [{prov}]", loc="left", fontsize=12.5)
        ax.set_ylim(0, span * 1.18)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_arch_colour(a)) for a in archs]
    labels = [f"{a}  {config.ABLATION_CONFIGS[a]['display']}" for a in archs]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
              frameon=False, fontsize=10.5)

    fig.suptitle("Computational complexity of the ablation configurations "
                 "(end-to-end models, CPU)",
                 fontsize=15, fontweight="bold")
    fig.text(0.01, -0.14,
             "Parameters and GFLOPs are calculated exactly from the model graph; "
             "latency is measured on this CPU (no CUDA GPU present). "
             "No value in this figure is an estimate.",
             fontsize=9.5, color=config.NEUTRAL_MID)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, out)


# ==========================================================================
# ROC curves (out-of-fold)
# ==========================================================================
def fig_roc(curves: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 8.4))

    for arch in config.ABLATION_ORDER:
        if arch not in curves:
            continue
        c = curves[arch]
        roc, ci = c["roc"], c["auc_ci"]
        if not roc["fpr"]:
            continue
        label = (f"{arch}  {c['display']}\n"
                 f"     AUC = {roc['auc']:.3f} "
                 f"[{ci['ci_lower']:.3f}-{ci['ci_upper']:.3f}]")
        ax.plot(roc["fpr"], roc["tpr"], color=_arch_colour(arch),
                linewidth=2.4 if arch == config.FULL_MODEL else 1.9,
                label=label, zorder=3 if arch == config.FULL_MODEL else 2,
                solid_capstyle="round")

    ax.plot([0, 1], [0, 1], color=config.NEUTRAL_MID, linestyle="--",
            linewidth=1.2, label="Chance (AUC = 0.500)", zorder=1)
    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.grid(alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("ROC curves from pooled out-of-fold predictions\n"
                 "Positive class: osteoporosis", loc="left")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.text(0.01, -0.01,
             "AUC 95% CI: stratified bootstrap percentile interval (2000 resamples) "
             "over pooled out-of-fold predictions.",
             fontsize=9.5, color=config.NEUTRAL_MID)
    fig.tight_layout()
    save(fig, out)


# ==========================================================================
# Confusion matrices
# ==========================================================================
def _draw_cm(ax, cm: np.ndarray, title: str, cmap: str = "Blues") -> None:
    classes = ["Normal", "Osteoporosis"]
    total = cm.sum()
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() if cm.max() else 1)
    thresh = (cm.max() if cm.max() else 1) * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / total if total else 0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else config.NEUTRAL_DARK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left", fontsize=12.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fig_confusion_matrices(curves: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    archs = [a for a in config.ABLATION_ORDER if a in curves]
    n = len(archs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 5.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, arch in zip(axes, archs):
        cm = np.array(curves[arch]["confusion_matrix"])
        _draw_cm(ax, cm, f"{arch}  {curves[arch]['display']}")
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion matrices from pooled out-of-fold predictions\n"
                 "Class order: Normal, Osteoporosis",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


def fig_confusion_single(curves: dict, arch: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cm = np.array(curves[arch]["confusion_matrix"])
    _draw_cm(ax, cm, f"{arch}  {curves[arch]['display']}\nPooled out-of-fold predictions")
    fig.tight_layout()
    save(fig, out)


# ==========================================================================
# Training curves
# ==========================================================================
def fig_training_curves(arch: str, fold: int, history: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    epochs = np.arange(1, len(history["loss"]) + 1)
    col = _arch_colour(arch)

    ax1.plot(epochs, 100 * np.array(history["accuracy"]), color=col,
             linewidth=2, label="Training")
    ax1.plot(epochs, 100 * np.array(history["val_accuracy"]), color=col,
             linewidth=2, linestyle="--", label="Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Accuracy", loc="left")
    ax1.legend(); ax1.grid(alpha=0.6); ax1.set_axisbelow(True)

    ax2.plot(epochs, history["loss"], color=col, linewidth=2, label="Training")
    ax2.plot(epochs, history["val_loss"], color=col, linewidth=2,
             linestyle="--", label="Validation")
    best = int(np.argmin(history["val_loss"])) + 1
    ax2.axvline(best, color=config.NEUTRAL_MID, linestyle=":", linewidth=1.5)
    ax2.annotate(f"best epoch {best}", xy=(best, min(history["val_loss"])),
                 xytext=(6, 14), textcoords="offset points",
                 fontsize=10, color=config.NEUTRAL_MID)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Binary cross-entropy loss")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Loss", loc="left")
    ax2.legend(); ax2.grid(alpha=0.6); ax2.set_axisbelow(True)

    fig.suptitle(f"{arch} — {config.ABLATION_CONFIGS[arch]['display']}  (fold {fold})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


# ==========================================================================
# Dataset composition
# ==========================================================================
def fig_dataset(stats: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    raw = stats["raw_scan"]["class_counts_including_duplicates"]
    uniq = stats["deduplicated"]["class_counts"]
    classes = ["normal", "osteopenia", "osteoporosis"]
    labels = ["Normal", "Osteopenia", "Osteoporosis"]
    x = np.arange(len(classes))

    r = [raw.get(c, 0) for c in classes]
    u = [uniq.get(c, 0) for c in classes]
    ax1.bar(x - 0.2, r, 0.38, label="Files on disk (with duplicates)",
            color=config.NEUTRAL_MID, edgecolor="white", linewidth=1.2)
    ax1.bar(x + 0.2, u, 0.38, label="Unique images after de-duplication",
            color=config.PALETTE["A5"], edgecolor="white", linewidth=1.2)
    for xi, v in zip(x - 0.2, r):
        ax1.text(xi, v + 8, str(v), ha="center", fontsize=10.5, fontweight="bold")
    for xi, v in zip(x + 0.2, u):
        ax1.text(xi, v + 8, str(v), ha="center", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Number of images")
    ax1.set_title("Class distribution before and after de-duplication", loc="left")
    ax1.legend(); ax1.grid(axis="y", alpha=0.6); ax1.set_axisbelow(True)

    ov = stats["duplicates"]["cross_dataset_pixel_overlap"]
    names = list(ov)
    mat = np.array([[ov[a][b] for b in names] for a in names])
    im = ax2.imshow(mat, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax2.text(j, i, str(mat[i, j]), ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if mat[i, j] > mat.max() * 0.55 else config.NEUTRAL_DARK)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names)
    ax2.set_title("Shared unique images between source datasets", loc="left")
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, label="Images in common")

    fig.suptitle("Dataset composition and inter-dataset overlap",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out)


# ==========================================================================
# Orchestration
# ==========================================================================
def generate_all_figures() -> dict:
    banner("STAGE: FIGURE GENERATION")
    config.ensure_dirs()
    apply_style()
    written: list[str] = []

    fold_metrics = pd.read_csv(config.CV_DIR / "fold_metrics.csv")
    cv_summary = pd.read_csv(config.CV_DIR / "cv_summary.csv")
    curves = read_json(config.CV_DIR / "roc_curves.json", default={}) or {}

    log.info("Paper figures ...")
    fig_cross_validation(fold_metrics, config.PAPER_FIG_DIR / "fig_22_real_cross_validation")
    fig_confidence_intervals(cv_summary, config.PAPER_FIG_DIR / "fig_23_real_cv_confidence_intervals")

    ablation_path = config.ABLATION_DIR / "ablation_results.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        summary_path = config.ABLATION_DIR / "ablation_summary.csv"
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        fig_ablation(ablation, summary, config.PAPER_FIG_DIR / "fig_24_real_ablation")

    cpath = config.COMPLEXITY_DIR / "computational_complexity.csv"
    if cpath.exists():
        fig_complexity(pd.read_csv(cpath), config.PAPER_FIG_DIR / "fig_26_real_complexity")

    log.info("ROC and confusion matrices ...")
    if curves:
        fig_roc(curves, config.ROC_FIG_DIR / "roc_out_of_fold")
        fig_roc(curves, config.PAPER_FIG_DIR / "fig_25_real_roc_curves")
        fig_confusion_matrices(curves, config.CM_FIG_DIR / "confusion_matrices_all")
        if config.FULL_MODEL in curves:
            fig_confusion_single(curves, config.FULL_MODEL,
                                 config.CM_FIG_DIR / "confusion_matrix_A5_full_hybrid")

    log.info("Training curves ...")
    from .train import run_dir

    for arch in config.ABLATION_ORDER:
        sub = fold_metrics[fold_metrics["architecture"] == arch]
        if sub.empty:
            continue
        best_fold = int(sub.loc[sub["accuracy"].idxmax(), "fold"])
        hist = read_json(run_dir(arch, best_fold) / "history.json")
        if hist:
            fig_training_curves(arch, best_fold, hist,
                                config.CURVES_FIG_DIR / f"training_curves_{arch}_fold{best_fold}")

    log.info("Dataset figure ...")
    stats = read_json(config.RESULTS_DIR / "dataset_statistics.json")
    if stats:
        fig_dataset(stats, config.FIGURES_DIR / "dataset_composition")

    for p in sorted(config.FIGURES_DIR.rglob("*.png")):
        written.append(str(p.relative_to(config.PROJECT_ROOT)))
    log.info("Generated %d PNG figures (each also saved as PDF)", len(written))
    return {"figures": written, "count": len(written)}


if __name__ == "__main__":
    generate_all_figures()
