"""Paired statistical comparison of the full hybrid against each reduced variant.

Because all architectures are trained and tested on identical folds, the fold
metrics are paired and paired tests are appropriate. Two complementary tests are
reported for every comparison:

* **paired t-test** - parametric, assumes roughly normal fold differences
* **Wilcoxon signed-rank** - non-parametric fallback; with n = 5 folds its
  smallest attainable two-sided p-value is 0.0625, so it *cannot* reach p < 0.05.
  That is a property of the sample size, not evidence of no effect, and is
  reported explicitly rather than quietly ignored.

Multiplicity is controlled with Holm-Bonferroni across the four comparisons
within each metric. Significance is only claimed when the Holm-adjusted p is
below alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .metrics import METRIC_KEYS
from .utils import banner, get_logger, write_json

log = get_logger("stats")


def holm_bonferroni(pvals: list[float], alpha: float = config.ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down adjustment. Returns (adjusted p, reject)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)          # enforce monotonicity
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist(), (adjusted < alpha).tolist()


def cohens_d_paired(diff: np.ndarray) -> float:
    """Effect size for paired differences (mean / SD of the differences)."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def compare(fold_metrics: pd.DataFrame, reference: str = config.FULL_MODEL) -> pd.DataFrame:
    """Paired tests of ``reference`` vs every other architecture, per metric."""
    rows: list[dict] = []
    others = [a for a in config.ABLATION_ORDER if a != reference]

    for metric in METRIC_KEYS:
        pt_p, wx_p, staged = [], [], []
        for other in others:
            ref = (fold_metrics[fold_metrics["architecture"] == reference]
                   .sort_values("fold")[metric].to_numpy())
            oth = (fold_metrics[fold_metrics["architecture"] == other]
                   .sort_values("fold")[metric].to_numpy())
            if len(ref) != len(oth) or len(ref) < 2:
                continue
            diff = ref - oth

            if np.allclose(diff, 0):
                t_stat, t_p = 0.0, 1.0
                w_stat, w_p = 0.0, 1.0
                note = "identical fold values; tests degenerate"
            else:
                note = ""
                t_stat, t_p = stats.ttest_rel(ref, oth)
                try:
                    w_stat, w_p = stats.wilcoxon(ref, oth, zero_method="wilcox",
                                                 alternative="two-sided")
                except ValueError as exc:      # all-zero differences
                    w_stat, w_p = float("nan"), 1.0
                    note = f"wilcoxon undefined: {exc}"

            staged.append({
                "metric": metric,
                "reference": reference,
                "reference_display": config.ABLATION_CONFIGS[reference]["display"],
                "comparator": other,
                "comparator_display": config.ABLATION_CONFIGS[other]["display"],
                "comparison": f"{reference} vs {other}",
                "n_folds": int(len(ref)),
                "reference_mean": float(ref.mean()),
                "comparator_mean": float(oth.mean()),
                "mean_difference": float(diff.mean()),
                "difference_std": float(diff.std(ddof=1)),
                "cohens_d_paired": cohens_d_paired(diff),
                "paired_t_statistic": float(t_stat),
                "paired_t_p_value": float(t_p),
                "wilcoxon_statistic": float(w_stat),
                "wilcoxon_p_value": float(w_p),
                "note": note,
            })
            pt_p.append(float(t_p))
            wx_p.append(float(w_p))

        if not staged:
            continue
        t_adj, t_rej = holm_bonferroni(pt_p)
        w_adj, w_rej = holm_bonferroni(wx_p)
        for r, ta, tr, wa, wr in zip(staged, t_adj, t_rej, w_adj, w_rej):
            r["paired_t_p_holm"] = ta
            r["paired_t_significant_holm"] = bool(tr)
            r["wilcoxon_p_holm"] = wa
            r["wilcoxon_significant_holm"] = bool(wr)
            r["alpha"] = config.ALPHA
            r["correction"] = "holm-bonferroni across the 4 comparisons within this metric"
            rows.append(r)

    return pd.DataFrame(rows)


def run_statistical_tests() -> dict:
    banner("STAGE: STATISTICAL SIGNIFICANCE TESTING")
    config.ensure_dirs()

    fold_metrics = pd.read_csv(config.CV_DIR / "fold_metrics.csv")
    df = compare(fold_metrics)
    df.to_csv(config.STATS_DIR / "statistical_comparison.csv",
              index=False, encoding="utf-8")

    n_folds = int(fold_metrics.groupby("architecture")["fold"].nunique().max())
    min_wilcoxon_p = float(2 / (2 ** n_folds)) if n_folds <= 25 else 0.0

    caveats = {
        "n_folds": n_folds,
        "wilcoxon_minimum_attainable_two_sided_p": min_wilcoxon_p,
        "wilcoxon_can_reach_alpha": bool(min_wilcoxon_p < config.ALPHA),
        "interpretation": (
            f"With n={n_folds} paired folds the Wilcoxon signed-rank test cannot "
            f"produce a two-sided p below {min_wilcoxon_p:.4f}. A non-significant "
            "Wilcoxon result at this sample size is therefore uninformative on its "
            "own and must not be read as evidence of equivalence."
        ),
        "folds_are_not_independent": (
            "Cross-validation folds share training data, so paired tests across "
            "folds violate strict independence and their p-values are "
            "anti-conservative. They are reported as a descriptive comparison, "
            "not as a confirmatory hypothesis test."
        ),
    }
    write_json(config.STATS_DIR / "statistical_caveats.json", caveats)

    acc = df[df["metric"] == "accuracy"]
    log.info("-" * 100)
    log.info("%-34s %10s %10s %10s %10s %6s", "Comparison (accuracy)",
             "mean diff", "t", "p", "p_holm", "sig")
    log.info("-" * 100)
    for _, r in acc.iterrows():
        log.info("%-34s %10.4f %10.3f %10.4f %10.4f %6s",
                 f"A5 vs {r['comparator']}", r["mean_difference"],
                 r["paired_t_statistic"], r["paired_t_p_value"],
                 r["paired_t_p_holm"], "yes" if r["paired_t_significant_holm"] else "no")
    log.info("-" * 100)
    log.warning(caveats["interpretation"])

    return {"n_comparisons": int(len(df)), "caveats": caveats,
            "table": df.to_dict(orient="records")}


if __name__ == "__main__":
    run_statistical_tests()
