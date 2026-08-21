"""Known-answer tests for the pipeline's numerical core.

These guard the parts where a silent error would corrupt every reported result:
metric derivation, the sensitivity/specificity definitions, the confidence
interval, Holm-Bonferroni adjustment, and split integrity.

Run:  python scripts/self_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_metrics, mean_ci, verify_against_confusion_matrix,
)
from src.stats_tests import holm_bonferroni  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, expected, tol: float = 1e-9) -> None:
    ok = (abs(got - expected) <= tol) if isinstance(got, (int, float)) else (got == expected)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, expected {expected!r}")
    if not ok:
        FAILURES.append(name)


def test_metrics() -> None:
    print("\nMetrics (6 TP, 2 FN, 1 FP, 11 TN)")
    y = np.array([1] * 8 + [0] * 12)
    p = np.array([0.9] * 6 + [0.1] * 2 + [0.8] * 1 + [0.2] * 11)
    m = compute_metrics(y, p)
    check("tp", m["tp"], 6)
    check("fn", m["fn"], 2)
    check("fp", m["fp"], 1)
    check("tn", m["tn"], 11)
    check("accuracy", m["accuracy"], 17 / 20)
    check("sensitivity = TP/(TP+FN)", m["sensitivity"], 6 / 8)
    check("specificity = TN/(TN+FP)", m["specificity"], 11 / 12)
    check("precision", m["precision"], 6 / 7)
    check("f1", m["f1"], 2 * (6 / 7) * (6 / 8) / ((6 / 7) + (6 / 8)))
    check("npv", m["npv"], 11 / 13)
    check("positive class pinned", m["positive_class"], config.POSITIVE_CLASS)
    v = verify_against_confusion_matrix(y, p)
    check("agrees with sklearn", v["agrees"], True)
    check("sensitivity == recall", v["sensitivity_equals_recall"], True)


def test_specificity_is_not_recall() -> None:
    """Guard the specific error of reporting recall where specificity is meant."""
    print("\nSensitivity and specificity must differ on an asymmetric case")
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    m = compute_metrics(y, p)
    check("sensitivity", m["sensitivity"], 1.0)
    check("specificity", m["specificity"], 0.5)
    check("they are not equal", m["sensitivity"] != m["specificity"], True)


def test_mean_ci() -> None:
    print("\nStudent-t confidence interval (n=5)")
    ci = mean_ci([0.90, 0.92, 0.88, 0.91, 0.89])
    check("mean", ci["mean"], 0.90, 1e-12)
    check("sd (ddof=1)", round(ci["std"], 6), round(float(np.std([0.90, 0.92, 0.88, 0.91, 0.89], ddof=1)), 6))
    check("n", ci["n"], 5)
    check("t critical df=4", round(ci["t_critical"], 4), 2.7764)
    half = ci["t_critical"] * ci["sem"]
    check("ci lower", round(ci["ci_lower"], 8), round(0.90 - half, 8))
    check("ci upper", round(ci["ci_upper"], 8), round(0.90 + half, 8))


def test_holm() -> None:
    print("\nHolm-Bonferroni step-down (with monotonicity)")
    adj, rej = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # sorted: .005*4=.02 | .01*3=.03 | .03*2=.06 | .04*1=.04 -> monotone max .06
    for i, exp in enumerate([0.03, 0.06, 0.06, 0.02]):
        check(f"adjusted p[{i}]", round(adj[i], 10), exp)
    check("rejections at alpha=0.05", rej, [True, False, False, True])
    adj2, _ = holm_bonferroni([0.5, 0.6], alpha=0.05)
    check("adjusted p never exceeds 1", max(adj2) <= 1.0, True)


def test_splits() -> None:
    print("\nSplit integrity")
    import pandas as pd

    if not (config.SPLITS_DIR / "fold_1" / "train.csv").exists():
        print("  [SKIP] splits not generated yet")
        return
    all_test = []
    for f in range(1, config.N_FOLDS + 1):
        d = config.SPLITS_DIR / f"fold_{f}"
        tr = pd.read_csv(d / "train.csv")
        va = pd.read_csv(d / "validation.csv")
        te = pd.read_csv(d / "test.csv")
        for key in ("group_id", "pixel_hash", "relpath"):
            check(f"fold{f} train/test disjoint on {key}",
                  len(set(tr[key]) & set(te[key])), 0)
            check(f"fold{f} train/val disjoint on {key}",
                  len(set(tr[key]) & set(va[key])), 0)
            check(f"fold{f} val/test disjoint on {key}",
                  len(set(va[key]) & set(te[key])), 0)
        all_test += te["relpath"].tolist()
    check("every image tested exactly once",
          len(all_test) == len(set(all_test)), True)


def main() -> int:
    print("=" * 70)
    print("PIPELINE SELF-TEST")
    print("=" * 70)
    test_metrics()
    test_specificity_is_not_recall()
    test_mean_ci()
    test_holm()
    test_splits()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
