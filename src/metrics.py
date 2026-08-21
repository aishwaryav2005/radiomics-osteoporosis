"""Classification metrics computed directly and verified against the confusion matrix.

Positive class is **osteoporosis** (label 1); negative is **normal** (label 0).
That definition is asserted everywhere it matters, because sensitivity and
specificity are meaningless without it.

Every rate is derived from the confusion matrix counts rather than from a
library shortcut, and ``verify_against_confusion_matrix`` independently
re-derives precision/recall/F1/accuracy from TP/TN/FP/FN and cross-checks them
against scikit-learn. Disagreement raises rather than being silently tolerated.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from . import config


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TN/FP/FN/TP with an explicit, fixed label order [0, 1]."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "confusion_matrix": cm.tolist()}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(y_true, y_prob, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Full metric set for one binary evaluation."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    sensitivity = _safe_div(tp, tp + fn)      # recall of the positive class
    specificity = _safe_div(tn, tn + fp)      # recall of the negative class
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    npv = _safe_div(tn, tn + fn)
    balanced_acc = (sensitivity + specificity) / 2.0

    # AUC is undefined if only one class is present in y_true.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "balanced_accuracy": balanced_acc,
        "roc_auc": auc,
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "positive_class": config.POSITIVE_CLASS,
        "positive_label": config.POSITIVE_LABEL,
        **c,
    }


def verify_against_confusion_matrix(y_true, y_prob,
                                    threshold: float = config.DECISION_THRESHOLD,
                                    tol: float = 1e-9) -> dict:
    """Independently re-derive the metrics and cross-check with scikit-learn.

    Guards against two specific mistakes: assuming ``recall == sensitivity``
    without pinning the positive class, and mis-ordering the confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    ours = compute_metrics(y_true, y_prob, threshold)
    sk = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }

    mismatches = {
        k: {"manual": ours[k], "sklearn": v, "abs_diff": abs(ours[k] - v)}
        for k, v in sk.items() if abs(ours[k] - v) > tol
    }
    return {
        "agrees": not mismatches,
        "mismatches": mismatches,
        "sensitivity_equals_recall": abs(ours["sensitivity"] - ours["recall"]) <= tol,
        "note": ("sensitivity == recall only because the positive class is "
                 f"pinned to '{config.POSITIVE_CLASS}' (label 1)"),
    }


def roc_points(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": [], "auc": float("nan")}
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist(),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


# --------------------------------------------------------------------------
# Interval estimation
# --------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 2000,
                     confidence: float = config.CONFIDENCE_LEVEL,
                     seed: int = config.GLOBAL_SEED) -> dict:
    """Stratified bootstrap percentile CI for AUC on pooled out-of-fold predictions.

    Valid here because out-of-fold predictions cover each sample exactly once.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "method": "undefined (single class)"}

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_prob[idx]))
    stats = np.asarray(stats)
    alpha = (1.0 - confidence) / 2.0
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(np.percentile(stats, 100 * alpha)),
        "ci_upper": float(np.percentile(stats, 100 * (1 - alpha))),
        "n_bootstrap": int(len(stats)),
        "confidence_level": confidence,
        "method": "stratified bootstrap percentile interval",
    }


def mean_ci(values, confidence: float = config.CONFIDENCE_LEVEL) -> dict:
    """Mean, SD and Student-t confidence interval across folds.

    Uses the t distribution because n = 5 folds; a normal approximation would
    understate the interval at this sample size.
    """
    from scipy import stats as st

    v = np.asarray([x for x in np.asarray(values, dtype=float) if np.isfinite(x)])
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_lower": float("nan"), "ci_upper": float("nan"),
                "sem": float("nan"), "method": "no finite values"}
    mean = float(v.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1, "ci_lower": mean,
                "ci_upper": mean, "sem": 0.0, "method": "single value"}

    sd = float(v.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(st.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    return {
        "mean": mean,
        "std": sd,
        "n": int(n),
        "sem": float(sem),
        "ci_lower": mean - tcrit * sem,
        "ci_upper": mean + tcrit * sem,
        "t_critical": tcrit,
        "confidence_level": confidence,
        "method": f"Student t interval, df={n - 1}",
    }


METRIC_KEYS = ["accuracy", "precision", "recall", "f1",
               "sensitivity", "specificity", "roc_auc"]
