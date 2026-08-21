"""Fixed stratified, group-aware five-fold splits for the binary ablation.

Design decisions, all of which are recorded in ``splits/split_manifest.json``:

* Exact pixel duplicates are dropped before splitting, so an image can never
  appear in two subsets under a different filename.
* Splitting is *grouped*: the unit of assignment is ``group_id``, which is a
  real patient id where one exists (dataset1) and a near-duplicate image
  cluster otherwise. This is strictly stronger than plain image-level
  splitting and is the honest ceiling given datasets 2-4 carry no patient
  metadata.
* Folds are stratified on the binary class so each fold preserves the overall
  normal/osteoporosis ratio.
* The five folds are generated once and written to disk. Every architecture in
  the ablation reads exactly these files, so model comparisons are paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from . import config
from .utils import banner, get_logger, write_json

log = get_logger("splits")


def load_manifest() -> pd.DataFrame:
    path = config.METADATA_DIR / "dataset_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the dataset stage first (`--stage dataset`)."
        )
    return pd.read_csv(path)


def binary_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.loc[manifest["usable_binary"] == True].copy()  # noqa: E712
    df = df.reset_index(drop=True)
    df["label"] = df["class"].map(config.BINARY_LABELS).astype(int)
    return df


def _check_no_leakage(train: pd.DataFrame, val: pd.DataFrame,
                      test: pd.DataFrame, fold: int) -> dict:
    """Hard assertion that no group, pixel hash or file path crosses subsets."""
    issues = []
    subsets = {"train": train, "validation": val, "test": test}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("group_id", "pixel_hash", "relpath"):
            overlap = set(subsets[a][key]) & set(subsets[b][key])
            if overlap:
                issues.append({
                    "fold": fold, "between": f"{a}|{b}", "key": key,
                    "n_overlap": len(overlap),
                    "examples": sorted(map(str, overlap))[:5],
                })
    if issues:
        for i in issues:
            log.error("LEAKAGE in fold %d: %s", fold, i)
        raise AssertionError(f"Leakage detected in fold {fold}: {issues}")
    return {
        "fold": fold,
        "group_overlap": 0, "pixel_hash_overlap": 0, "path_overlap": 0,
        "verified": True,
    }


def generate_folds() -> dict:
    banner("STAGE: FIVE-FOLD SPLIT GENERATION")
    config.ensure_dirs()

    manifest = load_manifest()
    df = binary_subset(manifest)
    log.info("Binary subset: %d images (%s)", len(df),
             df["class"].value_counts().to_dict())
    log.info("Grouping units: %d (%s)", df["group_id"].nunique(),
             df["group_kind"].value_counts().to_dict())

    y = df["label"].to_numpy()
    groups = df["group_id"].to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.FOLD_SEED
    )

    fold_records = []
    leakage_checks = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(sgkf.split(df, y, groups), start=1):
        trainval = df.iloc[trainval_idx]
        test = df.iloc[test_idx]

        # Carve a validation set out of train+val, grouped and stratified.
        tv_groups = trainval["group_id"].to_numpy()
        # Group-level stratification label = majority class of the group.
        grp_frame = (
            trainval.groupby("group_id")["label"]
            .agg(lambda s: int(round(s.mean())))
            .reset_index()
        )
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.VALIDATION_FRACTION_OF_TRAINVAL,
            random_state=config.FOLD_SEED + fold_idx,
        )
        g_train_i, g_val_i = next(sss.split(grp_frame, grp_frame["label"]))
        val_groups = set(grp_frame.iloc[g_val_i]["group_id"])

        val = trainval[np.isin(tv_groups, list(val_groups))]
        train = trainval[~np.isin(tv_groups, list(val_groups))]

        leakage_checks.append(_check_no_leakage(train, val, test, fold_idx))

        fold_dir = config.SPLITS_DIR / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cols = ["relpath", "filepath", "filename", "dataset_source", "class",
                "label", "patient_id", "group_id", "group_kind", "pixel_hash"]
        train[cols].to_csv(fold_dir / "train.csv", index=False, encoding="utf-8")
        val[cols].to_csv(fold_dir / "validation.csv", index=False, encoding="utf-8")
        test[cols].to_csv(fold_dir / "test.csv", index=False, encoding="utf-8")

        rec = {
            "fold": fold_idx,
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "train_class_counts": {str(k): int(v) for k, v in
                                   train["class"].value_counts().sort_index().items()},
            "validation_class_counts": {str(k): int(v) for k, v in
                                        val["class"].value_counts().sort_index().items()},
            "test_class_counts": {str(k): int(v) for k, v in
                                  test["class"].value_counts().sort_index().items()},
            "train_positive_rate": round(float(train["label"].mean()), 4),
            "validation_positive_rate": round(float(val["label"].mean()), 4),
            "test_positive_rate": round(float(test["label"].mean()), 4),
            "n_groups_train": int(train["group_id"].nunique()),
            "n_groups_validation": int(val["group_id"].nunique()),
            "n_groups_test": int(test["group_id"].nunique()),
        }
        fold_records.append(rec)
        log.info(
            "fold %d | train %4d (pos %.3f) | val %3d (pos %.3f) | test %3d (pos %.3f)",
            fold_idx, rec["n_train"], rec["train_positive_rate"],
            rec["n_validation"], rec["validation_positive_rate"],
            rec["n_test"], rec["test_positive_rate"],
        )

    # Every image must appear in exactly one test fold (out-of-fold coverage).
    test_paths: list[str] = []
    for f in range(1, config.N_FOLDS + 1):
        test_paths += pd.read_csv(config.SPLITS_DIR / f"fold_{f}" / "test.csv")["relpath"].tolist()
    coverage_ok = len(test_paths) == len(set(test_paths)) == len(df)
    if not coverage_ok:
        raise AssertionError(
            f"Out-of-fold coverage broken: {len(test_paths)} test rows, "
            f"{len(set(test_paths))} unique, {len(df)} images."
        )
    log.info("Out-of-fold coverage verified: every image is tested exactly once.")

    payload = {
        "n_folds": config.N_FOLDS,
        "fold_seed": config.FOLD_SEED,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "validation_carved_with": "StratifiedShuffleSplit over groups",
        "validation_fraction_of_trainval": config.VALIDATION_FRACTION_OF_TRAINVAL,
        "task": "binary: normal (0) vs osteoporosis (1)",
        "total_images": int(len(df)),
        "total_groups": int(df["group_id"].nunique()),
        "group_kind_counts": {str(k): int(v) for k, v in
                              df["group_kind"].value_counts().items()},
        "class_counts": {str(k): int(v) for k, v in
                         df["class"].value_counts().sort_index().items()},
        "folds": fold_records,
        "leakage_checks": leakage_checks,
        "out_of_fold_coverage_verified": coverage_ok,
        "patient_level_separation": {
            "achieved_for": "dataset1 only (real patient identifiers present)",
            "fallback_for_other_datasets": (
                "near-duplicate image clusters are treated as groups; this "
                "prevents duplicate/near-duplicate leakage but is NOT the same "
                "as verified patient-level separation"
            ),
        },
    }
    write_json(config.SPLITS_DIR / "split_manifest.json", payload)
    write_json(config.RESULTS_DIR / "folds" / "fold_summary.json", payload)
    pd.DataFrame(fold_records).to_csv(
        config.RESULTS_DIR / "folds" / "fold_summary.csv", index=False, encoding="utf-8")
    log.info("Split manifest written to %s", config.SPLITS_DIR / "split_manifest.json")
    return payload


if __name__ == "__main__":
    generate_folds()
