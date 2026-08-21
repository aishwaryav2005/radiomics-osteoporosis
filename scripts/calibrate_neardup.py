"""Calibrate the near-duplicate verification threshold on the real data.

The 64-bit dHash alone collapses radiographs that merely share a global layout
(dark background, centred bright bone). This script measures the distribution of
true pixel similarity among dHash candidate pairs so the verification threshold
can be chosen from evidence rather than guessed.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

THUMB = 64


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def thumb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(
            im.convert("L").resize((THUMB, THUMB), Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
    a -= a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def main() -> None:
    m = pd.read_csv(config.METADATA_DIR / "dataset_manifest.csv")
    df = m[m["usable_binary"] == True].reset_index(drop=True)  # noqa: E712
    print(f"binary images: {len(df)}")

    cache: dict[str, np.ndarray] = {}

    def corr(i: int, j: int) -> float:
        for k in (i, j):
            if k not in cache:
                cache[k] = thumb(df.loc[k, "filepath"])
        a, b = cache[i], cache[j]
        return float(np.mean(a * b))

    # dHash candidate pairs
    buckets = defaultdict(list)
    for i, h in enumerate(df["dhash"]):
        buckets[h[:4]].append(i)
        buckets[h[4:8]].append(i)

    pairs = set()
    for items in buckets.values():
        if len(items) < 2 or len(items) > 300:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = sorted((items[a], items[b]))
                if ham(df.loc[i, "dhash"], df.loc[j, "dhash"]) <= 6:
                    pairs.add((i, j))

    print(f"dHash candidate pairs (hamming<=6): {len(pairs)}")

    rows = []
    for i, j in sorted(pairs):
        c = corr(i, j)
        rows.append({
            "i": i, "j": j,
            "hamming": ham(df.loc[i, "dhash"], df.loc[j, "dhash"]),
            "corr": c,
            "same_class": df.loc[i, "class"] == df.loc[j, "class"],
            "cls_i": df.loc[i, "class"], "cls_j": df.loc[j, "class"],
            "file_i": df.loc[i, "filename"], "file_j": df.loc[j, "filename"],
            "src_i": df.loc[i, "dataset_source"], "src_j": df.loc[j, "dataset_source"],
        })
    r = pd.DataFrame(rows)
    r.to_csv(config.METADATA_DIR / "neardup_calibration.csv", index=False)

    print("\ncorrelation distribution over candidate pairs:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]:
        print(f"  q{q:<5} = {r['corr'].quantile(q):.4f}")

    print("\npairs by correlation band (and how many cross class):")
    bands = [(0.0, 0.90), (0.90, 0.95), (0.95, 0.97), (0.97, 0.98),
             (0.98, 0.99), (0.99, 0.995), (0.995, 1.01)]
    for lo, hi in bands:
        sel = r[(r["corr"] >= lo) & (r["corr"] < hi)]
        if len(sel) == 0:
            continue
        cross = int((~sel["same_class"]).sum())
        print(f"  [{lo:.3f},{hi:.3f}): n={len(sel):5d}  cross-class={cross:5d} "
              f"({100*cross/len(sel):5.1f}%)")

    print("\nHighest-correlation cross-class pairs (candidate label conflicts):")
    cc = r[~r["same_class"]].sort_values("corr", ascending=False).head(12)
    for _, x in cc.iterrows():
        print(f"  corr={x['corr']:.4f} ham={x['hamming']:2d}  "
              f"{x['src_i']}/{x['file_i']} [{x['cls_i']}]  vs  "
              f"{x['src_j']}/{x['file_j']} [{x['cls_j']}]")

    print("\nHighest-correlation same-class pairs (true near-duplicates):")
    sc = r[r["same_class"]].sort_values("corr", ascending=False).head(8)
    for _, x in sc.iterrows():
        print(f"  corr={x['corr']:.4f} ham={x['hamming']:2d}  "
              f"{x['src_i']}/{x['file_i']}  vs  {x['src_j']}/{x['file_j']}")


if __name__ == "__main__":
    main()
