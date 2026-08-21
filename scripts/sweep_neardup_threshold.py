"""Choose the near-duplicate threshold using a ground-truth false-positive test.

dataset1 assigns one distinct patient id per image. Two images with *different*
dataset1 patient ids are therefore, by construction, different patients and must
never be merged into one near-duplicate cluster. That gives a hard false-positive
signal against which a threshold can be selected rather than guessed.

Reported per threshold:
  fp_pairs      directly-verified pairs joining two different patient ids  (must be 0)
  bad_clusters  clusters containing >1 distinct patient id after transitive closure
  merged        number of verified pairs
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
from src.dataset_scan import hamming_hex  # noqa: E402

SIZES = (64, 128)


def thumb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                       dtype=np.float64)
    a -= a.mean()
    s = a.std()
    return (a / s if s > 1e-6 else a).ravel()


def main() -> None:
    m = pd.read_csv(config.METADATA_DIR / "dataset_manifest.csv")
    df = m[~m["is_duplicate"]].reset_index(drop=True)
    print(f"unique images: {len(df)}")
    pid = df["patient_id"].astype("string")
    n_pid = int(pid.notna().sum())
    print(f"images with patient id: {n_pid}")

    # candidate pairs from dHash
    buckets = defaultdict(list)
    for i, h in enumerate(df["dhash"]):
        for chunk in (h[:4], h[4:8], h[8:12]):
            buckets[chunk].append(i)
    cand = set()
    for items in buckets.values():
        if len(items) < 2 or len(items) > 400:
            continue
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = sorted((items[a], items[b]))
                if i != j and hamming_hex(df.loc[i, "dhash"], df.loc[j, "dhash"]) <= 6:
                    cand.add((i, j))
    cand = sorted(cand)
    print(f"dHash candidate pairs: {len(cand)}")

    for size in SIZES:
        cache: dict[int, np.ndarray] = {}

        def t(i: int, _s: int = size) -> np.ndarray:
            if i not in cache:
                cache[i] = thumb(df.loc[i, "filepath"], _s)
            return cache[i]

        dim = float(size * size)
        corrs = np.array([float(np.dot(t(i), t(j)) / dim) for i, j in cand])

        print(f"\n===== thumbnail {size}x{size} =====")
        print(f"{'thresh':>8} {'merged':>7} {'fp_pairs':>9} {'bad_clusters':>13} {'clusters':>9}")
        for th in (0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995):
            parent = list(range(len(df)))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            fp = 0
            merged = 0
            for (i, j), c in zip(cand, corrs):
                if c < th:
                    continue
                merged += 1
                pi, pj = pid.iloc[i], pid.iloc[j]
                if pd.notna(pi) and pd.notna(pj) and pi != pj:
                    fp += 1
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)

            groups = defaultdict(set)
            for i in range(len(df)):
                if pd.notna(pid.iloc[i]):
                    groups[find(i)].add(pid.iloc[i])
            bad = sum(1 for v in groups.values() if len(v) > 1)
            n_clusters = len({find(i) for i in range(len(df))})
            print(f"{th:>8} {merged:>7} {fp:>9} {bad:>13} {n_clusters:>9}")


if __name__ == "__main__":
    main()
