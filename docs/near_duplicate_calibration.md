# Near-duplicate detection: threshold calibration

This note records how the near-duplicate threshold used by `src/dataset_scan.py`
was chosen. It is evidence from this corpus, not a borrowed default.

Reproduce with:

```
python scripts/calibrate_neardup.py
python scripts/sweep_neardup_threshold.py
```

## Why a raw dHash is not sufficient

A 64-bit difference hash is a good *candidate generator* but a poor decision
rule on knee radiographs, because most of them share the same coarse layout
(dark background, bright centred bone). Measured on this corpus:

| dHash candidate pair | pixel correlation | verdict |
|---|---|---|
| `Normal_184.jpg` vs `Osteoporosis_110.jpg` (Hamming **0**) | 0.880 | different knees |
| `Normal_158.jpg` vs `Osteoporosis_105.jpg` (Hamming **0**) | 0.937 | different knees |

Grouping on dHash alone merged 1050 pairs and produced 12 mixed-class clusters
that were simply similar-looking, not duplicated.

## Verification step

Each dHash candidate (Hamming ≤ 6) is verified by normalised pixel correlation
on a 64×64 grayscale thumbnail. Correlation band vs. cross-class rate over 2 293
candidate pairs in the binary subset:

| correlation band | pairs | cross-class | cross-class rate |
|---|---|---|---|
| [0.00, 0.90) | 1080 | 403 | 37.3 % |
| [0.90, 0.95) | 654 | 261 | 39.9 % |
| [0.95, 0.97) | 146 | 64 | 43.8 % |
| [0.97, 0.98) | 36 | 10 | 27.8 % |
| [0.98, 0.99) | 8 | 2 | 25.0 % |
| **[0.99, 0.995)** | **17** | **0** | **0.0 %** |
| **[0.995, 1.01)** | **352** | **7** | **2.0 %** |

Below 0.98 a quarter to a half of all pairs join different classes — the
signature of "merely similar", not "duplicate". At and above 0.99 that collapses.
The threshold is set at **0.99**.

## False-positive test against ground truth

dataset1 assigns one patient id per image, so two images with *different*
dataset1 patient ids must never be merged. That gives a hard false-positive
signal. Sweeping the threshold over 5 443 dHash candidate pairs on the 841
unique images:

| thumbnail | threshold | pairs merged | FP pairs (different patient ids) | clusters mixing patient ids |
|---|---|---|---|---|
| 64×64 | **0.99** | **376** | **0** | **0** |
| 64×64 | 0.995 | 359 | 0 | 0 |
| 64×64 | 0.999 | 244 | 0 | 0 |
| 128×128 | 0.99 | 362 | 0 | 0 |
| 128×128 | 0.995 | 302 | 0 | 0 |

No threshold in the tested range produces a false merge. Since under-merging
risks leakage while over-merging only makes cross-validation more conservative,
the most inclusive validated setting — **64×64 thumbnails, correlation ≥ 0.99** —
is used.

## A caveat this test exposed

`OP101.jpg` and `OP135.jpg` in dataset1 have an **identical pixel hash** but are
filed as two different patients with different T-scores (−2.29 and −2.15). The
same holds for `OP99.jpg` / `OP105.jpg` (−1.61 and −2.28). The source dataset
itself files one radiograph under multiple patient identities.

Consequence: dataset1's patient ids cannot be treated as a fully reliable
patient key. Where such a cluster's T-scores still imply a single class, the
class is used and the exact T-score is treated as ambiguous; where they imply
different classes, the cluster is excluded (see `data/metadata/label_conflicts.json`).
