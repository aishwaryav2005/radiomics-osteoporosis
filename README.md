# Real Five-Fold Cross-Validation and Ablation Study
### Radiomics-Driven Knee X-Ray Analysis for Bone Mineral Density Risk Stratification

A fully automated, reproducible experimental pipeline that replaces the
manuscript's *illustrative* cross-validation and ablation sections with
**measured** results.

> **Scientific integrity.** Nothing in `results/`, `figures/` or `tables/` is
> simulated, estimated-then-presented-as-measured, or copied from the
> manuscript's illustrative tables. Where an analysis could not be run with the
> available data or hardware, the pipeline says so instead of producing a number.

---

## 1. Quick start

```bash
# 1. create the environment (Python 3.12; TensorFlow does not support 3.13/3.14)
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. put the four source archives' contents under data/raw/
#    data/raw/dataset1/, dataset2/, dataset3/, dataset4/

# 3. run everything
python run_all.py
```

Individual stages:

```bash
python run_all.py --stage dataset      # scan, hash, de-duplicate, manifest
python run_all.py --stage folds        # fixed five-fold splits
python run_all.py --stage features     # frozen-backbone feature bank (~26 min CPU)
python run_all.py --stage train        # 25 training runs
python run_all.py --stage evaluate
python run_all.py --stage stats
python run_all.py --stage ablation
python run_all.py --stage figures
python run_all.py --stage tables
python run_all.py --stage report

python run_all.py --list-stages        # show all stages
python run_all.py --smoke-test         # insert a smoke test before training
python run_all.py --from-stage evaluate
```

Training is **resumable**. Completed runs are recorded in `results/status.json`
and skipped on restart; pass `--force` to recompute.

---

## 2. What the experiment does

**Task.** Binary classification: `normal` (0) vs `osteoporosis` (1). Positive
class is osteoporosis. Osteopenia is retained in the manifest but excluded from
the binary ablation.

**Ablation configurations** — each is the *same* fusion architecture with
branches removed, so each isolates one component's contribution:

| Config | Backbones | Branch removed | Fused dim |
|---|---|---|---|
| A1 | VGG19 | IRv2 + MNv2 | 512 |
| A2 | VGG19 + InceptionResNetV2 | MobileNetV2 | 2048 |
| A3 | VGG19 + MobileNetV2 | InceptionResNetV2 | 1792 |
| A4 | InceptionResNetV2 + MobileNetV2 | VGG19 | 2816 |
| **A5** | **VGG19 + IRv2 + MNv2** | none — full proposed model | **3328** |

5 architectures × 5 folds = **25 real training runs**, all on identical folds so
comparisons are paired.

---

## 3. Two things worth knowing before reading the results

### 3.1 The dataset is far smaller than it appears

The four source folders contain 1,727 image files, but only **841 are unique
images**. Roughly half the corpus is duplicated:

- `dataset3` and `dataset4` are each **entirely contained** in `dataset2`
- `dataset3 ∩ dataset4 = ∅` — they are disjoint halves of `dataset2`
- there are effectively **two** distinct sources, not four

The manuscript's per-class counts (780 normal / 793 osteoporosis) are exactly
the duplicate-inflated totals. See `results/dataset_statistics.json`.

### 3.2 Backbones are frozen — this is stage 1 only

This machine has **no CUDA GPU**. The ImageNet backbones are therefore used as
frozen feature extractors and only the fusion head is trained. Stage-2
fine-tuning is implemented but **was not run**.

Because the backbones are frozen and augmentation is deterministic, each
backbone's output for a given (image, variant) pair is constant, so it is
computed once and cached. That is a *caching optimisation, mathematically
identical* to running the frozen backbone every epoch — not an approximation.

---

## 4. Leakage prevention

1. **Exact duplicates removed.** Detected by SHA-256 of the decoded pixel
   buffer, so re-encoded copies are caught too.
2. **Near-duplicates grouped, not deleted.** dHash generates candidates;
   each is *verified* by normalised pixel correlation ≥ 0.99. Threshold chosen
   from a false-positive sweep against patient-id ground truth — see
   `docs/near_duplicate_calibration.md`.
3. **Grouped splitting.** `StratifiedGroupKFold` over `group_id` (real patient
   id where available, else the verified near-duplicate cluster).
4. **Assertions.** Each fold is checked for shared `group_id`, pixel hash and
   file path across train/validation/test. The pipeline *raises* on violation.
5. **Augmentation applied to training data only.**

> **Honest limitation.** Only ~11% of binary-task images carry a real patient
> identifier (dataset1 only). The manuscript's claim that "patient-wise
> separation was maintained" is **not supportable** for the rest of the data.

---

## 5. Layout

```
data/raw/                     extracted source datasets (originals untouched)
data/metadata/
    dataset_manifest.csv      every image: hashes, class, patient id, group
    label_conflicts.csv       contradictory labels between sources
    near_duplicate_pairs.csv  verified near-duplicate pairs
splits/fold_{1..5}/           train.csv / validation.csv / test.csv
cache/                        frozen-backbone feature bank (.npy)
models/{A1..A5}/fold_{n}/     checkpoint, history, predictions, results.json
results/
    dataset_statistics.json   counts, duplicates, overlap matrix
    cross_validation/         fold_metrics, cv_summary, confidence_intervals
    statistical_tests/        paired t-test + Wilcoxon, Holm-corrected
    ablation/                 ablation_results, ablation_summary
    computational_complexity.csv
    experiment_status.csv     per-run status incl. failures
    status.json               resume state
    final_report.md           the full write-up
figures/paper/                fig_22 / 23 / 24 / 25 / 26  (PNG + PDF, 300 DPI)
tables/                       table_10..14 (CSV + XLSX)
docs/                         calibration notes
```

---

## 6. Reproducibility

- Seeds fixed: global `42`, folds `20250820`, augmentation `7`
- Folds generated once and committed to `splits/`; every model reads the same files
- Full environment captured in `environment.txt` and `results/environment.json`
- Dependencies pinned in `requirements.txt`

---

## 7. Notes and caveats

- **OneDrive.** This project lives in a synced folder. Pausing OneDrive sync
  during training avoids file-lock errors on checkpoints.
- **Disk space.** Installing TensorFlow leaves a multi-GB pip download cache
  (~5 GB was observed here, on a drive with only ~7 GB free). Run
  `pip cache purge` after installing, or `pip install --no-cache-dir -r
  requirements.txt`. The pipeline needs roughly 3 GB for the extracted data,
  feature bank, ImageNet weights and outputs.
- **Statistical power.** With 5 folds, the Wilcoxon signed-rank test cannot
  produce a two-sided p below 0.0625, so it can never reach p < 0.05 here. This
  is a property of the sample size and is reported explicitly.
- **Fold independence.** CV folds share training data, so paired tests across
  folds are descriptive, not confirmatory.
- **Grad-CAM** shows model attribution. It is not clinical validation.
- **LUMOS multimodal.** Not present locally, and it is a *lumbar spine* dataset
  while this study is *knee*. A feasibility report is produced instead of
  numbers: `results/multimodal/lumos_feasibility.md`. The manuscript's 98.1%
  has **no experimental basis**.

## 8. Provenance labelling

| Label | Meaning |
|---|---|
| **calculated** | derived exactly from the model graph (parameters, FLOPs) |
| **measured** | timed/observed on this machine (latency, size, memory) |
| **estimated** | *not used* — no reported value here is an estimate |
