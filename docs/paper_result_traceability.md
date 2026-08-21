# Paper → Pipeline Traceability Table

Maps every result-bearing claim in `radiomics_pdf.pdf` to the pipeline file that
now holds its real replacement — or, where nothing was reproduced, says so
plainly rather than leaving the gap implicit.

**Scope note, read first:** the real ablation pipeline (A1–A5) only covers the
VGG19 / InceptionResNetV2 / MobileNetV2 backbone family used in the paper's
**Hybrid Model 1**. It does **not** retrain Custom CNN, ResNet-50, DenseNet-121,
XceptionNet, or Hybrid Model 2 (DenseNet + EfficientNet) — those use different
backbones entirely and were out of scope for the hybrid-architecture ablation
requested. Rows for those models are marked **not reproduced** below; treat
their paper values as still-unverified until a separate run covers them.

---

## A. Dataset & methodology claims

| Paper location | Paper's claim | Real file(s) | Measured result |
|---|---|---|---|
| Abstract, p.2 "Dataset" | 1,947 images: 780 normal / 374 osteopenia / 793 osteoporosis, four independent datasets | `data/metadata/dataset_manifest.csv`, `results/dataset_statistics.json`, `tables/table_S3_dataset_composition.csv` | **1,727 files on disk, only 841 unique images** (886 exact duplicates). Unique: 353 normal / 192 osteopenia / 296 osteoporosis. `dataset3` and `dataset4` are each entirely contained in `dataset2` — effectively two sources, not four |
| p.6 "we have done the patient-wise separation was maintained" | Patient-wise leakage prevention across all data | `splits/split_manifest.json`, `results/final_report.md` §3 | Only **73 / 634 (11.5%)** binary-task images carry a real patient id (dataset1 only). Not supportable for the rest of the corpus |
| p.6 "70:15:15 stratified split" | Single stratified train/val/test split | `splits/fold_1..5/{train,validation,test}.csv` | Replaced with **grouped, stratified 5-fold CV** (leak-checked by assertion) rather than one split |
| p.6 Image Resolution — 224×224 | Uniform resize | `src/preprocessing.py` | Confirmed identical: 224×224 bilinear resize |
| p.6 "Min-Max scaling [0,1]" applied uniformly | One shared normalization for all backbones | `src/preprocessing.py::get_preprocess_fn`, `results/final_report.md` §4 | **Not applied as described** — each backbone uses its own official Keras preprocessing (VGG19: Caffe BGR mean-subtract; InceptionResNetV2/MobileNetV2: scale to [-1,1]). A shared [0,1] rescale would put two of three backbones out of distribution |
| p.6 "system equipped with NVIDIA GPU (RTX 3060)" | GPU training | `environment.txt`, `results/environment.json` | **No CUDA GPU on the execution machine** (Intel UHD integrated only). All training/inference figures are CPU measurements |
| p.6 hyperparameter grid (batch 16/32/64, lr 1e-4/1e-3/1e-2, dropout 0.3/0.5) | 18-way grid search | `results/final_report.md` §4, every `models/{arch}/fold_{n}/results.json` | Grid search **not performed**; one fixed, documented, paper-consistent config used for every run (batch 32, lr 1e-4, dropout 0.5/0.3, Adam) so all 25 runs are comparable |
| p.14 fused feature dimension = 512+1536+1280 = 3328 | Claimed dimensionality | `cache/feature_bank_meta.json`, `results/final_report.md` §5 | **Confirmed exactly**: measured 512 / 1536 / 1280 / 3328 |
| Verified near-duplicate labels vs DXA T-scores | *(not discussed in paper)* | `data/metadata/label_conflicts.csv`, `docs/near_duplicate_calibration.md` | **New finding**: 5+ images dataset2 labels "Normal" are DXA-confirmed osteoporotic in dataset1 (T ≤ −2.5) |

---

## B. Standalone / hybrid model accuracy — Table 1 (p.18) & Table X (p.19)

| Model in paper | Paper's accuracy | Real file(s) | Measured result |
|---|---|---|---|
| Custom CNN (multiclass) | 89.1% | — | **Not reproduced** (not in ablation scope) |
| VGG19 (binary) | 94.5% | `results/ablation/ablation_results.csv` (config **A1**), `tables/table_11_ablation.csv` | **75.37 ± 4.61%** (5-fold mean ± SD, grouped CV, de-duplicated data) |
| ResNet-50 (binary) | 82.16% | — | **Not reproduced** |
| DenseNet-121 (binary) | 92.99% | — | **Not reproduced** |
| XceptionNet (multiclass) | 89.2% | — | **Not reproduced** |
| **Hybrid Model 1** (VGG19+IRv2+MNv2, binary) | **97.5%** | `results/ablation/ablation_results.csv` (config **A5**), `results/cross_validation/cv_summary.csv`, `tables/table_10_cross_validation.csv` | **80.44 ± 2.23%** — a ~17 pp gap |
| Hybrid Model 2 (DenseNet+EfficientNet, multiclass) | 94.8% | — | **Not reproduced** (different backbone family) |

New rows the paper's Table 1 does not have, produced by this pipeline:

| New configuration | Real file(s) | Measured result |
|---|---|---|
| A2: VGG19 + InceptionResNetV2 | `results/ablation/ablation_results.csv` | **80.44 ± 3.12%**, AUC 0.870 |
| A3: VGG19 + MobileNetV2 | same | **79.51 ± 4.34%**, AUC 0.844 |
| A4: InceptionResNetV2 + MobileNetV2 | same | **79.81 ± 3.91%**, AUC 0.856 |

---

## C. Per-model training curves / confusion matrices — Fig 3–19

| Paper figure(s) | Model | Real file(s) | Status |
|---|---|---|---|
| Fig 3a/3b, 4a/4b | Custom CNN accuracy/loss/CM | — | **Not reproduced** |
| Fig 5a/5b | VGG19 accuracy/loss | `figures/training_curves/training_curves_A1_fold1.png` | Real replacement (different run, same architecture role) |
| Fig 6a/6b | VGG19 scores / confusion matrix | `figures/confusion/confusion_matrices_all.png` (A1 panel), `results/cross_validation/fold_metrics.csv` | Real replacement |
| Fig 7a/7b, 8a/8b | ResNet-50 accuracy/loss/CM | — | **Not reproduced** |
| Fig 9a/9b, 10a/10b | DenseNet-121 accuracy/loss/CM | — | **Not reproduced** |
| Fig 11a/11b, 12a/12b | XceptionNet accuracy/loss/CM | — | **Not reproduced** |
| Fig 13 | Hybrid Model 1 architecture diagram | `src/models.py::build_end_to_end_model` (code, not a rendered diagram) | Architecture implemented exactly as diagrammed; no diagram image generated |
| Fig 14a/14b | Hybrid Model 1 accuracy/loss | `figures/training_curves/training_curves_A5_fold1.png` | Real replacement |
| Fig 15a/15b | Hybrid Model 1 scores / confusion matrix | `figures/confusion/confusion_matrix_A5_full_hybrid.png`, `results/cross_validation/out_of_fold_metrics.csv` | Real replacement — pooled out-of-fold CM: 266 TN / 75 FP / 49 FN / 244 TP |
| Fig 16, 17a/17b, 18a/18b | Hybrid Model 2 architecture/curves/CM | — | **Not reproduced** |
| Fig 19a/19b | Grad-CAM (AP + lateral knee) | `figures/gradcam/gradcam_A5_fold1.png` | Real replacement — per-branch (VGG19/IRv2/MNv2) Grad-CAM over 6 real test images, class-discriminative gradient (paper's version doesn't specify per-branch attribution) |

---

## D. Overall comparison figures — Fig 20a/20b, 21a/21b (p.19–20)

| Paper figure | Paper's content | Real file(s) | Status |
|---|---|---|---|
| Fig 20a | Accuracy bar chart, 7 models | `figures/paper/fig_24_real_ablation.png` (left panel) | Partial replacement — only the 5 hybrid-family configs (A1–A5), not the other 4 standalone models |
| Fig 20b | Confusion-matrix diagonal sum | `figures/confusion/confusion_matrices_all.png` | Partial replacement, A1–A5 only |
| Fig 21a | Validation accuracy line chart | `figures/paper/fig_22_real_cross_validation.png` | Different form (boxplot across real folds vs a single line across models) but conveys the same comparison, A1–A5 only |
| Fig 21b | Validation loss line chart | `figures/training_curves/training_curves_{A1..A5}_fold1.png` (per-model loss curves) | No single consolidated loss-comparison chart was generated; per-model curves exist individually |

---

## E. Five-fold cross-validation — Table 10, Fig 22–23 (p.20–21)

| Paper location | Paper's value | Real file(s) | Measured result |
|---|---|---|---|
| Table 10 — Accuracy | 97.50 ± 0.158%, CI [97.304, 97.696] | `tables/table_10_cross_validation.csv/.xlsx`, `results/cross_validation/cv_summary.csv` | **80.44 ± 2.23%**, 95% CI [77.66, 83.21]% |
| Table 10 — Precision | 98.00 ± 0.158% | same | **77.03 ± 3.96%** |
| Table 10 — Recall | 97.20 ± 0.158% | same | **82.85 ± 5.98%** |
| Table 10 — F1 | 97.60 ± 0.158% | same | **79.61 ± 1.58%** |
| *(identical SD=0.158 across 4 different metrics — statistically implausible; a signature of hand-authored/illustrative numbers, not a measured result)* | | `results/cross_validation/confidence_intervals.csv` | Real SDs differ per metric (2.23–5.98 pp), as expected for genuinely independent metrics |
| Fig 22 — fold-wise distribution | Illustrative boxplot, "five representative folds" | `figures/paper/fig_22_real_cross_validation.png`, `results/cross_validation/fold_metrics.csv` | Real per-fold values for all 5 architectures, all 7 metrics |
| Fig 23 — mean + 95% CI | Illustrative error-bar chart for Hybrid Model 1 only | `figures/paper/fig_23_real_cv_confidence_intervals.png` | Real Student-t 95% CIs for all 5 architectures, all 7 metrics |
| *(not in paper)* ROC-AUC not tabulated for any of the 7 models | — | `figures/paper/fig_25_real_roc_curves.png`, `results/cross_validation/out_of_fold_metrics.csv` | **New**: pooled out-of-fold ROC curves + bootstrap 95% CI on AUC for A1–A5 |

---

## F. Ablation study — Table 11, Fig 24 (p.21)

| Paper location | Paper's content | Real file(s) | Status |
|---|---|---|---|
| Table 11 "ablation-style analysis" | Compares 7 unrelated networks, mixing binary and multiclass tasks | `results/ablation/ablation_results.csv`, `tables/table_11_ablation.csv` | **This is not a valid ablation and is not what the real table replaces it with.** Real ablation is a genuine component study: same fusion architecture, branches removed one at a time (A1–A4), same folds, same hyperparameters, all binary |
| Fig 24 — "3.00 pp improvement over VGG19, 2.70 pp over Hybrid Model 2" | Full hybrid strictly best | `figures/paper/fig_24_real_ablation.png`, `results/ablation/ablation_summary.csv` | **A5 vs A1: +5.06 pp** (branch removal matters). **A5 vs A2: −0.01 pp** — adding the MobileNetV2 branch back to A2 makes **no measurable difference**; A5 does not beat its own two-branch subset |
| *(not in paper)* | No significance testing anywhere in the paper | `results/statistical_tests/statistical_comparison.csv`, `tables/table_13_statistical_tests.csv` (**this is the pipeline's own Table 13 — unrelated to the paper's Table 13, which is the multimodal table in section G below**) | **New**: paired t-test + Wilcoxon, Holm-corrected. **0 of 4** ablation comparisons reach significance at α=0.05 (n=5 folds limits power) |

---

## G. Computational complexity — Table 12, Fig 26 (p.21–22)

| Model | Paper's params / GFLOPs / size / inference | Real file(s) | Measured (params calc. / GFLOPs calc. / size meas. / inference meas., **CPU**) |
|---|---|---|---|
| Custom CNN | 3.2M / 1.4 / 12MB / 12ms | — | **Not reproduced** |
| ResNet-50 | 25.6M / 4.1 / 98MB / 28ms | — | **Not reproduced** |
| VGG19 | 20.1M / 19.6 / 81MB / 36ms | `results/computational_complexity.csv` (A1), `tables/table_12_complexity.csv` | **20.19M** / 39.04 GFLOPs (**19.52 GMACs — matches paper's "19.6" almost exactly, i.e. the paper's number is MACs, not true FLOPs**) / **77.1MB** / **285.2ms** (CPU; ~8× paper's GPU figure) |
| DenseNet-121 | 8.1M / 2.9 / 33MB / 18ms | — | **Not reproduced** |
| XceptionNet | 22.9M / 8.4 / 88MB / 31ms | — | **Not reproduced** |
| Hybrid Model 2 | 28.4M / 7.2 / 110MB / 39ms | — | **Not reproduced** |
| **Hybrid Model 1** | **77.4M / 30.8 / 300MB / 54ms** | `results/computational_complexity.csv` (A5), `figures/paper/fig_26_real_complexity.png` | **77.50M** (matches closely) / 52.62 GFLOPs (26.31 GMACs) / **298.9MB** (matches closely) / **1411.9ms** (CPU; ~26× paper's GPU figure) |
| — | GPU memory not addressed | `results/computational_complexity.csv` (`memory_provenance` column) | Explicitly recorded as **"no GPU memory to report"** (no CUDA GPU present) rather than omitted |

Params and model size are close to the paper's stated values for the two overlapping rows (VGG19, Hybrid Model 1) — those numbers appear to have been derived reasonably. GFLOPs and inference time diverge substantially: GFLOPs because the paper's number is MACs relabeled as FLOPs, and inference time because it was never measured on the stated RTX 3060 in this reproduction (this machine has no GPU).

---

## H. Multimodal LUMOS extension — Table 13, Fig 27–30 (p.22–25)

| Paper location | Paper's value | Real file(s) | Status |
|---|---|---|---|
| §4.11.1–4.11.8, Table 13, Fig 27–30 | X-ray-only 97.5%, CT-only 93.4%, X-ray+CT 98.1%; AUC 0.980/0.945/0.987; full confusion matrices, sensitivity/specificity | `results/multimodal/lumos_feasibility.json`, `results/multimodal/lumos_feasibility.md` | **No experimental basis of any kind.** LUMOS is not present on this machine, and is a *lumbar-spine* dataset while this study is *knee* — an anatomical mismatch the paper's Table 13 doesn't disclose (it reuses the knee X-ray accuracy as the "X-ray-only" reference for a lumbar comparison). A feasibility report was produced instead of any number. The paper's own text already flags these three values as "demonstrative"/"should not be interpreted as experimentally validated" — the pipeline treats that caveat as binding and produces nothing to contradict it |

---

## Quick index: every pipeline artifact referenced above

| File | Contents |
|---|---|
| `data/metadata/dataset_manifest.csv` | Every image: hash, class, patient id, duplicate/near-duplicate group |
| `data/metadata/label_conflicts.csv` | Cross-source label disagreements, resolved by DXA T-score |
| `results/dataset_statistics.json` | Full dataset counts, duplicate/overlap summary |
| `splits/fold_{1..5}/*.csv`, `splits/split_manifest.json` | The 5 fixed, leak-checked folds every model shares |
| `results/cross_validation/fold_metrics.csv` | Per-fold, per-architecture metrics (the primary data table) |
| `results/cross_validation/cv_summary.csv` | Mean ± SD across folds |
| `results/cross_validation/confidence_intervals.csv` | 95% CIs per metric |
| `results/cross_validation/out_of_fold_metrics.csv` | Pooled OOF metrics + bootstrap AUC CI |
| `results/ablation/ablation_results.csv`, `ablation_summary.csv` | Component ablation + Δ vs full model |
| `results/statistical_tests/statistical_comparison.csv` | Paired t-test / Wilcoxon, Holm-corrected |
| `results/computational_complexity.csv` | Params/GFLOPs (calculated), size/latency/memory (measured) |
| `results/multimodal/lumos_feasibility.{json,md}` | Why the multimodal extension wasn't run, and what would be needed |
| `results/final_report.md` | Narrative version of everything in this table, §16 has its own claim-by-claim verdict |
| `figures/paper/fig_22..26_real_*.{png,pdf}` | Direct replacements for the paper's Fig. 22/23/24/26 |
| `figures/paper/fig_25_real_roc_curves.{png,pdf}` | New — ROC/AUC evidence the paper's main results section lacked |
| `figures/confusion/`, `figures/training_curves/`, `figures/gradcam/` | Supporting per-model figures |
| `tables/table_10..14_*.{csv,xlsx}` | Manuscript-ready tables |
