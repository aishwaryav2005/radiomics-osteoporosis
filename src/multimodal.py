"""LUMOS multimodal (X-ray + CT) extension: discovery, feasibility, and pipeline.

The manuscript's multimodal section currently reports X-ray-only 97.5%,
CT-only 93.4% and X-ray+CT 98.1%. Those are demonstrative placeholders. This
module never reproduces them.

Behaviour:

* If a LUMOS installation is found locally, it is inspected and characterised:
  modality counts, patient identifiers, X-ray/CT pairing, label availability.
  Whether the experiment can actually be run is then decided from what is
  found, not assumed.
* If LUMOS is absent - or present but insufficient - a feasibility report is
  written setting out exactly what exists, what is required, and what blocks
  the experiment. No performance number is produced in that case.

Point ``LUMOS_DIR`` (env var or ``--lumos-dir``) at the extracted dataset to
enable discovery.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import config
from .utils import banner, get_logger, write_json

log = get_logger("multimodal")

XRAY_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CT_EXTS = {".nii", ".gz", ".dcm", ".mha", ".mhd", ".nrrd", ".npy", ".npz"}

# What the manuscript states about LUMOS (documentation, not measurement).
LUMOS_PUBLISHED = {
    "source_zenodo": "https://zenodo.org/records/18173664",
    "source_project": "https://keyueshi.github.io/LUMOS/",
    "anatomy": "lumbar spine",
    "xray_images_reported": 1620,
    "ct_scans_reported": 280,
    "patients_reported": 803,
    "labels": "DXA T-score derived (normal >= -1.0, osteopenia -2.5..-1.0, osteoporosis <= -2.5)",
}


def candidate_dirs(explicit: str | None = None) -> list[Path]:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("LUMOS_DIR")
    if env:
        cands.append(Path(env))
    home = Path.home()
    cands += [
        config.DATA_DIR / "lumos",
        config.RAW_DIR / "lumos",
        config.RAW_DIR / "LUMOS",
        home / "Downloads" / "LUMOS",
        home / "Downloads" / "lumos",
        home / "Downloads",
    ]
    return [c for c in cands if c.exists() and c.is_dir()]


def discover(explicit: str | None = None) -> dict:
    """Look for a LUMOS installation and describe whatever is found."""
    for base in candidate_dirs(explicit):
        # Only treat a directory as LUMOS if it actually looks like it.
        looks_like = (
            base.name.lower() == "lumos"
            or any(p.name.lower().startswith("lumos") for p in base.iterdir() if p.is_dir())
        )
        if not looks_like:
            continue
        root = base if base.name.lower() == "lumos" else next(
            p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("lumos"))

        xrays = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in XRAY_EXTS]
        cts = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in CT_EXTS]
        meta = [p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".json", ".tsv"}]
        return {
            "found": True,
            "root": str(root),
            "n_xray_files": len(xrays),
            "n_ct_files": len(cts),
            "n_metadata_files": len(meta),
            "metadata_files": [str(p.relative_to(root)) for p in meta[:20]],
            "sample_xray": [str(p.relative_to(root)) for p in xrays[:5]],
            "sample_ct": [str(p.relative_to(root)) for p in cts[:5]],
        }
    return {"found": False, "searched": [str(c) for c in candidate_dirs(explicit)]}


def build_feasibility_report(found: dict) -> dict:
    """Assess feasibility from what was actually found on this machine."""
    from .utils import read_json

    env = read_json(config.RESULTS_DIR / "environment.json", default={}) or {}
    gpu = bool(env.get("gpu_available"))

    blockers: list[dict] = []
    if not found.get("found"):
        blockers.append({
            "blocker": "Dataset not available locally",
            "detail": ("No LUMOS installation was found on this machine. The "
                       "directories searched are listed under `searched`. The "
                       "dataset must be downloaded from Zenodo and extracted, "
                       "then LUMOS_DIR set to point at it."),
            "severity": "blocking",
            "resolvable": True,
        })
    else:
        if found["n_ct_files"] == 0:
            blockers.append({
                "blocker": "No CT volumes found in the LUMOS installation",
                "detail": f"Scanned {found['root']}; found 0 files with CT extensions.",
                "severity": "blocking", "resolvable": True,
            })
        if found["n_metadata_files"] == 0:
            blockers.append({
                "blocker": "No metadata file found",
                "detail": ("Patient identifiers, T-scores and modality pairing "
                           "cannot be established without the clinical metadata."),
                "severity": "blocking", "resolvable": True,
            })

    blockers.append({
        "blocker": "Anatomical mismatch with the main study",
        "detail": (
            "LUMOS is a LUMBAR SPINE dataset (X-ray and CT). The model validated "
            "in this project is trained on KNEE radiographs. A knee-trained "
            "feature extractor cannot be applied to lumbar images without "
            "retraining, and the manuscript's Table 13 currently places a knee "
            "X-ray accuracy (97.5%) in the same row as lumbar CT results. Those "
            "are different anatomies, different cohorts and different tasks, so "
            "the three configurations as tabulated are not comparable. A valid "
            "multimodal comparison must train and evaluate all three arms "
            "(X-ray-only, CT-only, X-ray+CT) on the SAME lumbar cohort."
        ),
        "severity": "design-level",
        "resolvable": True,
        "resolution": ("re-derive the X-ray-only baseline on LUMOS lumbar "
                       "radiographs instead of reusing the knee result"),
    })

    blockers.append({
        "blocker": "Paired-case count is the binding statistical constraint",
        "detail": (
            f"LUMOS is documented as {LUMOS_PUBLISHED['xray_images_reported']} lumbar "
            f"X-rays and {LUMOS_PUBLISHED['ct_scans_reported']} CT scans across "
            f"{LUMOS_PUBLISHED['patients_reported']} patients. The multimodal arm is "
            "limited by patients having BOTH modalities, which is at most the CT "
            "count (280) and in practice fewer. After patient-level splitting and "
            "three-class stratification this may leave too few cases per class per "
            "fold for a stable estimate. The paired count must be measured from the "
            "metadata before any multimodal training is attempted."
        ),
        "severity": "design-level", "resolvable": "unknown until measured",
    })

    if not gpu:
        blockers.append({
            "blocker": "No GPU for 3D CT processing",
            "detail": (
                "This machine has no CUDA GPU (Intel integrated graphics only) and "
                f"{env.get('total_ram_gb', 'unknown')} GB RAM. 3D CT volume "
                "processing - or 2.5D slice-ensemble processing over 280 volumes - "
                "is orders of magnitude heavier than the 2D knee pipeline. The 2D "
                "knee feature extraction alone took roughly 26 minutes on this CPU."
            ),
            "severity": "resource", "resolvable": True,
            "resolution": "run the CT arm on GPU hardware",
        })

    feasible = not any(b["severity"] == "blocking" for b in blockers)

    return {
        "generated_by": "src/multimodal.py",
        "conclusion": (
            "NOT FEASIBLE on this machine with the currently available data."
            if not feasible else
            "Data present; feasibility depends on the measured paired-case count."
        ),
        "multimodal_experiment_executed": False,
        "performance_values_produced": None,
        "integrity_statement": (
            "No multimodal performance value has been produced by this pipeline. "
            "The manuscript's X-ray-only 97.5%, CT-only 93.4% and X-ray+CT 98.1% "
            "are demonstrative placeholders and must not be presented as "
            "experimentally achieved results. In particular the 98.1% figure has "
            "no experimental basis in this project."
        ),
        "discovery": found,
        "published_characteristics": LUMOS_PUBLISHED,
        "blockers": blockers,
        "requirements_to_run": [
            "Download LUMOS from Zenodo (record 18173664) and extract it locally.",
            "Set the LUMOS_DIR environment variable to the extracted root.",
            "Parse the clinical metadata to obtain patient ids and DXA T-scores.",
            "Measure how many patients have BOTH a lumbar X-ray and a CT scan.",
            "Confirm at least ~30-50 paired cases per class survive patient-level "
            "stratified splitting; otherwise report the multimodal arm as "
            "under-powered rather than reporting an accuracy.",
            "Re-derive the X-ray-only baseline on LUMOS lumbar radiographs so all "
            "three arms share one cohort.",
            "Run the CT arm on GPU hardware.",
        ],
        "planned_architecture_if_feasible": {
            "xray_branch": ("2D CNN backbone (the same frozen-fusion design used "
                            "for knee) over lumbar radiographs -> GAP -> feature vector"),
            "ct_branch": ("3D CNN, or 2.5D slice ensemble over the lumbar vertebral "
                          "region, -> pooled feature vector"),
            "fusion": "concatenate X-ray and CT feature vectors -> dense layers -> classifier",
            "constraint": "X-ray and CT samples must belong to the same patient",
            "splitting": "patient-level stratified splitting, verified by assertion",
            "comparison_arms": ["X-ray only", "CT only", "X-ray + CT"],
            "metrics": ["accuracy", "precision", "recall", "F1", "sensitivity",
                        "specificity", "ROC-AUC", "confusion matrix"],
        },
    }


def run_multimodal(lumos_dir: str | None = None) -> dict:
    banner("STAGE: MULTIMODAL (LUMOS) FEASIBILITY")
    config.ensure_dirs()

    found = discover(lumos_dir)
    if found.get("found"):
        log.info("LUMOS found at %s (%d X-ray files, %d CT files)",
                 found["root"], found["n_xray_files"], found["n_ct_files"])
    else:
        log.warning("LUMOS dataset not found locally.")

    report = build_feasibility_report(found)
    write_json(config.MULTIMODAL_DIR / "lumos_feasibility.json", report)
    _write_markdown(report)

    log.info("Conclusion: %s", report["conclusion"])
    log.warning("No multimodal performance values were produced. "
                "The manuscript's 98.1%% remains a placeholder with no "
                "experimental basis.")
    return report


def _write_markdown(report: dict) -> None:
    lines = [
        "# LUMOS Multimodal Extension — Feasibility Report",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        "## Integrity statement",
        "",
        report["integrity_statement"],
        "",
        "## What was found on this machine",
        "",
    ]
    d = report["discovery"]
    if d.get("found"):
        lines += [
            f"- LUMOS root: `{d['root']}`",
            f"- X-ray-format files: **{d['n_xray_files']}**",
            f"- CT-format files: **{d['n_ct_files']}**",
            f"- Metadata files: **{d['n_metadata_files']}**",
        ]
        if d.get("metadata_files"):
            lines.append(f"- Metadata: {', '.join(f'`{m}`' for m in d['metadata_files'])}")
    else:
        lines += [
            "The LUMOS dataset is **not present on this machine**. Directories searched:",
            "",
        ] + [f"- `{s}`" for s in d.get("searched", [])]

    lines += ["", "## Published characteristics (from the dataset description, not measured)", ""]
    for k, v in report["published_characteristics"].items():
        lines.append(f"- **{k}**: {v}")

    lines += ["", "## Blockers", ""]
    for i, b in enumerate(report["blockers"], start=1):
        lines += [
            f"### {i}. {b['blocker']}  _({b['severity']})_",
            "",
            b["detail"],
            "",
        ]
        if b.get("resolution"):
            lines += [f"*Resolution:* {b['resolution']}", ""]

    lines += ["## What would be required to run this experiment", ""]
    lines += [f"{i}. {r}" for i, r in enumerate(report["requirements_to_run"], start=1)]

    lines += ["", "## Planned architecture, if the data supports it", ""]
    arch = report["planned_architecture_if_feasible"]
    for k, v in arch.items():
        val = ", ".join(v) if isinstance(v, list) else v
        lines.append(f"- **{k}**: {val}")

    lines += [
        "",
        "## Recommended manuscript change",
        "",
        "Section 4.11 should state explicitly that the multimodal analysis is a",
        "*planned design*, not an executed experiment, and Table 13 should either be",
        "removed or relabelled as a hypothetical illustration with the numbers",
        "withdrawn. As written, the section reads as reporting measured performance.",
        "The anatomical mismatch (knee vs lumbar) should also be stated, since the",
        "current table compares a knee X-ray result against lumbar CT results.",
        "",
    ]
    (config.MULTIMODAL_DIR / "lumos_feasibility.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_multimodal()
