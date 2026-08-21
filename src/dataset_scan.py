"""Dataset discovery, integrity checking, duplicate detection and manifest build.

The four source datasets are discovered by walking ``data/raw`` rather than by
hard-coding folder names, because the published archives differ in layout. Every
image is opened once so that corrupt files are detected rather than crashing a
training run 40 minutes in.

Duplicate signals
-----------------
* ``file_hash``   - SHA-256 of the raw file bytes (exact byte duplicates)
* ``pixel_hash``  - SHA-256 of the decoded RGB pixel buffer at native size
                    (catches identical images re-encoded to another format)
* ``dhash``       - 64-bit difference hash, used only as a cheap *candidate*
                    generator for near-duplicates.

Near-duplicate verification
---------------------------
A 64-bit dHash alone is far too permissive on knee radiographs: images that
merely share a global layout (dark background, centred bright bone) collide.
Calibration on this corpus (``scripts/calibrate_neardup.py``) showed dHash
candidate pairs with Hamming distance 0 whose actual pixel correlation was only
0.88, and 25-44% of candidate pairs below correlation 0.98 were cross-class.

Every dHash candidate pair is therefore *verified* by normalised pixel
correlation on a 64x64 grayscale thumbnail, and merged only above
``NEAR_DUP_CORR_THRESHOLD``. The calibration showed a clean separation there:
below 0.98 the cross-class rate is 25-44% (i.e. merely similar images), while
above 0.99 it collapses to ~2% (i.e. genuine re-encoded copies).

Label-conflict resolution
-------------------------
Some verified near-duplicate clusters carry contradictory labels across source
datasets. dataset1 ships DXA T-scores whose labels are 100% consistent with the
WHO/ISCD criterion, so where a cluster contains a T-score-backed image, that
label wins. Clusters with conflicting labels and no T-score evidence are
excluded from the binary task, because their ground truth is undeterminable.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from . import config
from .utils import banner, get_logger, write_json

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None

log = get_logger("dataset")

_XL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Near-duplicate verification, calibrated in scripts/calibrate_neardup.py
NEAR_DUP_THUMB = 64
NEAR_DUP_DHASH_HAMMING = 6      # candidate generation only
NEAR_DUP_CORR_THRESHOLD = 0.99  # verification; evidence-based cut

# WHO / ISCD T-score thresholds
T_SCORE_OSTEOPOROSIS = -2.5
T_SCORE_NORMAL = -1.0


# --------------------------------------------------------------------------
# Class-name normalisation
# --------------------------------------------------------------------------
def normalise_class(raw_name: str) -> str | None:
    key = re.sub(r"[^a-z]", "", raw_name.lower())
    return config.CLASS_ALIASES.get(key)


def infer_class_from_path(path: Path, dataset_root: Path) -> str | None:
    """Walk the path components from the dataset root looking for a class name.

    The deepest matching component wins, which handles both
    ``dataset4/normal/normal/x.jpg`` and ``dataset1/Osteoporosis Knee X-ray/normal/x.jpg``.
    """
    rel = path.relative_to(dataset_root)
    found = None
    for part in rel.parts[:-1]:
        cls = normalise_class(part)
        if cls is not None:
            found = cls
    return found


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """64-bit difference hash of a grayscale thumbnail."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def normalised_thumb(image: Image.Image, size: int = NEAR_DUP_THUMB) -> np.ndarray:
    """Zero-mean unit-variance grayscale thumbnail used for duplicate verification."""
    arr = np.asarray(
        image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    arr -= arr.mean()
    sd = arr.std()
    if sd > 1e-6:
        arr /= sd
    return arr.ravel()


# --------------------------------------------------------------------------
# Patient metadata (dataset1 only)
# --------------------------------------------------------------------------
def _xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_XL_NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_XL_NS}t")))
        sheet = next(n for n in z.namelist()
                     if n.startswith("xl/worksheets/") and n.endswith(".xml"))
        root = ET.fromstring(z.read(sheet))

    def col_idx(ref: str) -> int:
        letters = "".join(c for c in ref if c.isalpha())
        n = 0
        for c in letters:
            n = n * 26 + (ord(c.upper()) - 64)
        return n - 1

    rows: list[list[str]] = []
    for row in root.iter(f"{_XL_NS}row"):
        cells: dict[int, str] = {}
        for c in row.findall(f"{_XL_NS}c"):
            v = c.find(f"{_XL_NS}v")
            isel = c.find(f"{_XL_NS}is")
            if c.get("t") == "s" and v is not None:
                val = shared[int(v.text)]
            elif c.get("t") == "inlineStr" and isel is not None:
                val = "".join(t.text or "" for t in isel.iter(f"{_XL_NS}t"))
            elif v is not None:
                val = v.text or ""
            else:
                val = ""
            cells[col_idx(c.get("r", "A1"))] = val.strip()
        if cells:
            width = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(width)])
    return rows


def load_patient_metadata() -> pd.DataFrame:
    """Load dataset1's patient spreadsheet, if present.

    Returns an empty frame when unavailable, so callers can degrade gracefully.
    """
    matches = list(config.RAW_DIR.rglob("*.xlsx"))
    if not matches:
        log.warning("No patient metadata spreadsheet found.")
        return pd.DataFrame()

    path = matches[0]
    rows = _xlsx_rows(path)
    if not rows:
        return pd.DataFrame()

    header = [h.strip() for h in rows[0]]
    body = [r + [""] * (len(header) - len(r)) for r in rows[1:]]
    body = [r[: len(header)] for r in body]
    df = pd.DataFrame(body, columns=header)
    df = df[df.iloc[:, 1].astype(str).str.strip() != ""]

    rename = {}
    for c in df.columns:
        lc = c.lower().strip().rstrip(":")
        if lc in ("patient id", "patientid"):
            rename[c] = "patient_id"
        elif lc == "age":
            rename[c] = "age"
        elif lc == "gender":
            rename[c] = "gender"
        elif lc.startswith("t-score"):
            rename[c] = "t_score"
        elif lc.startswith("z-score"):
            rename[c] = "z_score"
        elif lc == "diagnosis":
            rename[c] = "diagnosis"
        elif lc == "bmi":
            rename[c] = "bmi"
        elif lc == "site":
            rename[c] = "site"
    df = df.rename(columns=rename)

    keep = [c for c in ("patient_id", "age", "gender", "t_score", "z_score",
                        "bmi", "site", "diagnosis") if c in df.columns]
    df = df[keep].copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.upper()
    for num in ("age", "t_score", "z_score", "bmi"):
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.lower()

    log.info("Loaded patient metadata: %d rows from %s", len(df), path.name)
    return df


_PATIENT_STEM_RE = re.compile(r"^(N|OP|OS)\s*_?-?\s*(\d+)", re.IGNORECASE)


def patient_id_from_stem(stem: str) -> str | None:
    """dataset1 filenames (``N1.JPEG``, ``OS12.jpg``) encode the patient id."""
    m = _PATIENT_STEM_RE.match(stem.strip())
    if not m:
        return None
    return f"{m.group(1).upper()}{int(m.group(2))}"


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------
def discover_dataset_roots() -> list[Path]:
    roots = sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir())
    if not roots:
        raise FileNotFoundError(
            f"No dataset folders found under {config.RAW_DIR}. "
            "Extract the source archives there first."
        )
    return roots


def scan_images() -> tuple[pd.DataFrame, list[dict], np.ndarray]:
    """Walk every dataset root, open every image, and record its properties.

    Also returns the stacked normalised thumbnails (row-aligned with the frame)
    so near-duplicate verification never has to re-read the images.
    """
    records: list[dict] = []
    failures: list[dict] = []
    thumbs: list[np.ndarray] = []

    for root in discover_dataset_roots():
        source = root.name
        files = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS]
        log.info("Scanning %-10s : %d candidate image files", source, len(files))

        for path in sorted(files):
            cls = infer_class_from_path(path, root)
            if cls is None:
                failures.append({
                    "filepath": str(path), "dataset_source": source,
                    "issue": "unmapped_class_folder",
                    "detail": str(path.relative_to(root).parent),
                })
                continue
            try:
                with Image.open(path) as im:
                    im.verify()                       # structural check
                with Image.open(path) as im:
                    im.load()                         # full decode
                    width, height = im.size
                    mode = im.mode
                    fmt = (im.format or path.suffix.lstrip(".")).upper()
                    rgb = im.convert("RGB")
                    pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    dh = dhash(rgb)
                    thumb = normalised_thumb(rgb)
            except Exception as exc:  # noqa: BLE001
                failures.append({
                    "filepath": str(path), "dataset_source": source,
                    "issue": "unreadable_image", "detail": f"{type(exc).__name__}: {exc}",
                })
                continue

            stem = path.stem
            thumbs.append(thumb)
            records.append({
                "filepath": str(path),
                "relpath": str(path.relative_to(config.PROJECT_ROOT)),
                "filename": path.name,
                "stem": stem,
                "dataset_source": source,
                "class": cls,
                "image_width": width,
                "image_height": height,
                "image_channels": len(mode),
                "image_mode": mode,
                "file_format": fmt,
                "file_size_bytes": path.stat().st_size,
                "file_hash": sha256_file(path),
                "pixel_hash": pixel_hash,
                "dhash": dh,
                "patient_id_raw": patient_id_from_stem(stem) if source == "dataset1" else None,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Scan produced no usable images.")
    log.info("Scanned %d readable images (%d failures)", len(df), len(failures))
    return df, failures, np.stack(thumbs).astype(np.float32)


# --------------------------------------------------------------------------
# Duplicate analysis
# --------------------------------------------------------------------------
def analyse_duplicates(df: pd.DataFrame, thumbs: np.ndarray,
                       corr_threshold: float = NEAR_DUP_CORR_THRESHOLD
                       ) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Flag exact and verified-near duplicates; assign leakage-safe ``group_id``s."""
    thumb_by_relpath = {r: thumbs[i] for i, r in enumerate(df["relpath"])}
    df = df.copy()

    # -- exact duplicates by decoded pixel content ------------------------
    df = df.sort_values(["dataset_source", "filename"]).reset_index(drop=True)
    df["exact_dup_group"] = df.groupby("pixel_hash").ngroup()
    first_idx = df.groupby("pixel_hash").head(1).index
    df["is_duplicate"] = True
    df.loc[first_idx, "is_duplicate"] = False
    df["duplicate_of"] = ""
    rep_by_hash = df.loc[first_idx].set_index("pixel_hash")["relpath"].to_dict()
    dup_mask = df["is_duplicate"]
    df.loc[dup_mask, "duplicate_of"] = df.loc[dup_mask, "pixel_hash"].map(rep_by_hash)

    n_byte_dup = int(len(df) - df["file_hash"].nunique())
    n_pixel_dup = int(dup_mask.sum())

    # cross-dataset overlap matrix on unique pixel content
    sources = sorted(df["dataset_source"].unique())
    overlap: dict[str, dict[str, int]] = {}
    hashes_by_source = {
        s: set(df.loc[df["dataset_source"] == s, "pixel_hash"]) for s in sources
    }
    for a in sources:
        overlap[a] = {b: len(hashes_by_source[a] & hashes_by_source[b]) for b in sources}

    # -- duplicate filenames (independent signal) -------------------------
    fname_counts = df["filename"].str.lower().value_counts()
    dup_filenames = int((fname_counts > 1).sum())

    # -- near duplicates among the *representatives* ----------------------
    reps = df.loc[~df["is_duplicate"]].reset_index()
    parent = {int(i): int(i) for i in reps["index"]}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Stage 1: cheap dHash candidate generation, bucketed to avoid an O(n^2) sweep.
    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for _, row in reps.iterrows():
        dh = row["dhash"]
        for chunk in (dh[:4], dh[4:8], dh[8:12]):
            buckets[chunk].append((int(row["index"]), dh))

    candidates: set[tuple[int, int]] = set()
    for items in buckets.values():
        if len(items) < 2 or len(items) > 400:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ia, ha = items[i]
                ib, hb = items[j]
                if ia == ib:
                    continue
                if hamming_hex(ha, hb) <= NEAR_DUP_DHASH_HAMMING:
                    candidates.add((min(ia, ib), max(ia, ib)))

    # Stage 2: verify each candidate by normalised pixel correlation.
    relpath_of = df["relpath"].to_dict()
    n_dim = float(thumbs.shape[1])
    n_near = 0
    verified_pairs: list[dict] = []
    for ia, ib in sorted(candidates):
        ta = thumb_by_relpath[relpath_of[ia]]
        tb = thumb_by_relpath[relpath_of[ib]]
        corr = float(np.dot(ta, tb) / n_dim)
        if corr >= corr_threshold:
            union(ia, ib)
            n_near += 1
            verified_pairs.append({
                "a_relpath": relpath_of[ia], "b_relpath": relpath_of[ib],
                "a_class": df.loc[ia, "class"], "b_class": df.loc[ib, "class"],
                "a_source": df.loc[ia, "dataset_source"],
                "b_source": df.loc[ib, "dataset_source"],
                "correlation": round(corr, 6),
                "dhash_hamming": hamming_hex(df.loc[ia, "dhash"], df.loc[ib, "dhash"]),
                "label_conflict": df.loc[ia, "class"] != df.loc[ib, "class"],
            })

    df["near_dup_group"] = -1
    for idx in reps["index"]:
        df.loc[int(idx), "near_dup_group"] = find(int(idx))

    # propagate the representative's near-dup group to its exact duplicates
    rep_group = df.loc[~df["is_duplicate"]].set_index("pixel_hash")["near_dup_group"].to_dict()
    df.loc[dup_mask, "near_dup_group"] = df.loc[dup_mask, "pixel_hash"].map(rep_group)

    summary = {
        "total_files_scanned": int(len(df)),
        "unique_by_file_bytes": int(df["file_hash"].nunique()),
        "unique_by_pixel_content": int(df["pixel_hash"].nunique()),
        "exact_byte_duplicates": n_byte_dup,
        "exact_pixel_duplicates_removed": n_pixel_dup,
        "duplicate_filenames": dup_filenames,
        "near_duplicate_candidate_pairs": int(len(candidates)),
        "near_duplicate_pairs_verified": int(n_near),
        "near_duplicate_dhash_hamming_max": NEAR_DUP_DHASH_HAMMING,
        "near_duplicate_correlation_threshold": corr_threshold,
        "near_duplicate_verification": (
            "dHash candidates confirmed by normalised pixel correlation on "
            f"{NEAR_DUP_THUMB}x{NEAR_DUP_THUMB} grayscale thumbnails"
        ),
        "label_conflict_pairs": int(sum(p["label_conflict"] for p in verified_pairs)),
        "cross_dataset_pixel_overlap": overlap,
    }
    return df, summary, verified_pairs


def t_score_to_class(t: float) -> str:
    """WHO / ISCD densitometric criterion."""
    if t <= T_SCORE_OSTEOPOROSIS:
        return config.CLASS_OSTEOPOROSIS
    if t < T_SCORE_NORMAL:
        return config.CLASS_OSTEOPENIA
    return config.CLASS_NORMAL


def resolve_label_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Resolve contradictory labels inside verified near-duplicate clusters.

    Rule, applied only to clusters that actually disagree:

    1. If any member carries a DXA T-score (dataset1), the T-score-derived
       class is authoritative and is propagated to the whole cluster.
    2. Otherwise ground truth is undeterminable, and the cluster is excluded
       from the binary task. Nothing is deleted from the manifest.
    """
    df = df.copy()
    df["label_conflict"] = False
    df["label_resolved"] = False
    df["label_resolution"] = ""
    df["excluded_reason"] = ""
    df["class_original"] = df["class"]

    reports: list[dict] = []
    for gid, sub in df.groupby("near_dup_group"):
        classes = set(sub["class"])
        if len(classes) <= 1:
            continue

        df.loc[sub.index, "label_conflict"] = True
        with_t = sub[sub["t_score"].notna()] if "t_score" in sub.columns else sub.iloc[0:0]

        entry = {
            "near_dup_group": int(gid),
            "n_images": int(len(sub)),
            "classes_present": sorted(classes),
            "files": sub["relpath"].tolist(),
            "sources": sorted(set(sub["dataset_source"])),
        }

        if len(with_t) > 0:
            t_classes = {t_score_to_class(float(t)) for t in with_t["t_score"]}
            if len(t_classes) == 1:
                resolved = t_classes.pop()
                df.loc[sub.index, "class"] = resolved
                df.loc[sub.index, "label_resolved"] = True
                df.loc[sub.index, "label_resolution"] = (
                    f"resolved to '{resolved}' from DXA T-score "
                    f"({', '.join(f'{float(t):.2f}' for t in with_t['t_score'])})"
                )
                entry.update({
                    "resolution": "t_score",
                    "resolved_class": resolved,
                    "t_scores": [round(float(t), 2) for t in with_t["t_score"]],
                    "patient_ids": [str(p) for p in with_t["patient_id"]],
                })
                reports.append(entry)
                continue

        df.loc[sub.index, "excluded_reason"] = "unresolvable_label_conflict"
        entry.update({"resolution": "excluded", "resolved_class": None})
        reports.append(entry)

    return df, reports


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the grouping key used by grouped cross-validation.

    Priority: real patient id > near-duplicate cluster > the image itself.
    """
    df = df.copy()
    groups: list[str] = []
    kinds: list[str] = []
    for _, r in df.iterrows():
        pid = r.get("patient_id")
        if isinstance(pid, str) and pid and pid.lower() != "nan":
            groups.append(f"patient:{pid}")
            kinds.append("patient_id")
        else:
            groups.append(f"imgcluster:{int(r['near_dup_group'])}")
            kinds.append("near_duplicate_cluster")
    df["group_id"] = groups
    df["group_kind"] = kinds
    return df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_manifest() -> dict:
    banner("STAGE: DATASET SCAN & MANIFEST")
    config.ensure_dirs()

    df, failures, thumbs = scan_images()
    df, dup_summary, verified_pairs = analyse_duplicates(df, thumbs)

    # -- attach patient metadata -----------------------------------------
    meta = load_patient_metadata()
    if not meta.empty:
        df["patient_id_raw"] = df["patient_id_raw"].astype("object")
        df = df.merge(
            meta.drop_duplicates("patient_id"),
            how="left", left_on="patient_id_raw", right_on="patient_id",
            suffixes=("", "_meta"),
        )
        matched = int(df["patient_id"].notna().sum())
        log.info("Patient metadata matched to %d / %d images", matched, len(df))
    else:
        df["patient_id"] = None
        df["t_score"] = np.nan
        matched = 0

    # -- resolve contradictory labels inside verified clusters ------------
    df, conflict_reports = resolve_label_conflicts(df)
    n_resolved = int(df["label_resolved"].sum())
    n_excluded = int((df["excluded_reason"] == "unresolvable_label_conflict").sum())
    if conflict_reports:
        log.warning("Label conflicts in %d verified near-duplicate clusters "
                    "(%d images relabelled from T-score, %d excluded)",
                    len(conflict_reports), n_resolved, n_excluded)

    df = assign_groups(df)

    # -- binary-task membership ------------------------------------------
    df["label_binary"] = df["class"].map(config.BINARY_LABELS)
    df["in_binary_task"] = df["class"].isin(config.BINARY_CLASSES)
    df["usable_binary"] = (
        df["in_binary_task"]
        & (~df["is_duplicate"])
        & (df["excluded_reason"] == "")
    )

    cols = [
        "relpath", "filepath", "filename", "dataset_source", "class",
        "class_original", "label_binary", "patient_id", "age", "gender",
        "t_score", "z_score", "bmi", "site", "diagnosis", "image_width",
        "image_height", "image_channels", "image_mode", "file_format",
        "file_size_bytes", "file_hash", "pixel_hash", "dhash", "is_duplicate",
        "duplicate_of", "near_dup_group", "group_id", "group_kind",
        "label_conflict", "label_resolved", "label_resolution",
        "excluded_reason", "in_binary_task", "usable_binary",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    manifest = df[cols].copy()

    manifest_path = config.METADATA_DIR / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    log.info("Manifest written: %s", manifest_path)

    if failures:
        fp = config.METADATA_DIR / "scan_failures.csv"
        pd.DataFrame(failures).to_csv(fp, index=False, encoding="utf-8")
        log.warning("%d files could not be used; see %s", len(failures), fp)

    dup_path = config.METADATA_DIR / "duplicate_report.csv"
    manifest.loc[manifest["is_duplicate"], ["relpath", "dataset_source", "class",
                                            "duplicate_of", "pixel_hash"]].to_csv(
        dup_path, index=False, encoding="utf-8")

    if verified_pairs:
        pd.DataFrame(verified_pairs).to_csv(
            config.METADATA_DIR / "near_duplicate_pairs.csv",
            index=False, encoding="utf-8")
    if conflict_reports:
        write_json(config.METADATA_DIR / "label_conflicts.json", conflict_reports)
        rows = []
        for c in conflict_reports:
            for f in c["files"]:
                rows.append({
                    "near_dup_group": c["near_dup_group"],
                    "relpath": f,
                    "classes_present": "/".join(c["classes_present"]),
                    "resolution": c["resolution"],
                    "resolved_class": c.get("resolved_class"),
                    "t_scores": ";".join(map(str, c.get("t_scores", []))),
                })
        pd.DataFrame(rows).to_csv(
            config.METADATA_DIR / "label_conflicts.csv", index=False, encoding="utf-8")

    stats = _compute_statistics(manifest, dup_summary, failures, matched)
    stats["label_conflicts"] = {
        "clusters_with_conflicting_labels": len(conflict_reports),
        "images_relabelled_from_t_score": n_resolved,
        "images_excluded_as_unresolvable": n_excluded,
        "rule": (
            "Within a verified near-duplicate cluster, a DXA T-score (dataset1) "
            "is authoritative; clusters with conflicting labels and no T-score "
            "evidence are excluded from the binary task."
        ),
        "detail": conflict_reports,
    }
    write_json(config.RESULTS_DIR / "dataset_statistics.json", stats)
    log.info("Statistics written: %s", config.RESULTS_DIR / "dataset_statistics.json")
    _log_summary(stats)
    return stats


def _compute_statistics(manifest: pd.DataFrame, dup_summary: dict,
                        failures: list[dict], matched: int) -> dict:
    unique = manifest.loc[~manifest["is_duplicate"]]
    binary = manifest.loc[manifest["usable_binary"]]

    def counts(frame: pd.DataFrame, col: str = "class") -> dict:
        return {str(k): int(v) for k, v in frame[col].value_counts().sort_index().items()}

    per_source_raw = (
        manifest.groupby(["dataset_source", "class"]).size()
        .unstack(fill_value=0).astype(int).to_dict(orient="index")
    )
    per_source_unique = (
        unique.groupby(["dataset_source", "class"]).size()
        .unstack(fill_value=0).astype(int).to_dict(orient="index")
    )

    dims = manifest[["image_width", "image_height"]]
    resolutions = (
        manifest.groupby(["image_width", "image_height"]).size()
        .sort_values(ascending=False).head(15)
    )

    n_groups_binary = int(binary["group_id"].nunique())
    patient_backed = binary["group_kind"].eq("patient_id").sum()

    return {
        "generated_by": "src/dataset_scan.py",
        "raw_scan": {
            "total_files_scanned": int(len(manifest)),
            "readable_images": int(len(manifest)),
            "failed_files": len(failures),
            "class_counts_including_duplicates": counts(manifest),
            "per_dataset_class_counts_including_duplicates": per_source_raw,
        },
        "deduplicated": {
            "unique_images": int(len(unique)),
            "class_counts": counts(unique),
            "per_dataset_class_counts": per_source_unique,
        },
        "duplicates": dup_summary,
        "binary_task": {
            "definition": "normal (label 0) vs osteoporosis (label 1); osteopenia excluded",
            "positive_class": config.POSITIVE_CLASS,
            "usable_images": int(len(binary)),
            "class_counts": counts(binary),
            "per_dataset_class_counts": (
                binary.groupby(["dataset_source", "class"]).size()
                .unstack(fill_value=0).astype(int).to_dict(orient="index")
            ),
            "n_groups_for_grouped_cv": n_groups_binary,
            "images_with_real_patient_id": int(patient_backed),
            "fraction_with_real_patient_id": round(
                float(patient_backed) / max(1, len(binary)), 4),
        },
        "osteopenia": {
            "note": "retained in the manifest but excluded from the binary ablation",
            "unique_images": int((unique["class"] == config.CLASS_OSTEOPENIA).sum()),
        },
        "patient_metadata": {
            "spreadsheet_found": matched > 0,
            "images_matched_to_patient_records": int(matched),
            "datasets_with_patient_ids": ["dataset1"] if matched else [],
            "limitation": (
                "Patient identifiers exist only for dataset1. Datasets 2-4 are "
                "public Kaggle exports with no patient-level metadata, so true "
                "patient-wise separation cannot be verified for them."
            ),
        },
        "image_properties": {
            "width_min": int(dims["image_width"].min()),
            "width_max": int(dims["image_width"].max()),
            "height_min": int(dims["image_height"].min()),
            "height_max": int(dims["image_height"].max()),
            "most_common_resolutions": {f"{w}x{h}": int(n)
                                        for (w, h), n in resolutions.items()},
            "modes": {str(k): int(v) for k, v in manifest["image_mode"].value_counts().items()},
            "formats": {str(k): int(v) for k, v in manifest["file_format"].value_counts().items()},
        },
    }


def _log_summary(stats: dict) -> None:
    raw = stats["raw_scan"]
    dedup = stats["deduplicated"]
    binary = stats["binary_task"]
    log.info("-" * 70)
    log.info("Files scanned (incl. duplicates) : %d", raw["total_files_scanned"])
    log.info("  class counts                   : %s", raw["class_counts_including_duplicates"])
    log.info("Unique images after dedup        : %d", dedup["unique_images"])
    log.info("  class counts                   : %s", dedup["class_counts"])
    log.info("Exact duplicates removed         : %d",
             stats["duplicates"]["exact_pixel_duplicates_removed"])
    log.info("Near-dup candidates / verified   : %d / %d",
             stats["duplicates"]["near_duplicate_candidate_pairs"],
             stats["duplicates"]["near_duplicate_pairs_verified"])
    lc = stats.get("label_conflicts", {})
    log.info("Label-conflict clusters          : %d (relabelled %d, excluded %d)",
             lc.get("clusters_with_conflicting_labels", 0),
             lc.get("images_relabelled_from_t_score", 0),
             lc.get("images_excluded_as_unresolvable", 0))
    log.info("Binary task usable images        : %d  %s",
             binary["usable_images"], binary["class_counts"])
    log.info("Grouping units for CV            : %d", binary["n_groups_for_grouped_cv"])
    log.info("Images with real patient id      : %d (%.1f%%)",
             binary["images_with_real_patient_id"],
             100 * binary["fraction_with_real_patient_id"])
    log.info("-" * 70)


if __name__ == "__main__":
    build_manifest()
