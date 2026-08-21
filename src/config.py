"""Central configuration for the knee-osteoporosis radiomics pipeline.

Every path, seed and hyper-parameter used anywhere in the pipeline is defined
here so that a single file fully determines a run. Nothing in this module reads
or writes the original dataset archives in Downloads; ``data/raw`` holds an
extracted working copy.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
INTERIM_DIR = DATA_DIR / "interim"

SPLITS_DIR = PROJECT_ROOT / "splits"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
CV_DIR = RESULTS_DIR / "cross_validation"
ABLATION_DIR = RESULTS_DIR / "ablation"
STATS_DIR = RESULTS_DIR / "statistical_tests"
COMPLEXITY_DIR = RESULTS_DIR / "complexity"
LOGS_DIR = RESULTS_DIR / "logs"
FOLDS_DIR = RESULTS_DIR / "folds"
MULTIMODAL_DIR = RESULTS_DIR / "multimodal"

FIGURES_DIR = PROJECT_ROOT / "figures"
PAPER_FIG_DIR = FIGURES_DIR / "paper"
GRADCAM_DIR = FIGURES_DIR / "gradcam"
ROC_FIG_DIR = FIGURES_DIR / "roc"
CM_FIG_DIR = FIGURES_DIR / "confusion"
CURVES_FIG_DIR = FIGURES_DIR / "training_curves"
COMPLEXITY_FIG_DIR = FIGURES_DIR / "complexity"

TABLES_DIR = PROJECT_ROOT / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

ALL_DIRS = [
    DATA_DIR, RAW_DIR, METADATA_DIR, INTERIM_DIR, SPLITS_DIR, CACHE_DIR,
    MODELS_DIR, RESULTS_DIR, CV_DIR, ABLATION_DIR, STATS_DIR, COMPLEXITY_DIR,
    LOGS_DIR, FOLDS_DIR, MULTIMODAL_DIR, FIGURES_DIR, PAPER_FIG_DIR,
    GRADCAM_DIR, ROC_FIG_DIR, CM_FIG_DIR, CURVES_FIG_DIR, COMPLEXITY_FIG_DIR,
    TABLES_DIR, DOCS_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
GLOBAL_SEED = 42
FOLD_SEED = 20250820          # seed used only for fold construction
AUGMENTATION_SEED = 7         # seed used only for offline augmentation

# --------------------------------------------------------------------------
# Dataset definition
# --------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Canonical class vocabulary. Folder names are normalised onto these.
CLASS_NORMAL = "normal"
CLASS_OSTEOPENIA = "osteopenia"
CLASS_OSTEOPOROSIS = "osteoporosis"

CLASS_ALIASES = {
    "normal": CLASS_NORMAL,
    "healthy": CLASS_NORMAL,
    "normalknee": CLASS_NORMAL,
    "osteopenia": CLASS_OSTEOPENIA,
    "osteopeniaknee": CLASS_OSTEOPENIA,
    "osteoporosis": CLASS_OSTEOPOROSIS,
    "osteoporotic": CLASS_OSTEOPOROSIS,
    "osteoporosisknee": CLASS_OSTEOPOROSIS,
}

# Binary ablation task. Positive class = osteoporosis (documented everywhere).
BINARY_CLASSES = [CLASS_NORMAL, CLASS_OSTEOPOROSIS]
BINARY_LABELS = {CLASS_NORMAL: 0, CLASS_OSTEOPOROSIS: 1}
POSITIVE_CLASS = CLASS_OSTEOPOROSIS
POSITIVE_LABEL = 1

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

# Offline augmentation: number of augmented variants generated per TRAINING
# image, in addition to the un-augmented original (variant 0).
N_AUG_VARIANTS = 4

AUG_ROTATION_DEG = 12.0
AUG_ZOOM_RANGE = (0.90, 1.10)
AUG_BRIGHTNESS_RANGE = (0.85, 1.15)
AUG_CONTRAST_RANGE = (0.85, 1.15)
AUG_HORIZONTAL_FLIP = True   # left/right knee radiographs -> medically reasonable

# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
N_FOLDS = 5
VALIDATION_FRACTION_OF_TRAINVAL = 0.1765  # ~= 15% of the whole dataset

# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------
BACKBONES = ["vgg19", "inceptionresnetv2", "mobilenetv2"]

BACKBONE_FEATURE_DIMS = {
    "vgg19": 512,
    "inceptionresnetv2": 1536,
    "mobilenetv2": 1280,
}

BACKBONE_DISPLAY = {
    "vgg19": "VGG19",
    "inceptionresnetv2": "InceptionResNetV2",
    "mobilenetv2": "MobileNetV2",
}

# --------------------------------------------------------------------------
# Ablation configurations
# --------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "A1": {
        "name": "A1",
        "backbones": ["vgg19"],
        "display": "VGG19",
        "description": "Single backbone baseline (VGG19 only).",
    },
    "A2": {
        "name": "A2",
        "backbones": ["vgg19", "inceptionresnetv2"],
        "display": "VGG19 + InceptionResNetV2",
        "description": "Full hybrid with MobileNetV2 branch removed.",
    },
    "A3": {
        "name": "A3",
        "backbones": ["vgg19", "mobilenetv2"],
        "display": "VGG19 + MobileNetV2",
        "description": "Full hybrid with InceptionResNetV2 branch removed.",
    },
    "A4": {
        "name": "A4",
        "backbones": ["inceptionresnetv2", "mobilenetv2"],
        "display": "InceptionResNetV2 + MobileNetV2",
        "description": "Full hybrid with VGG19 branch removed.",
    },
    "A5": {
        "name": "A5",
        "backbones": ["vgg19", "inceptionresnetv2", "mobilenetv2"],
        "display": "VGG19 + InceptionResNetV2 + MobileNetV2",
        "description": "Full proposed hybrid model (all three branches).",
    },
}

FULL_MODEL = "A5"
ABLATION_ORDER = ["A1", "A2", "A3", "A4", "A5"]

# --------------------------------------------------------------------------
# Fusion head hyper-parameters
# --------------------------------------------------------------------------
HEAD_DENSE_UNITS = (256, 128)
HEAD_DROPOUT = (0.5, 0.3)
HEAD_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "sigmoid"      # binary task

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
MONITOR_METRIC = "val_loss"

# Decision threshold used to turn probabilities into hard labels.
DECISION_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
MULTIPLE_COMPARISON_METHOD = "holm"

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")

# Okabe-Ito colour-blind-safe qualitative palette.
# This specific assignment was chosen because it passes CVD separation on every
# adjacent pair (worst adjacent deltaE 11.0, deuteranopia); the more obvious
# ordering puts #009E73 next to #CC79A7, which only reaches deltaE 7.6.
# Low-contrast fills are relieved by direct value labels plus the CSV tables.
PALETTE = {
    "A1": "#E69F00",   # orange
    "A2": "#56B4E9",   # sky blue
    "A3": "#009E73",   # bluish green
    "A4": "#D55E00",   # vermillion
    "A5": "#0072B2",   # blue - full proposed model
}
NEUTRAL_DARK = "#333333"
NEUTRAL_MID = "#777777"
GRID_COLOR = "#D9D9D9"
SURFACE = "#FFFFFF"

# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
FEATURE_EXTRACTION_BATCH_SIZE = 8


def set_tf_threading_env() -> None:
    """Keep TensorFlow from oversubscribing this laptop's efficiency cores."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
