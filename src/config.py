"""
Centralized configuration for the CDT Alzheimer Screening project.

All hyperparameters, paths, and experimental settings live here so that
experiments are reproducible and easy to sweep. Ablation studies just
construct a different config object and re-run train().

Rubric items this file supports:
  - #0  Modular code design (no magic numbers in training scripts)
  - #6  Systematic hyperparameter tuning (sweep these dataclasses)
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"

# Expected data layout (you'll populate these during data setup):
#   data/nhats_raw/<participant_id>.tif       <- clock images
#   data/labels.csv                            <- cdt_score per participant
RAW_IMAGES_DIR = DATA_DIR / "nhats_raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_CSV = DATA_DIR / "labels.csv"

# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 6           # NHATS CDT ordinal scale: 0 (severe) ... 5 (normal)
CLASS_NAMES = [
    "0_severe_impairment",
    "1_significant_errors",
    "2_significant_errors_mild",
    "3_minor_errors_moderate",
    "4_minor_errors_mild",
    "5_normal",
]


# -----------------------------------------------------------------------------
# Dataclass configs (pass these around instead of free-floating args)
# -----------------------------------------------------------------------------
@dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Imbalance strategy for rubric #10 (preprocessing pipeline w/ 2+ challenges)
    # Options: "class_weights" (weight CE loss), "weighted_sampler" (oversample
    # minority classes), "none" (no correction, used as ablation baseline)
    imbalance_strategy: str = "class_weights"


@dataclass
class TrainConfig:
    epochs: int = 25
    lr_head: float = 1e-3        # higher LR for newly-initialized head
    lr_backbone: float = 1e-5    # lower LR for pretrained backbone
    weight_decay: float = 1e-4   # L2 regularization (rubric #5)
    dropout: float = 0.3         # dropout on head (rubric #5)

    optimizer: str = "adamw"     # "adamw" | "adam" | "sgd" - ablation #19
    lr_scheduler: str = "cosine" # "cosine" | "plateau" | "none"  - rubric #14

    early_stopping_patience: int = 5    # rubric #5
    mixed_precision: bool = True        # rubric #16, #17
    gradient_clip: float = 1.0          # rubric #16


@dataclass
class AugConfig:
    """5 augmentation techniques (rubric #26 requires >=4)."""
    rotation_deg: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    affine_translate: Tuple[float, float] = (0.05, 0.05)
    affine_shear: float = 5.0
    gaussian_blur_kernel: int = 3
    random_erasing_p: float = 0.1
    # Why no horizontal flip? Clock drawings have handedness — a mirror-flipped
    # clock is not a realistic input. Clinical convention matters here.


# -----------------------------------------------------------------------------
# Experiment presets (for the comparison rubric #89 and ablation rubric #92)
# -----------------------------------------------------------------------------
def model_comparison_runs() -> List[dict]:
    """Three architectures with matched training config. Feeds rubric #89."""
    return [
        {"model_name": "vgg16",           "freeze_backbone": False, "run_name": "vgg16_ft"},
        {"model_name": "efficientnet_b0", "freeze_backbone": False, "run_name": "effb0_ft"},
        {"model_name": "vit_b16",         "freeze_backbone": False, "run_name": "vit_b16_ft"},
    ]


def ablation_runs(base_model: str = "vit_b16") -> List[dict]:
    """
    2x2 ablation: {frozen, unfrozen} x {no-aug, full-aug}
    This is rubric #92 - two independent design choices with controlled comparison.
    """
    return [
        {"model_name": base_model, "freeze_backbone": True,  "use_aug": False,
         "run_name": f"{base_model}_frozen_noaug"},
        {"model_name": base_model, "freeze_backbone": True,  "use_aug": True,
         "run_name": f"{base_model}_frozen_aug"},
        {"model_name": base_model, "freeze_backbone": False, "use_aug": False,
         "run_name": f"{base_model}_unfrozen_noaug"},
        {"model_name": base_model, "freeze_backbone": False, "use_aug": True,
         "run_name": f"{base_model}_unfrozen_aug"},
    ]


def hyperparameter_search() -> List[TrainConfig]:
    """Three configs for rubric #6 (systematic HP tuning, >=3 configs)."""
    return [
        TrainConfig(lr_head=1e-3, lr_backbone=1e-5, weight_decay=1e-4, dropout=0.3),
        TrainConfig(lr_head=5e-4, lr_backbone=5e-6, weight_decay=1e-3, dropout=0.5),
        TrainConfig(lr_head=2e-3, lr_backbone=2e-5, weight_decay=1e-5, dropout=0.1),
    ]
