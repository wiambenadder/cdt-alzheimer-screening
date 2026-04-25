"""
The Main configuration for this project.

All of the hyperparameters, paths, and settings are defined here for
reproducibility. For the ablation studies I will just construct a 
different config objects and re-run train().

"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


# The paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"

# The data layout 
#   data/nhats_raw/<participant_id>.tif, for clock images
#   data/labels.csv  that will have the cdt_score per participant
RAW_IMAGES_DIR = DATA_DIR / "nhats_raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_CSV = DATA_DIR / "labels.csv"

# The task
SEED = 42
NUM_CLASSES = 6         # from 0 to 5
CLASS_NAMES = [
    "0_severe_impairment",
    "1_significant_errors",
    "2_significant_errors_mild",
    "3_minor_errors_moderate",
    "4_minor_errors_mild",
    "5_normal",
]


# dataclass configurations (instead of free-floating args)
@dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # for the imbalance classes we can use"class_weights" (use weight CE loss), 
    # or "weighted_sampler" (to oversample minority classes), 
    # or "none" (no correction, to use as ablation baseline)
    imbalance_strategy: str = "class_weights"


@dataclass
class TrainConfig:
    epochs: int = 25
    lr_head: float = 1e-3        # we use higher LR for newly-initialized head
    lr_backbone: float = 1e-5    # we use lower LR for pretrained backbone
    weight_decay: float = 1e-4   # L2 regularization 
    dropout: float = 0.3         # dropout on head 

    optimizer: str = "adamw"     # "adamw" 
    lr_scheduler: str = "cosine" # "cosine" 

    early_stopping_patience: int = 5
    mixed_precision: bool = True       
    gradient_clip: float = 1.0          


@dataclass
class AugConfig:
    """5 augmentation techniques."""
    rotation_deg: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    affine_translate: Tuple[float, float] = (0.05, 0.05)
    affine_shear: float = 5.0
    gaussian_blur_kernel: int = 3
    random_erasing_p: float = 0.1
    # No horizontal flip because mirrored clocks don't happen in a clinical practice.


def model_comparison_runs() -> List[dict]:
    """We have 3 architectures with matching training config."""
    return [
        {"model_name": "vgg16",           "freeze_backbone": False, "run_name": "vgg16_ft"},
        {"model_name": "efficientnet_b0", "freeze_backbone": False, "run_name": "effb0_ft"},
        {"model_name": "vit_b16",         "freeze_backbone": False, "run_name": "vit_b16_ft"},
    ]


def ablation_runs(base_model: str = "vit_b16") -> List[dict]:
    """
    2x2 ablation: {frozen, unfrozen} x {no-aug, full-aug}
    We have 2 independent design choices with controlled comparison.
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
    """Three configurations (systematic HP tuning, these are 3 configs)."""
    return [
        TrainConfig(lr_head=1e-3, lr_backbone=1e-5, weight_decay=1e-4, dropout=0.3),
        TrainConfig(lr_head=5e-4, lr_backbone=5e-6, weight_decay=1e-3, dropout=0.5),
        TrainConfig(lr_head=2e-3, lr_backbone=2e-5, weight_decay=1e-5, dropout=0.1),
    ]
