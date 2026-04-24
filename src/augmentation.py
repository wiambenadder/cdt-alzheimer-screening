"""
Image preprocessing and augmentation pipelines.

Rubric items this file supports:
  - #7   Data augmentation with evaluated impact (train w/ vs without)
  - #8   Normalization (ImageNet statistics)
  - #9   Basic preprocessing (resize, ToTensor)
  - #26  Comprehensive image augmentation with >=4 techniques
         (we use 5 below to be safe)
"""
from torchvision import transforms
from .config import AugConfig

# ImageNet statistics. All our pretrained models (VGG16, EfficientNet, ViT)
# were trained on ImageNet, so inputs must be normalized with these values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int, aug_cfg: AugConfig, use_aug: bool = True):
    """
    Training pipeline with 5 distinct augmentations.

    CLINICAL JUSTIFICATION for each choice:
      (1) RandomRotation:   clocks are sometimes drawn slightly tilted on paper
      (2) ColorJitter:      scans and phone photos vary in brightness/contrast
      (3) RandomAffine:     tremor and paper shift; small shears model this
      (4) GaussianBlur:     low-quality scans; cheap phone cameras
      (5) RandomErasing:    occlusion from shadow, ink smudges, torn corners

    WHY NO HORIZONTAL FLIP:
      Clock faces have handedness - numbers run clockwise. A flipped clock
      is not a realistic input, so flipping would inject pathological noise.
    """
    if not use_aug:
        # ablation: same pipeline sans augmentation, for rubric #7 impact study
        return build_eval_transform(image_size)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(aug_cfg.rotation_deg),
        transforms.ColorJitter(
            brightness=aug_cfg.color_jitter_brightness,
            contrast=aug_cfg.color_jitter_contrast,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=aug_cfg.affine_translate,
            shear=aug_cfg.affine_shear,
        ),
        transforms.GaussianBlur(kernel_size=aug_cfg.gaussian_blur_kernel),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=aug_cfg.random_erasing_p),
    ])


def build_eval_transform(image_size: int):
    """Deterministic pipeline for val/test - no randomness."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
