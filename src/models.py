"""
Models: VGG16, EfficientNet-B0, and ViT-B/16.

Each part returns (model, backbone_params, head_params) where the
param lists support differential learning rates (lower on pretrained
backbone, higher on newly-initialized head).

This files fulfills the following rubric items:
  - 1. rubric #21  Used a pretrained model as frozen feature extractor    (freeze_backbone=True)
  - 2. rubric #22  Modified/Adapted pretrained model by fine-tuning       (freeze_backbone=False)
  - 3. rubric #24  Used a pretrained vision CNN (VGG16 / EfficientNet)
  - 4. rubric #25  Modified/Adapted vision CNN by fine-tuning on dataset
  - 5. rubric #28  Used a vision transformer
  - 6. rubric #29  Modified/Adapted vision transformer by fine-tuning     

I need to claim ONE tier per item. My strongest claim is #29 for ViT.
"""
import torch.nn as nn
import torchvision.models as tv

from .config import NUM_CLASSES

# VGG16
def build_vgg16(num_classes: int = NUM_CLASSES,
                freeze_backbone: bool = False,
                dropout: float = 0.3):
    """
    VGG16 is pretrained on ImageNet. We replace the final FC layer with a small
    head sized for NUM_CLASSES. We optionally freeze the convolutional
    feature extractor for the frozen-baseline ablation.
    """
    model = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1)

    # in the original: model.classifier[-1] = Linear(4096, 1000) for ImageNet
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return model, backbone_params, head_params

# EfficientNet-B0
def build_efficientnet_b0(num_classes: int = NUM_CLASSES,
                          freeze_backbone: bool = False,
                          dropout: float = 0.3):
    """The EfficientNet-B0 is pretrained on ImageNet."""
    model = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return model, backbone_params, head_params


# ViT-B/16
def build_vit_b16(num_classes: int = NUM_CLASSES,
                  freeze_backbone: bool = False,
                  dropout: float = 0.3):
    """
    Vision Transformer (ViT-B/16) is pretrained on ImageNet.
    We replace the classification head and optionally freeze the encoder blocks.
    """
    model = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        # we freeze everything except the heads
        for name, p in model.named_parameters():
            if not name.startswith("heads"):
                p.requires_grad = False

    head_params = [p for p in model.heads.parameters() if p.requires_grad]
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("heads")
    ]
    return model, backbone_params, head_params


# uniform construction (rubric #89)
MODEL_REGISTRY = {
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b16": build_vit_b16,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def get_gradcam_target_layer(model, model_name: str):
    """
    This returns the last convolutional block, which is the standard Grad-CAM target.
    For ViT we handle interpretability separately via attention rollout.
    """
    if model_name == "vgg16":
        return model.features[28]       # last convolutional layer of VGG16
    if model_name == "efficientnet_b0":
        return model.features[-1]       # last convolutional block
    if model_name == "vit_b16":
        raise ValueError(
            "ViT does not have conv layers; use attention_rollout() instead."
        )
    raise ValueError(f"Unknown model {model_name}")
