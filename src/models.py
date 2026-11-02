"""
Models: VGG16, EfficientNet-B0, and ViT-B/16 (at 224 or 384 pixels).

Each part returns (model, backbone_params, head_params) where the
param lists support differential learning rates (lower on pretrained
backbone, higher on newly-initialized head).

pretrained=False is only for the offline tests.
"""
import torch.nn as nn
import torchvision.models as tv

from .config import NUM_CLASSES


def _split_params(model, head_prefix: str):
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith(head_prefix)]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith(head_prefix)]
    return backbone_params, head_params


# VGG16
def build_vgg16(num_classes: int = NUM_CLASSES,
                freeze_backbone: bool = False,
                dropout: float = 0.3,
                pretrained: bool = True):
    """
    VGG16 is pretrained on ImageNet. We replace the final FC layer with a small
    head sized for NUM_CLASSES. We optionally freeze the convolutional
    feature extractor for the frozen-baseline ablation.
    """
    weights = tv.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv.vgg16(weights=weights)

    # in the original: model.classifier[-1] = Linear(4096, 1000) for ImageNet
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    backbone_params, head_params = _split_params(model, "classifier")
    return model, backbone_params, head_params


# EfficientNet-B0
def build_efficientnet_b0(num_classes: int = NUM_CLASSES,
                          freeze_backbone: bool = False,
                          dropout: float = 0.3,
                          pretrained: bool = True):
    """The EfficientNet-B0 is pretrained on ImageNet."""
    weights = tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv.efficientnet_b0(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    backbone_params, head_params = _split_params(model, "classifier")
    return model, backbone_params, head_params


# ViT-B/16
def build_vit_b16(num_classes: int = NUM_CLASSES,
                  freeze_backbone: bool = False,
                  dropout: float = 0.3,
                  pretrained: bool = True,
                  image_size: int = 224):
    """
    Vision Transformer (ViT-B/16) is pretrained on ImageNet.
    We replace the classification head and optionally freeze the encoder blocks.

    image_size=384 uses torchvision's SWAG weights, to test if more resolution
    helps with the small numbers (it's slower, 576 patches instead of 196).
    """
    if image_size == 224:
        weights = tv.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.vit_b_16(weights=weights)
    elif image_size == 384:
        weights = tv.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        model = tv.vit_b_16(weights=weights, image_size=384)
    else:
        raise ValueError(f"vit_b16 supports image_size 224 or 384, got {image_size}")

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

    backbone_params, head_params = _split_params(model, "heads")
    return model, backbone_params, head_params


def build_vit_b16_384(**kwargs):
    return build_vit_b16(image_size=384, **kwargs)


# uniform construction
MODEL_REGISTRY = {
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b16": build_vit_b16,
    "vit_b16_384": build_vit_b16_384,
}

MODEL_INPUT_SIZE = {
    "vgg16": 224,
    "efficientnet_b0": 224,
    "vit_b16": 224,
    "vit_b16_384": 384,
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
    For ViT we use patch occlusion instead (gradcam.vit_occlusion_map).
    """
    if model_name == "vgg16":
        return model.features[28]       # last convolutional layer of VGG16
    if model_name == "efficientnet_b0":
        return model.features[-1]       # last convolutional block
    if model_name.startswith("vit_b16"):
        raise ValueError(
            "ViT does not have conv layers; use gradcam.vit_occlusion_map() instead."
        )
    raise ValueError(f"Unknown model {model_name}")
