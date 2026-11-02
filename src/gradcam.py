"""
This file has the code for the interpretability maps: Grad-CAM for the
convolutional models and patch occlusion for the Vision Transformer.

Grad-CAM works on CNNs by weighting activation maps of a chosen convolutional
layer by the gradient of the target class with respect to those activations. The
heatmap shows which spatial regions most increased the predicted class logit.

ViT has no convolutional layers, so for it we use patch occlusion: we cover
one patch at a time and see how much the prediction drops.
"""
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .augmentation import IMAGENET_MEAN, IMAGENET_STD


# GradCam
class GradCAM:
    """
    This hooks into a target convolutional layer, records the activations (forward) and
    gradients (backward), then produces a class-discriminative heatmap.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None
                 ) -> Tuple[np.ndarray, int]:
        """
        x: [1, 3, H, W] image tensor (already normalized)
        It returns (cam_HxW in [0,1] numpy, class_idx used).
        """
        self.model.eval()
        # this enable grads even if model is in evaluation mode
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self.gradients[0]     # [C, H, W]
        acts = self.activations[0]    # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.cpu().numpy(), class_idx


def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray,
                         alpha: float = 0.4, title: str = "") -> plt.Figure:
    """This resize cam to image size and overlay as a jet-colored heatmap."""
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
             .resize(pil_img.size, Image.BILINEAR)
    ) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pil_img)
    ax.imshow(cam_resized, cmap="jet", alpha=alpha)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig


def denormalize(x: torch.Tensor) -> np.ndarray:
    # we undo the normalization to show exactly what the model saw
    if x.dim() == 4:
        x = x[0]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[:, None, None]
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def paper_white(device=None) -> torch.Tensor:
    # a white pixel after normalization, shape [3, 1, 1]
    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)
    return ((1.0 - mean) / std)[:, None, None]


@torch.no_grad()
def vit_occlusion_map(model, x: torch.Tensor, patch_size: int = 16,
                      class_idx: Optional[int] = None,
                      fill: str = "white", batch_size: int = 64) -> np.ndarray:
    """
    The occlusion-based interpretability for ViT: we cover one patch with blank
    paper and measure how much the target-class probability drops. A big drop
    means that patch was important for the prediction.

    fill="white" is the default now. fill="gray" is v1, but on white paper a gray
    patch looks like ink, so it adds something instead of removing it.
    """
    model.eval()
    _, _, H, W = x.shape
    base_probs = torch.softmax(model(x), dim=1)
    if class_idx is None:
        class_idx = int(base_probs.argmax(dim=1).item())
    base_score = base_probs[0, class_idx].item()

    if fill == "white":
        fill_value = paper_white(x.device)
    elif fill == "gray":
        fill_value = torch.zeros(3, 1, 1, device=x.device)
    else:
        raise ValueError(f"Unknown fill {fill!r}")

    n_h, n_w = H // patch_size, W // patch_size
    coords = [(i, j) for i in range(n_h) for j in range(n_w)]
    scores = []

    # we run the patches in batches, much faster than one at a time
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        x_occ = x.repeat(len(chunk), 1, 1, 1)
        for k, (i, j) in enumerate(chunk):
            x_occ[k, :,
                  i*patch_size:(i+1)*patch_size,
                  j*patch_size:(j+1)*patch_size] = fill_value
        probs = torch.softmax(model(x_occ), dim=1)
        scores.append(probs[:, class_idx].cpu().numpy())

    scores = np.concatenate(scores)
    heatmap = (base_score - scores).reshape(n_h, n_w).astype(np.float32)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def plot_attribution_grid(model, test_df, y_true, y_pred, eval_tf, device,
                          save_path=None, patch_size: int = 16, n_per_class: int = 1,
                          title: str = "What ViT-B/16 looks at when scoring clocks"):
    """
    One row per class: the model input on the left and the occlusion map on top of
    that same input on the right. Before, NB05 drew the map on the uncropped page,
    so the two didn't line up.
    """
    from .config import NUM_CLASSES

    picks = []
    for cls in range(NUM_CLASSES):
        idx = np.where((y_true == cls) & (y_pred == cls))[0]
        if len(idx) == 0:
            idx = np.where(y_true == cls)[0]   # fallback: any example of this class
        picks += [(cls, i) for i in idx[:n_per_class]]

    fig, axes = plt.subplots(len(picks), 2, figsize=(9, 4.5 * len(picks)), squeeze=False)
    for row, (cls, idx) in enumerate(picks):
        img = Image.open(test_df.iloc[idx]["image_path"]).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        heatmap = vit_occlusion_map(model, x, patch_size=patch_size,
                                    class_idx=int(y_pred[idx]))
        shown = denormalize(x)
        heat_full = np.kron(heatmap, np.ones((patch_size, patch_size)))

        axes[row, 0].imshow(shown)
        axes[row, 0].set_title(f"model input: true={y_true[idx]} pred={y_pred[idx]}")
        axes[row, 1].imshow(shown)
        axes[row, 1].imshow(heat_full, cmap="jet", alpha=0.45,
                            extent=(0, shown.shape[1], shown.shape[0], 0))
        axes[row, 1].set_title("occlusion attribution (red = important)")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(title, fontsize=14, y=1.0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
