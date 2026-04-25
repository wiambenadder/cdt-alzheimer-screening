# This is for the preprocessing with edge-stripping crop for the NHATS page scans
import numpy as np
from PIL import Image
from torchvision import transforms
from .config import AugConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SmartCropClock:
    """
    We have 2 stages:
      Stage 1: we strip the outer 15% of each edge to remove the scanner bars
      Stage 2: within the inner region, we find the largest connected dark region
               (the clock) using dilation + component labeling, not just bbox
    """

    def __init__(self, threshold: int = 200, margin_frac: float = 0.05,
                 edge_strip_frac: float = 0.12, min_box_frac: float = 0.005):
        self.threshold = threshold
        self.margin_frac = margin_frac
        self.edge_strip_frac = edge_strip_frac
        self.min_box_frac = min_box_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert('L') if img.mode != 'L' else img
        arr = np.array(gray)
        H, W = arr.shape

        # Stage 1: we strip the outer 15% of each edge to remove the scanner bars
        ex = int(W * self.edge_strip_frac)
        ey = int(H * self.edge_strip_frac)
        inner = arr[ey:H - ey, ex:W - ex]

        dark = inner < self.threshold
        if dark.sum() < 100:
            # if no meaningful content is there, we return center crop
            side = min(H, W)
            y0 = (H - side) // 2; x0 = (W - side) // 2
            return img.crop((x0, y0, x0 + side, y0 + side))

        # Stage 2: we find the largest connected dark component 
        # we use 1D density projection, to answer: where along each axis is most ink?
        row_density = dark.sum(axis=1)
        col_density = dark.sum(axis=0)

        # we find the rows/cols that contain >5% of max density 
        row_thresh = max(5, row_density.max() * 0.05)
        col_thresh = max(5, col_density.max() * 0.05)
        dense_rows = np.where(row_density > row_thresh)[0]
        dense_cols = np.where(col_density > col_thresh)[0]

        if len(dense_rows) == 0 or len(dense_cols) == 0:
            return img

        y0 = dense_rows.min() + ey
        y1 = dense_rows.max() + ey
        x0 = dense_cols.min() + ex
        x1 = dense_cols.max() + ex

        # just to check
        if (y1 - y0) * (x1 - x0) < self.min_box_frac * H * W:
            side = min(H, W)
            y0c = (H - side) // 2; x0c = (W - side) // 2
            return img.crop((x0c, y0c, x0c + side, y0c + side))

        # some small margin
        bh, bw = y1 - y0, x1 - x0
        mh = int(bh * self.margin_frac)
        mw = int(bw * self.margin_frac)
        y0 = max(0, y0 - mh); y1 = min(H, y1 + mh)
        x0 = max(0, x0 - mw); x1 = min(W, x1 + mw)

        # we make it square
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        side = max(y1 - y0, x1 - x0)
        half = side // 2
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        x0 = max(0, cx - half); x1 = min(W, cx + half)

        return img.crop((x0, y0, x1, y1))

# this creates the image preprocessing pipeline for training
# by applying the SmartCropClock to isolate the clock region, resizes to the target size, 
# adds augmentations like rotation and color jitter, converts to grayscale tensor, normalizes 
# using ImageNet statistics, and applies random erasing.
def build_train_transform(image_size, aug_cfg, use_aug=True):
    if not use_aug:
        return build_eval_transform(image_size)
    return transforms.Compose([
        SmartCropClock(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(aug_cfg.rotation_deg),
        transforms.ColorJitter(brightness=aug_cfg.color_jitter_brightness,
                               contrast=aug_cfg.color_jitter_contrast),
        transforms.RandomAffine(degrees=0, translate=aug_cfg.affine_translate,
                                shear=aug_cfg.affine_shear),
        transforms.GaussianBlur(kernel_size=aug_cfg.gaussian_blur_kernel),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=aug_cfg.random_erasing_p),
    ])

# this creates the image preprocessing pipeline for validation and testing 
# it applies SmartCropClock to isolate the clock region, 
# it resizes to the target size, 
# it converts to grayscale tensor,
# normalizes using ImageNet statistics without any data augmentation.

def build_eval_transform(image_size):
    return transforms.Compose([
        SmartCropClock(),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
