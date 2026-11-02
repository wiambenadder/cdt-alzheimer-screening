# This is for the preprocessing with edge-stripping crop for the NHATS page scans
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from torchvision import transforms
from .config import AugConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _center_square(img: Image.Image) -> Image.Image:
    W, H = img.size
    side = min(H, W)
    y0 = (H - side) // 2; x0 = (W - side) // 2
    return img.crop((x0, y0, x0 + side, y0 + side))


def _square_crop_with_padding(img: Image.Image, x0, y0, x1, y1) -> Image.Image:
    # we pad with white instead of clamping, so the crop is always square
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    side = int(round(max(x1 - x0, y1 - y0)))
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))

    canvas = Image.new(img.mode, (side, side), color="white")
    W, H = img.size
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(W, left + side), min(H, top + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas.paste(img.crop((sx0, sy0, sx1, sy1)), (sx0 - left, sy0 - top))
    return canvas


class SmartCropClock:
    """
    method="projection" is v1 (the original results), we keep it as is:
      Stage 1: we strip the outer 12% of each edge to remove the scanner bars
      Stage 2: we keep the rows/cols with more than 5% of the max ink density

    method="component" is v2:
      Stage 1: same edge strip
      Stage 2: we remove the leftover scanner bars (solid rectangles)
      Stage 3: we keep the biggest ink blob and the pieces close to it
      Stage 4: we pad to a square with white
    """

    def __init__(self, threshold: int = 200, margin_frac: float = 0.05,
                 edge_strip_frac: float = 0.12, min_box_frac: float = 0.005,
                 method: str = "component"):
        if method not in ("projection", "component"):
            raise ValueError(f"Unknown crop method {method!r}")
        self.threshold = threshold
        self.margin_frac = margin_frac
        self.edge_strip_frac = edge_strip_frac
        self.min_box_frac = min_box_frac
        self.method = method

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.method == "projection":
            return self._crop_projection(img)
        return self._crop_component(img)

    # v1
    def _crop_projection(self, img: Image.Image) -> Image.Image:
        gray = img.convert('L') if img.mode != 'L' else img
        arr = np.array(gray)
        H, W = arr.shape

        # Stage 1: we strip the outer 12% of each edge to remove the scanner bars
        ex = int(W * self.edge_strip_frac)
        ey = int(H * self.edge_strip_frac)
        inner = arr[ey:H - ey, ex:W - ex]

        dark = inner < self.threshold
        if dark.sum() < 100:
            # if no meaningful content is there, we return center crop
            return _center_square(img)

        # Stage 2: we use 1D density projection, to answer: where along each axis is most ink?
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
            return _center_square(img)

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

    # v2
    def _crop_component(self, img: Image.Image) -> Image.Image:
        gray = img.convert('L') if img.mode != 'L' else img
        arr = np.array(gray)
        H, W = arr.shape

        # Stage 1: edge strip
        ex = int(W * self.edge_strip_frac)
        ey = int(H * self.edge_strip_frac)
        dark = arr[ey:H - ey, ex:W - ex] < self.threshold
        if dark.sum() < 100:
            return _center_square(img)

        # we work on a smaller ink map so it stays fast, thin strokes still survive
        f = max(1, int(round(max(dark.shape) / 800)))
        h_s, w_s = dark.shape[0] // f, dark.shape[1] // f
        small = dark[:h_s * f, :w_s * f].reshape(h_s, f, w_s, f).any(axis=(1, 3))

        # Stage 2: a bar fills >85% of its box, a pen stroke never does
        # (even a filled-in circle is only pi/4 = 78%)
        labels, n = ndimage.label(small, structure=np.ones((3, 3)))
        if n > 0:
            boxes = ndimage.find_objects(labels)
            areas = ndimage.sum(small, labels, index=np.arange(1, n + 1))
            min_bar_area = 0.0002 * h_s * w_s
            for i, (box, area) in enumerate(zip(boxes, areas), start=1):
                bh = box[0].stop - box[0].start
                bw = box[1].stop - box[1].start
                fill = area / float(bh * bw)
                # a straight pen line also fills its box, but it is thin
                if fill > 0.85 and area >= min_bar_area and min(bh, bw) >= 4:
                    small[labels == i] = False

        if small.sum() < 5:
            return _center_square(img)

        # Stage 3: we dilate so the strokes of one clock merge into one blob
        radius = max(2, int(0.015 * max(h_s, w_s)))
        merged = ndimage.binary_dilation(small, iterations=radius)
        blobs, n_blobs = ndimage.label(merged)
        ink_per_blob = ndimage.sum(small, blobs, index=np.arange(1, n_blobs + 1))
        main = int(np.argmax(ink_per_blob)) + 1
        boxes = ndimage.find_objects(blobs)

        def ink_box(i):
            ys, xs = np.where(small & (blobs == i))
            return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

        y0, y1, x0, x1 = ink_box(main)
        main_side = max(y1 - y0, x1 - x0)
        main_ink = ink_per_blob[main - 1]
        for i in range(1, n_blobs + 1):
            if i == main:
                continue
            ink = ink_per_blob[i - 1]
            by, bx = boxes[i - 1]
            gap = max(0, by.start - y1, y0 - by.stop, bx.start - x1, x0 - bx.stop)
            # numbers drawn outside the circle count for the score, so small pieces
            # stay in when they are close. Far notes and dust stay out
            big_and_near = ink >= 0.05 * main_ink and gap < 0.5 * main_side
            small_and_close = ink >= 0.005 * main_ink and gap < 0.25 * main_side
            if big_and_near or small_and_close:
                fy0, fy1, fx0, fx1 = ink_box(i)
                y0, y1 = min(y0, fy0), max(y1, fy1)
                x0, x1 = min(x0, fx0), max(x1, fx1)

        # back to full page coordinates
        y0 = y0 * f + ey; y1 = y1 * f + ey
        x0 = x0 * f + ex; x1 = x1 * f + ex

        # just to check
        if (y1 - y0) * (x1 - x0) < self.min_box_frac * H * W * 0.1:
            return _center_square(img)

        # some small margin, then Stage 4
        side = max(y1 - y0, x1 - x0)
        m = int(side * self.margin_frac)
        return _square_crop_with_padding(img, x0 - m, y0 - m, x1 + m, y1 + m)


class ThickenStrokes:
    """
    At 224 the pen strokes end up less than a pixel wide and fade to light gray.
    We shrink to 4x the target first, then make the strokes a bit thicker.
    """

    def __init__(self, target_size: int):
        self.intermediate = 4 * target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        if max(W, H) > self.intermediate:
            scale = self.intermediate / max(W, H)
            img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BOX)
        return img.filter(ImageFilter.MinFilter(3))


def _crop_steps(image_size, crop_method, thicken_strokes):
    steps = [SmartCropClock(method=crop_method)]
    if thicken_strokes:
        steps.append(ThickenStrokes(image_size))
    steps.append(transforms.Resize((image_size, image_size)))
    return steps


# this creates the image preprocessing pipeline for training
# by applying the SmartCropClock to isolate the clock region, resizes to the target size,
# adds augmentations like rotation and color jitter, converts to grayscale tensor, normalizes
# using ImageNet statistics, and applies random erasing.
# rotation and affine fill the corners with white, the default was black
def build_train_transform(image_size, aug_cfg, use_aug=True,
                          crop_method="component", thicken_strokes=False):
    if not use_aug:
        return build_eval_transform(image_size, crop_method, thicken_strokes)
    return transforms.Compose(_crop_steps(image_size, crop_method, thicken_strokes) + [
        transforms.RandomRotation(aug_cfg.rotation_deg, fill=255),
        transforms.ColorJitter(brightness=aug_cfg.color_jitter_brightness,
                               contrast=aug_cfg.color_jitter_contrast),
        transforms.RandomAffine(degrees=0, translate=aug_cfg.affine_translate,
                                shear=aug_cfg.affine_shear, fill=255),
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

def build_eval_transform(image_size, crop_method="component", thicken_strokes=False):
    return transforms.Compose(_crop_steps(image_size, crop_method, thicken_strokes) + [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_display_transform(image_size, crop_method="component", thicken_strokes=False):
    # same crop and resize as the model, but as a PIL image for the plots
    return transforms.Compose(_crop_steps(image_size, crop_method, thicken_strokes))
