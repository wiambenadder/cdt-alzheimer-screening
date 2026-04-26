r# Attribution

This file documents all of the external resources, datasets, AI tools, 
and code sources used in this project.

---

## Reference Research Papers

### Primary Reproduction Target

**Hu, M., Qin, T., Gonzalez, R., Freedman, V. A., Zahodne, L. B., Melipillán, E. R., & Murphey, Y. L. (2026).** A novel vision transformer model produces clock drawing test scores as accurate as expert human coders. *Scientific Reports, 16*, 4032.  
https://doi.org/10.1038/s41598-025-34064-6

We reproduce their core 3-architecture comparison (ResNet101, EfficientNet, and ViT) on NHATS ordinal CDT scoring, with VGG16 substituted for ResNet101 as a representative CNN baseline. Both are ImageNet-pretrained, pre-transformer CNNs from the same era. The paper's central claim that ViT outperforms CNN architectures is preserved in our comparison since we retain EfficientNet-B0 and ViT-B/16 from the original study.

### Supporting Architecture Papers

**Simonyan, K., & Zisserman, A. (2015).** Very deep convolutional networks for large-scale image recognition. *ICLR 2015.*  
https://arxiv.org/abs/1409.1556  
VGG16 architecture used as CNN baseline.

**Tan, M., & Le, Q. (2019).** EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019.*  
https://arxiv.org/abs/1905.11946  
EfficientNet-B0 architecture.

**Dosovitskiy, A., et al. (2021).** An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021.*  
https://arxiv.org/abs/2010.11929  
ViT-B/16 architecture.

**Selvaraju, R. R., et al. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017.*  
https://arxiv.org/abs/1610.02391  
Grad-CAM interpretability method.

---

## Dataset

**National Health and Aging Trends Study (NHATS), Rounds 1–14 (2011–2024).**  
Sponsor: National Institute on Aging.  
Data access: https://nhats.org/researcher/data-access/public-use-files

| Detail | Value |
|---|---|
| Clock drawing images | ~73,769 TIFF scans across 14 rounds |
| Labels | NHATS CDT ordinal score (0–5), extracted from SP questionnaire files |
| Data use agreement | Accepted prior to download. Raw data is gitignored and not distributed in this repository |
| Participant IDs | De-identified in NHATS public-use files |

## Pretrained Model Weights

| Model | Source | License |
|---|---|---|
| VGG16 | `torchvision.models.vgg16(pretrained=True)` — ImageNet weights | BSD |
| EfficientNet-B0 | `torchvision.models.efficientnet_b0(pretrained=True)` — ImageNet weights | BSD |
| ViT-B/16 | `torchvision.models.vit_b_16(pretrained=True)` — ImageNet weights | BSD |

All pretrained weights are downloaded automatically via PyTorch Hub on first run.

---

## Libraries and Frameworks

| Library | Use |
|---|---|
| PyTorch | Core deep learning framework |
| torchvision | Model architectures, pretrained weights, and transforms |
| grad-cam (pytorch-grad-cam) | Grad-CAM and occlusion interpretability |
| scikit-learn | GroupShuffleSplit, compute_class_weight, and metrics |
| pandas, numpy | Data manipulation |
| Pillow (PIL) | TIFF image loading and conversion |
| matplotlib, seaborn | Visualization |
| tqdm | Training progress bars |

---

## AI Tools Used

This project used AI coding assistants (Claude Code) during development.

### What AI Tools Were Used For

| Task | Used AI |
|---|---|
| Debugging code related errors (scheduler ordering, AMP deprecation warnings) | Yes |
| Debugging scheduler ordering | Claude identified that scheduler.step() was being called before optimizer.step(), which causes PyTorch to skip the first LR value. I verified the fix and updated src/train.py. |
| Debugging AMP deprecation warnings | Claude suggested updating torch.cuda.amp.autocast to torch.amp.autocast and similarly for GradScaler. I tested that the updated calls produced the same training behavior. |
| Grammar and documentation phrasing | Claude reviewed README and ATTRIBUTION.md for grammar. All technical decisions and result interpretations are my own. |

### What AI Tools Were Not Used For

| Task | Used AI |
|---|---|
| Data exploration findings and decision-making (SmartCropClock design, class imbalance strategy, participant-disjoint split) | No |
| Designing the core experimental methodology (architecture choices, split design, metric selection) | No |
| Interpreting results or writing the technical analysis | No |
| Documentation drafting | No |

All core design decisions were made independently: the participant-disjoint split design,
the SmartCropClock algorithm, the choice to use GroupShuffleSplit on participant_id,
the architecture selection, the metric choices, and all result interpretation.
---

## What I Modified, Fixed, and Modified

### Participant-Disjoint Split (`src/data.py`)

NHATS is a longitudinal study, which means that the same participant can appear in up to 14 rounds. A random-stratified split would leak the same person across the train and test sets. I designed `participant_disjoint_split()` using `GroupShuffleSplit` on `participant_id` to enforce zero participant overlap. This is a stronger evaluation procedure than the one used in the original paper.

### SmartCropClock Preprocessing (`src/augmentation.py`)

During data exploration, I found that NHATS clock drawings occupy only about 10% of each scanned page, with the rest being blank paper and scanner registration bars. A naive `Resize(224)` reduces the clock to roughly 10 to 20 pixels of usable signal. To fix this, I designed `SmartCropClock`, which does the following:

1. Strips the outer 12% of each edge to remove the scanner artifacts
2. Finds the clock region using per-axis ink-density projection, which handles stray specks better than a simple bounding box would
3. Pads to a square crop with a 5% margin

I validated this by manually reviewing examples from each of the 6 score classes, as documented in notebooks/01_data_exploration.ipynb.

### VGG16 Substituted for ResNet101

The target paper tested ResNet101 as its CNN baseline. I use VGG16 instead, a comparable ImageNet-pretrained CNN from the same era, because VGG16 is simpler to fine-tune (no residual connections examine) and is a well-understood baseline for transfer learning comparisons. This was a methodological choice.

### Fixed `scheduler.step()` Ordering (`src/train.py`)

The LR scheduler was originally being called before `optimizer.step()`, which causes PyTorch to skip the first learning rate value. I fixed the order to: `optimizer.step()` then `scaler.update()` then `scheduler.step()`.

### Updated AMP API Calls (`src/train.py`)

`torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` are deprecated in PyTorch 2.x. I updated these to `torch.amp.autocast('cuda', ...)` and `torch.amp.GradScaler('cuda', ...)`.

### Extended Dataset Scope

Hu et al. (2026) used only Rounds 1–9 and applied a quality filter, utilizing 24,991 images. This project uses all available Rounds 1–14 with minimal filtering, resulting in 59,417 images, which is 2.4 times more training data. Rounds 13 and 14 have a different CDT variable structure because NHATS changed their annotation workflow in those rounds. This is documented in `notebooks/01_data_exploration.ipynb`.
