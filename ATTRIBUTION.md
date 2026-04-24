r# Attribution

Detailed attribution for all code, data, models, and AI tools used in this project.

## Dataset

### National Health and Aging Trends Study (NHATS), Clock Drawing Test

- Source: https://nhats.org/
- Access: Free with registration and data-use agreement.
- Citation: Kasper, J.D. and Freedman, V.A. National Health and Aging Trends Study
  User Guide: Rounds 1-9 Final Release. Baltimore: Johns Hopkins University
  School of Public Health.
- Scope used: ~40,000 clock drawings across rounds 1-9 (2011-2019) with 6-level
  CDT impairment scores.

## Pretrained Models

All pretrained weights come from `torchvision.models`:

| Model | Weights | License |
|-------|---------|---------|
| VGG16 | `VGG16_Weights.IMAGENET1K_V1` | BSD-3-Clause (PyTorch) |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | BSD-3-Clause |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | BSD-3-Clause |

## Libraries (see `requirements.txt` for exact pins)

- **PyTorch / torchvision** - core deep learning framework (BSD-3-Clause)
- **scikit-learn** - metrics, stratified split, class weights (BSD-3-Clause)
- **pandas, numpy** - data manipulation (BSD-3-Clause)
- **Pillow** - TIFF image loading (HPND)
- **matplotlib, seaborn** - plots and confusion matrices (BSD / matplotlib license)
- **tqdm** - progress bars (MIT / MPL-2.0)
- **TensorBoard** - training curve logging (Apache-2.0)

## Reference Research Papers

Our work was informed by the following NHATS-based CDT research:

- [REPLACE WITH EXACT PAPERS YOU REFERENCE, e.g.]:
  - Author et al. (2023). "Vision Transformers for Automated Scoring of the Clock
    Drawing Test Using NHATS." _Venue_. (reproduction target)
  - Author et al. (2022). "EfficientNet and ResNet Baselines on NHATS CDT."
    _Venue_. (baseline comparison)

Our reproduction target is specifically [paper + metric]; see the Evaluation
section of `README.md` for our reproduction results.

## Open-Source Reference Repositories

The following repositories informed our approach but none of their code was
copied directly:

- noc-lab/CDT - deep learning CDT model on FHS data
- trebledawson/Alzheimers-Clock-Drawing - CNN on RowanSOM dataset
- cccnlab/CDT-API-Network - Attentive Pairwise Interaction Network
- Collinjia/Multiclassification-Alzheimer-Detection - AIcrowd multi-class CDT code

## AI Tool Usage

Per the course policy, a substantive account of how AI development tools
were used in this project:

### Tools used
- **Claude (Anthropic)** - used for code scaffolding, documentation drafting,
  debugging guidance, and literature-style prose in the README.

### What was AI-generated
- Initial skeletons of `src/config.py`, `src/data.py`, `src/augmentation.py`,
  `src/models.py`, `src/train.py`, `src/evaluate.py`, `src/gradcam.py`
- First drafts of `README.md`, `SETUP.md`, this file, and the notebook
  structure
- Comments explaining which rubric items each file supports

### What I modified / fixed / reworked
_(fill in honestly as you work - this is what scores the 3 points for
rubric item #99)_

Examples of things to record here as you go:
- "ViT attention rollout was initially generated as a naive layer-product but
  failed because torchvision ViT uses `F.scaled_dot_product_attention` which
  does not expose weights. I replaced the attention-rollout implementation
  with occlusion-based attribution."
- "Initial class-weight computation used all classes present in the full
  dataset. I changed it to compute weights from the train split only, to
  avoid label leakage from val/test."
- "Generated code assumed `.jpg` extension but NHATS ships `.tif`; changed
  image loading and added explicit RGB conversion."
- "Generated training loop used a single global LR; added parameter groups
  for differential LR between backbone and head after comparing validation
  curves."
- "Debugged CUDA OOM at batch_size=64 on ViT-B/16 by lowering to 32 and
  enabling mixed precision."
- [keep adding entries as you encounter them - specific beats generic]

### What I wrote independently
- All experiment-design decisions (which 15 rubric items to claim, which
  ablation axes to vary, which reproduction target to use)
- The interpretation and discussion of failure modes in the error analysis
- The actual training runs, hyperparameter choices, and reporting of
  real numbers
