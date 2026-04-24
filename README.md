# Deep Learning for Alzheimer's Screening from Clock Drawing Tests

Fine-tune and compare VGG16, EfficientNet-B0, and a Vision Transformer (ViT-B/16)
on the NHATS Clock Drawing Test dataset to classify cognitive impairment on the
6-point ordinal CDT scale (0 = severe impairment ... 5 = normal), with Grad-CAM /
occlusion-based interpretability and an ablation study across backbone-freezing
and data-augmentation choices.

## What it Does

This project builds a vision-based screening tool for cognitive impairment by
training deep models on Clock Drawing Test (CDT) images from the National
Health and Aging Trends Study (NHATS). A clinician's interpretation of a CDT
is a widely-used bedside screen for Alzheimer's and related dementias; our
system learns the same mapping from ~40,000 hand-drawn clocks to the NHATS
6-level impairment score. We fine-tune three pretrained architectures -
VGG16, EfficientNet-B0, and ViT-B/16 - compare them across seven evaluation
metrics, run a controlled ablation study, and visualize what the models
attend to via Grad-CAM (for CNNs) and patch occlusion (for the ViT), so that
failure modes are interpretable to a clinician.

## Quick Start

See `SETUP.md` for full install details. TL;DR:

```bash
# 1. clone and install
git clone https://github.com/<your-username>/cdt-alzheimer-screening
cd cdt-alzheimer-screening
pip install -r requirements.txt

# 2. place your NHATS data
# images -> data/nhats_raw/<participant_id>.tif
# labels -> data/labels.csv  (columns: participant_id, cdt_score)

# 3. reproduce the main experiments
python -m src.run_comparison     # trains all 3 models
python -m src.run_ablation       # 2x2 ablation on ViT
python -m src.run_eval           # writes metrics + confusion matrices + errors
```

Jupyter-first alternative: open `notebooks/01_data_exploration.ipynb` and
proceed through the numbered notebooks in order.

## Video Links

- **Demo video (3-5 min, non-technical pitch):** [INSERT YOUTUBE/DRIVE LINK]  
- **Technical walkthrough (5-10 min, for ML engineers):** [INSERT LINK]

Both are also in the `videos/` directory.

## Evaluation

### Architecture comparison on the held-out test set

| Model            | Accuracy | Macro-F1 | Quadratic-Kappa | Macro-AUC | ms/image |
|------------------|---------:|---------:|----------------:|----------:|---------:|
| Majority-class baseline | 0.00  | 0.00 | 0.00 | - | - |
| VGG16 fine-tuned        | 0.00  | 0.00 | 0.00 | 0.00 | 0.0 |
| EfficientNet-B0 fine-tuned | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| **ViT-B/16 fine-tuned** | **0.00** | **0.00** | **0.00** | **0.00** | 0.0 |

_Fill in after training. See `docs/results/comparison.csv` for the machine-readable version._

### Ablation study (ViT-B/16, 2x2 design)

| Backbone | Augmentation | Macro-F1 |
|----------|--------------|---------:|
| Frozen   | Off          | 0.00 |
| Frozen   | On           | 0.00 |
| Unfrozen | Off          | 0.00 |
| Unfrozen | On           | 0.00 |

### Reproduction of published results

We compare our best model to [PAPER TITLE AND CITATION] which reports
[METRIC = X] on NHATS. Our best run achieves [METRIC = Y], which [matches /
exceeds / is slightly below] the published number.

### Qualitative results

- `docs/results/<run_name>/confusion_matrix.png` - per-class confusion matrix  
- `docs/results/<run_name>/top_errors.png` - 12 most confidently-misclassified clocks  
- `docs/results/<run_name>/gradcam_examples.png` - Grad-CAM overlays on correct vs. wrong predictions  

## Individual Contributions

Solo project - all work (data preparation, model training, evaluation,
interpretability, documentation, videos) done by [YOUR NAME].

## Repository Structure

```
cdt-alzheimer-screening/
  src/                 all importable Python source
    config.py          hyperparameters + paths + experiment presets
    data.py            NHATS dataset, stratified splits, class weights
    augmentation.py    5 augmentation techniques + eval transform
    models.py          VGG16 / EfficientNet-B0 / ViT-B/16 factories
    train.py           training loop with AMP, schedulers, early stopping
    evaluate.py        metrics, confusion matrix, errors, timing
    gradcam.py         Grad-CAM (CNN) and occlusion maps (ViT)
    utils.py           seeding, checkpoint I/O, device selection
  data/                data lives here (gitignored); README explains layout
  models/              trained checkpoints (gitignored)
  notebooks/           numbered Jupyter notebooks for each phase
  videos/              demo + technical walkthrough
  docs/                rubric mapping, results, figures
  requirements.txt     pip dependencies
  environment.yml      conda alternative
  SETUP.md             install + run instructions
  ATTRIBUTION.md       sources, datasets, AI-tool usage
```
