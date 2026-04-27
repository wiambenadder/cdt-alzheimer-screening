# Notebooks

Run these in order. Each writes its outputs to `results/<run_name>/` and
trained weights to `models/`. Between notebooks you can kill the kernel - all
state is persisted to disk.

| # | Notebook | What it does | Approx time on T4 |
|---|----------|--------------|-------------------|
| 01 | `01_data_exploration.ipynb` | Load labels, check column names, plot class histogram, view sample clocks, verify image loading | 15 min |
| 02 | `02_baseline.ipynb` | Majority-class baseline (no training); VGG16 frozen feature-extractor baseline | 1 hour |
| 03 | `03_finetuning.ipynb` | Fine-tune VGG16, EfficientNet-B0, and ViT-B/16. Run the 3-config hyperparameter sweep on the best architecture | 10 hours |
| 04 | `04_ablation.ipynb` | 2x2 ablation on ViT: {frozen, unfrozen} x {no-aug, aug} | 5 hours |
| 05 | `05_error_analysis.ipynb` | Load best checkpoint per model, compute all 7 metrics, plot confusion matrices, analyze top-12 errors, generate Grad-CAM + occlusion maps |  1.5 hours |

## Running on Kaggle or Colab

Both platforms can run these notebooks. A few platform notes:

- **Kaggle Notebooks**: Upload your NHATS data as a private Kaggle dataset,
  then add it as an input to your notebook. Paths will change - set
  `RAW_IMAGES_DIR` in the first cell.
- **Google Colab**: Mount Google Drive, place data under `/content/drive/...`,
  then override paths in the first cell.
