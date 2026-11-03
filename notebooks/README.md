# Notebooks

Run these in order. Each writes its outputs to `results/<run_name>/` and
trained weights to `models/`. 

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

`06_results_report.ipynb` puts every run side by side (table, training curves, confusion matrices). It reads the results on Drive, so it needs no GPU and works after a disconnect.

`runs/B_vit_v2_seed42.ipynb`, `runs/B_vit_v2_seed43.ipynb`, `runs/B_vit_v2_seed44.ipynb` retrain the ViT with the v2 preprocessing, one notebook per seed. They continue a run that was cut by a disconnect and write `RUN_RECORD.md` on Drive. They are made from `runs/make_notebooks.py`.
