# Rubric Mapping

This file maps each of the 15 rubric items we claim to the specific code,
notebook, or artifact that provides evidence. Graders can use this as an
index. This also doubles as raw material for the self-assessment template
you submit on Gradescope.

## Category 1: Machine Learning (target 73 / 73, raw sum ~91)

| # | Item | Points | Evidence location |
|---|------|--------|-------------------|
| 1 | Modular code design with reusable functions and classes | 3 | `src/*.py` - clean module separation; no logic in `__main__` blocks except thin CLI wrappers |
| 2 | Baseline model for comparison | 3 | `notebooks/02_baseline.ipynb` - majority-class baseline + frozen-VGG16 baseline |
| 3 | Training curves tracked and visualized | 3 | `src/train.py` TensorBoard logging; plots in `notebooks/03_finetuning.ipynb`; `docs/results/<run>/tb/` |
| 4 | Regularization (>=2 of L2, dropout, early stopping) | 5 | `src/train.py` - AdamW `weight_decay`, `nn.Dropout` in each model head, early-stopping loop |
| 5 | Systematic hyperparameter tuning (>=3 configs) | 5 | `config.hyperparameter_search()` + `notebooks/03_finetuning.ipynb` comparison table |
| 6 | Data augmentation with evaluated impact | 5 | `src/augmentation.py`; impact measured in `notebooks/04_ablation.ipynb` (aug vs. no-aug rows) |
| 7 | Preprocessing pipeline addressing >=2 quality challenges | 7 | `src/data.py` - class imbalance (weighted sampler + class weights), image heterogeneity (TIFF->RGB, resize, normalize) |
| 8 | Modified/Adapted vision transformer by fine-tuning | 7 | `src/models.py::build_vit_b16` with `freeze_backbone=False`; trained in `notebooks/03_finetuning.ipynb` |
| 9 | Comprehensive image augmentation (>=4 techniques) | 5 | `src/augmentation.py` - rotation, color jitter, affine, gaussian blur, random erasing = 5 techniques |
| 10 | Compared multiple architectures quantitatively | 7 | `src/evaluate.py::aggregate_results` + README comparison table + `docs/results/comparison.csv` |
| 11 | Error analysis with failure-case visualization | 7 | `src/evaluate.py::analyze_errors` + `docs/results/<run>/top_errors.png` + discussion in notebook |
| 12 | Ablation study (>=2 independent design choices) | 7 | `config.ablation_runs()` - 2x2 frozen x augmentation; results in `docs/results/ablation.csv` |
| 13 | Interpretable model design or explainability analysis | 7 | `src/gradcam.py` (Grad-CAM for CNNs, occlusion for ViT); examples in `notebooks/05_error_analysis.ipynb` |
| 14 | Reproduced quantitative results from published paper | 10 | Comparison to [CITED PAPER] in README Evaluation section; numbers in `docs/results/reproduction.md` |
| 15 | Solo project credit | 10 | Only one contributor; git commit history shows single author |

**Raw total:** 3+3+3+5+5+5+7+7+5+7+7+7+7+10+10 = **91** (caps at 73)

## Category 2: Following Directions (target 15 / 15)

| Item | Evidence |
|------|----------|
| Self-assessment submitted | Gradescope submission |
| SETUP.md present | `SETUP.md` |
| ATTRIBUTION.md present | `ATTRIBUTION.md` |
| requirements.txt present | `requirements.txt` + `environment.yml` |
| README has What It Does section | `README.md` |
| README has Quick Start section | `README.md` |
| README has Video Links section | `README.md` |
| README has Evaluation section | `README.md` |
| README has Individual Contributions section | `README.md` |
| Demo video correct length | `videos/demo.mp4` (3-5 min) |
| Technical walkthrough correct length | `videos/technical_walkthrough.mp4` (5-10 min) |
| Workshop attendance (if applicable) | up to 4 points |

## Category 3: Project Cohesion and Motivation (target 15 / 15)

Every one of the 15 ML items above maps directly to the single research
question: _"Can deep vision models reliably score NHATS clock drawings on
the 6-level CDT impairment scale, and do they attend to clinically
meaningful regions?"_

| Cohesion item | Evidence |
|---------------|----------|
| README articulates unified goal | `README.md` opening paragraph |
| Demo explains why project matters to non-technical audience | `videos/demo.mp4` |
| Project addresses real-world problem | CDT is a clinically-used dementia screen; NHATS is a real federally-funded longitudinal cohort |
| Technical walkthrough shows synergy of components | `videos/technical_walkthrough.mp4` |
| Progression problem -> approach -> solution -> evaluation | README structure mirrors this; notebooks numbered in this order |
| Design choices justified | `ATTRIBUTION.md` (what I modified) + comments in each module |
| Metrics directly measure stated objectives | All 7 metrics relate to 6-class ordinal CDT scoring |
| No superfluous "point collecting" | Every rubric item above is used by the main pipeline |
| Clean codebase | no dead files, modules <300 lines each |
