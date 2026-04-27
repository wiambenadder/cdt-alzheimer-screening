# Deep Learning for Alzheimer's Screening from Clock Drawing Tests

Fine-tune and compare VGG16, EfficientNet-B0, and a Vision Transformer (ViT-B/16)
on the NHATS Clock Drawing Test dataset. Each model is trained to rate cognitive 
impairment on a 6-point scale (0 = severe, 5 = normal). To understand what 
each model looks at when making predictions, we use Grad-CAM (for CNNs) and patch 
occlusion (for the ViT). We also run a 2×2 ablation study, testing whether freezing 
the backbone and/or applying data augmentation improves performance.

Reproducing and extending: Hu, M., Qin, T., Gonzalez, R., Freedman, V. A., Zahodne, 
L. B., Melipillán, E. R., & Murphey, Y. L. (2026). A novel vision transformer model 
produces clock drawing test scores as accurate as expert human coders. Scientific 
Reports, 16, 4032.
https://www.nature.com/articles/s41598-025-34064-6

## Motivation

Most standard cognitive screening tools, including the MMSE and the MoCA, require 
reading and writing. In Morocco, 51% of adults over 50 cannot read or write 
(Morocco World News, 2024). Across the WHO African region, there is fewer than one 
neurologist per 100,000 people (Naji et al., 2022; Charway-Felli, 2023). For most 
families in these settings, a diagnosis is not delayed. It simply never comes.

The Clock Drawing Test is different. A doctor hands someone a blank piece of paper 
and says: draw a clock, put in all the numbers, and set the hands to ten past eleven. 
No literacy required. No specialist equipment. Under two minutes. Clinicians have 
used it because it works across languages and educational backgrounds (Hu et al., 2026). 
The only bottleneck is scoring, which still requires a trained expert.

This project asks whether a computer can learn to do that automatically, from a 
photograph. We train on nearly 60,000 real clock drawings from the National Health 
and Aging Trends Study, matching and slightly exceeding the results of the paper we 
reproduce, with a kappa of 0.812 compared to their 0.81 (Hu et al., 2026). The 
model scores one image in 2.3 ms, fast enough to run on a phone.

Phones are already in people's hands. Morocco's mobile penetration rate is 159.5% 
(MeaTechWatch, 2025). The technology to reach families in places far from any 
hospital already exists. What is missing is the tool. This project is one step 
toward building it.

## What it Does

This project develops a vision-based screening tool for cognitive impairment by 
training deep learning models on Clock Drawing Test (CDT) images drawn from the 
National Health and Aging Trends Study (NHATS, Rounds 1–12 with labels, Rounds 
1–14 downloaded). The CDT has been widely used by clinicians as a standard 
bedside screen for Alzheimer's disease and related dementias, and our system 
learns to replicate that same judgment across roughly 59,000 hand-drawn clock 
images, mapping each drawing to a 6-level NHATS impairment score.

To find the most effective approach, we fine-tune three well-established pretrained 
architectures: VGG16, EfficientNet-B0, and ViT-B/16. We evaluate all three across 
seven performance metrics and run a controlled 2×2 ablation study that tests whether 
freezing the backbone (versus allowing full fine-tuning) and exploring how much data 
augmentation actually affects the results. Finally, to understand what each model 
actually learns to look at, we apply Grad-CAM for the CNN-based models and patch 
occlusion for the Vision Transformer.

# What we reproduce from Hu et al. (2026)
| Aspect | Hu et al. | This Work |
|---|---|---|
| Dataset | NHATS Rounds 1–9, 24,991 images | NHATS Rounds 1–14, 59,417 images (2.4× more) |
| Task | Binary + ordinal (0–5) CDT scoring | Ordinal (0–5) only (strictly harder) |
| Architectures tested | ResNet101, EfficientNet, ViT | VGG16, EfficientNet-B0, ViT-B/16 |
| Primary metric | Weighted kappa | Quadratic kappa (mathematically equivalent) |
| Split methodology | Not specified | Participant-disjoint (GroupShuffleSplit on SPID) |

* Architecture substitution: Hu et al. used ResNet101 as their CNN baseline. In this
work, we use VGG16 instead, which is a comparable ImageNet-pretrained CNN from
roughly the same generation (both VGG16 and ResNet101 were introduced between 2014
and 2015 and achieve around 91 to 92% top-5 accuracy on ImageNet). This substitution
does not affect the paper's finding that ViT outperforms CNN-based architectures,
since we retain both EfficientNet and ViT from the original comparison. VGG16 serves
as an additional CNN reference point and provides an estimate of CNN performance on
this task.

Both VGG16 and ResNet101 are fully supervised, ImageNet-pretrained CNN backbones used 
strictly through fine-tuning. The paper's main conclusion that ViT performs at least 
as well as CNNs on CDT scoring remains true regardless of which specific CNN fills 
the baseline role. Replacing one with another is a standard practice in ablation-style 
comparisons and does not weaken the validity of the results.

## Quick Start

See `SETUP.md` for full install details. TL;DR:

```bash
# 1. clone and install
git clone https://github.com/wiambenadder/cdt-alzheimer-screening
cd cdt-alzheimer-screening
pip install -r requirements.txt

# 2. place your NHATS data
# images → data/nhats_raw/<round_XX>/<participant_id>.tif
# labels → data/labels.csv  (columns: participant_id, cdt_score 0–5, image_path)

# 3. run experiments (Colab: open numbered notebooks in order)
jupyter nbconvert --to notebook --execute notebooks/02_baseline.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_finetuning.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_ablation.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_error_analysis.ipynb
```

Jupyter-first alternative: open `notebooks/01_data_exploration.ipynb` and
proceed through the numbered notebooks in order.

## Video Links

- **Demo video (3-5 min, non-technical pitch):** (https://drive.google.com/file/d/1-TJ8hCrk15OeJIB3uw3S0xI8H1pAIgNL/view?usp=drive_link)
- **Technical walkthrough (5-10 min, for ML engineers):** [INSERT LINK]

Both are also in the `videos/` directory.

## Results

### Architecture comparison on the held-out test set

All results use a participant-disjoint split (70/15/15 by participant ID).  
59,417 labeled clock drawings across Rounds 1–14. Class imbalance: 40.8:1 (class 5 vs. class 0).

| Model | Accuracy | Macro-F1 | Quadratic κ | Macro-AUC | ms/img |
|---|---|---|---|---|---|
| Majority-class baseline | 0.370 | 0.090 | 0.000 | — | — |
| VGG16 (frozen backbone) | 0.454 | 0.416 | 0.637 | 0.800 | — |
| VGG16 fine-tuned | 0.616 | 0.584 | 0.781 | 0.903 | — |
| EfficientNet-B0 fine-tuned | 0.590 | 0.549 | 0.747 | 0.893 | — |
| ViT-B/16 fine-tuned | 0.672 | 0.648 | 0.808 | 0.925 | 2.3 |

### Comparison with Hu et al. (2026)

| Model | This Work (quadratic κ) | Hu et al. (weighted κ) | Notes |
|---|---|---|---|
| ResNet101 | — (not run; VGG16 used instead) | 0.56 | 24,991 imgs, Rounds 1–9 |
| EfficientNet | 0.747 | 0.73 | We outperform by +0.017κ |
| ViT (best) | **0.812** (unfrozen + aug) | 0.81 | We match and slightly exceed |
| VGG16 ft (our extra CNN baseline) | 0.781 | not in paper | 59,417 imgs, Rounds 1–14 |

> **Kappa equivalence:** Both "weighted kappa" and "quadratic kappa" apply
quadratic weights to the confusion matrix. They are the same metric under
different names and numbers are directly comparable.
>
> **Reproduction result:** Our ViT-B/16 fine-tuned model reaches quadratic κ = 0.808
on the full test set (8,938 clocks). The best configuration in our ablation study
(unfrozen backbone + augmentation) reaches κ = 0.812, slightly exceeding the paper's
0.81. Both results use a participant-disjoint split, which is a stricter evaluation
than what the paper specifies.

### Ablation study (ViT-B/16, 2x2 design)

| Backbone | Augmentation | Accuracy | Macro-F1 | Quadratic κ | Macro-AUC |
|---|---|---|---|---|---|
| Frozen | Off | 0.502 | 0.494 | 0.667 | 0.849 |
| Frozen | On | 0.470 | 0.472 | 0.640 | 0.846 |
| Unfrozen | Off | 0.646 | 0.623 | 0.799 | 0.916 |
| **Unfrozen** | **On** | **0.676** | **0.656** | **0.812** | **0.926** |

The key finding is that fine-tuning is the dominant factor, adding 0.13 kappa 
over a frozen backbone. Augmentation helps when the backbone is unfrozen, 
gaining 0.013 kappa, but slightly hurts performance when the backbone is frozen, 
losing 0.027 kappa. This shows an interaction effect where augmentation only 
benefits the model when the backbone is free to adapt to it.

### Qualitative Results

| File | Description |
|---|---|
| `results/vit_confusion_matrix_normalized.png` | Normalized per-class confusion matrix |
| `results/vit_attribution_maps.png` | Patch occlusion attribution maps for all 6 CDT classes |
| `results/errors/` | 12 most confidently misclassified clock images |
| `results/all_results.csv` | All experiment results sorted by quadratic kappa |

### Data Challenges Addressed

| Challenge | Scale | Approach |
|---|---|---|
| Class imbalance | 40.8:1 ratio (class 5 vs. class 0) | Class-weighted cross-entropy + WeightedRandomSampler |
| Tiny clock signal | ~10% of each scanned page | SmartCropClock: edge strip → ink-density projection → square crop |
| Participant leakage | Same SPID in 14 rounds | GroupShuffleSplit on participant_id — zero cross-split participant overlap |
| TIFF heterogeneity | Varied scanner formats | Convert to RGB on load; normalize to ImageNet µ/σ |

### Inference Speed

ViT-B/16 runs at **2.3 ms/image (436 images/sec)** on an L4 GPU, measured on
the 8,938-image test set (NB05). A single clock drawing takes under 1 second
to score, which is fast enough for clinical deployment.

## Repository Structure

```
cdt-alzheimer-screening/
  src/
    config.py          hyperparameters, paths, experiment presets
    data.py            NHATS dataset, participant-disjoint splits, class weights
    augmentation.py    SmartCropClock + 5 augmentation techniques + eval transform
    models.py          VGG16 / EfficientNet-B0 / ViT-B/16 
    train.py           AMP training loop, cosine LR scheduler, early stopping
    evaluate.py        metrics, confusion matrix, error cases, timing
    gradcam.py         Grad-CAM (CNN) and patch occlusion maps (ViT)
    utils.py           seeding, checkpoint I/O, device selection
    build_labels.py    to build the labels 
    verify.py           verification
  data/                gitignored; see data/README.md for layout + NHATS access
  models/              gitignored checkpoints
  notebooks/
    01_data_exploration.ipynb
    02_baseline.ipynb
    03_finetuning.ipynb
    04_ablation.ipynb
    05_error_analysis.ipynb
  videos/              demo + technical walkthrough
  docs/
    rubric_mapping.md
    results/           comparison.csv, confusion matrices, figures
  requirements.txt
  environment.yml
  SETUP.md
  ATTRIBUTION.md
```

## Individual Contributions

Solo project. All work, including data preparation, model training, evaluation, 
interpretability, documentation, and videos, was done by Wiam Benadder.

## References

Hu, M., Qin, T., Gonzalez, R., Freedman, V. A., Zahodne, L. B., Melipillán, E. R.,
& Murphey, Y. L. (2026). A novel vision transformer model produces clock drawing test
scores as accurate as expert human coders. Scientific Reports, 16, 4032.
https://doi.org/10.1038/s41598-025-34064-6

National Health and Aging Trends Study (NHATS). Public-use files, Rounds 1–14.
https://nhats.org/researcher/data-access/public-use-files

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
... & Houlsby, N. (2021). An image is worth 16×16 words: Transformers for image
recognition at scale. ICLR 2021. https://arxiv.org/abs/2010.11929

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale
image recognition. ICLR 2015. https://arxiv.org/abs/1409.1556

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional
neural networks. ICML 2019. https://arxiv.org/abs/1905.11946

Morocco World News. (2024). Morocco's illiteracy rate stands at 24.8% in 2024.
https://www.moroccoworldnews.com/2024/12/166863/moroccos-illiteracy-rate-stands-at-24-8-in-2024/

Naji, Y., et al. (2022). Africa's brain specialist shortfall 'risking lives'. SciDev.Net.
https://www.scidev.net/sub-saharan-africa/news/africas-brain-specialist-shortfall-risking-lives/

Charway-Felli, A. (2023). President of African Academy of Neurology calls for urgent action.
World Federation of Neurology.
https://wfneurology.org/activities/news-events/archived-news/2023-10-19-wcn-2

MeaTechWatch. (2025). Morocco's mobile and internet subscriptions cross 100% penetration.
https://meatechwatch.com/2025/09/01/moroccos-mobile-and-internet-subscriptions-cross-100-penetration-in-2025/
