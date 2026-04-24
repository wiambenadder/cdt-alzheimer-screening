# Trained Models

Model checkpoints are written here by the training loop:

```
models/
  vgg16_ft_best.pt
  effb0_ft_best.pt
  vit_b16_ft_best.pt
  vit_b16_frozen_noaug_best.pt
  vit_b16_frozen_aug_best.pt
  vit_b16_unfrozen_noaug_best.pt
  vit_b16_unfrozen_aug_best.pt
```

These are gitignored because each one is ~100 MB - 1 GB. If you want to
submit the trained checkpoints alongside your code (useful for grading),
you have three options:

1. **Git LFS** (recommended): `git lfs track "models/*.pt"` and commit
2. **HuggingFace Hub**: upload weights as a public or unlisted model
3. **Drive / Dropbox**: link in the README

The repo is fully reproducible from code + data alone, so graders without the
checkpoints can retrain by running the notebooks in `notebooks/`.
