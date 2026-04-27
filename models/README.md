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

These are gitignored because each one is ~100 MB - 1 GB. 

## Checkpoint access for graders

The trained checkpoints are stored on Google Drive and are not distributed in this repo.
To reproduce any result, run the corresponding notebook from the beginning.
Approximate training time on a T4 GPU: VGG16 around 2 hours, EfficientNet-B0 around 2 hours,
ViT-B/16 around 3 hours, and the full ablation (4 runs) around 6 hours.
