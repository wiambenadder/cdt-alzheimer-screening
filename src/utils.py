"""Small helpers: seeding, checkpoint I/O, parameter counting."""
import random
import numpy as np
import torch
from pathlib import Path

from .config import SEED


def set_seed(seed: int = SEED):
    """Seed every RNG that PyTorch touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cudnn hurts speed; flip off if runs are too slow
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: Path, device: str = "cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
