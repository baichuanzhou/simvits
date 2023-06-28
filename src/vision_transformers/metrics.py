import torch
import torch.nn as nn
from typing import Any, Tuple


def compute_metrics(model: nn.Module, sample: Tuple[Any, Any], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        X, y = sample
        X, y = X.to(device=device), y.to(device=device)
        predictions = model(X)
        return predictions, y

