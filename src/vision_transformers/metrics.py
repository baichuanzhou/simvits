import torch
import torch.nn as nn
from typing import Any, Tuple


def compute_metrics(model: nn.Module, sample: Tuple[Any, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = sample
    predictions = model(X)
    return predictions

