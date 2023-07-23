import torch
import torch.nn as nn


class QuickGLEU(nn.Module):
    """
    This module is originated from OpenAI's CLIP repo:
        - https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)