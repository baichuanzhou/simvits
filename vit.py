""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

This code is under MIT licence.
"""


import torch
import torch.nn as nn
from einops import rearrange, repeat


class Attention(nn.Module):
    """
    The Attention module described in the Vision Transformer paper.
    """
    def __init__(self):
        super().__init__()
