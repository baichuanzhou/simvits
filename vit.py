import torch
import torch.nn as nn
from einops import rearrange, repeat


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
