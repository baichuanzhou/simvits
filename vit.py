"""
Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

This code is under MIT licence.
"""


import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from collections import OrderedDict
from typing import Union, Tuple
from einops.layers.torch import Rearrange


def make_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    """
    The Attention module described in the Vision Transformer paper.
    """
    def __init__(self, d_model: int, n_head: int, output_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model or output_dim)
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)     # q, k, v is of shape B x N x (HD) (H refers to number of heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)    # B x H x N x D
        H = q.size(0)
        scores = torch.matmul(q, k.transpose(2, 3)) * (H ** -0.5)   # B x H x N x N

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout)

        out = torch.matmul(attn, v)     # B x H x N x D
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.n_head)
        out = F.dropout(self.c_proj(out), self.dropout)
        return out


class QuickGLEU(nn.Module):
    """
    This module is originated from OpenAI's CLIP repo:
        - https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim: int,
                 hidden_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            QuickGLEU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.proj(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, dropout: float):
        super().__init__()

        self.attn = Attention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, ffn_dim)),
            ("gleu", QuickGLEU()),
            ("fc_c", nn.Linear(ffn_dim, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, depth: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.blocks = nn.Sequential(*[ResidualAttentionBlock(d_model, n_head, ffn_dim, dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, image_size: Union[int, Tuple], patch_size: Union[int, Tuple], channel: int,
                 d_model: int, n_head: int, ffn_dim: int, depth: int, dropout: float,
                 pool: str = 'cls', pos_embed: str = 'random'):
        super().__init__()
        self.image_height, self.image_width = make_pair(image_size)
        self.patch_height, self.patch_width = make_pair(patch_size)
        self.n_head = n_head
        self.pool = pool

        d_model = self.patch_height * self.patch_width * channel
        assert self.image_height % self.patch_height == 0, \
            f"Image height {self.image_height} is not divisible by patch height {self.patch_height} "

        assert self.image_width % self.patch_width == 0, \
            f"Image width {self.image_height} is not divisible by patch width {self.patch_height} "

        self.patch_proj = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b n d',
                      h=self.image_height, w=self.image_width, ph=self.patch_height, pw=self.patch_width,
                      d=d_model),

        )