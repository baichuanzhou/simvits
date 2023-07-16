"""
Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

This code is under MIT licence.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from collections import OrderedDict
from typing import Union, Tuple
from einops.layers.torch import Rearrange
from torch.autograd import Variable

__all__ = ['VisionTransformer']


def make_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SelfAttention(nn.Module):
    def __init__(self,
                 d_model: int, n_head: int, output_dim: int = None,
                 attn_dropout: float = 0.1, qkv_bias: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        self.c_proj = nn.Linear(d_model, output_dim)

        self.n_head = n_head
        self.d_model = d_model
        self.attn_dropout = attn_dropout
        assert d_model % n_head == 0, f"n_head: {n_head}, must be divisible by d_model: {d_model}"

    def _self_attn_init(self):
        pass

    def attn_transpose(self, x):
        transpose_size = x.size()[:-1] + (self.n_head, self.d_model // self.n_head)
        x = x.reshape(transpose_size).permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        q = self.q_proj(x)      # q: B x N x (H x D)
        original_size = q.size()
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = q.size(1)

        q = self.attn_transpose(q)      # q: B x H x N x D
        k = self.attn_transpose(k)      # k: B x H x N x D
        v = self.attn_transpose(v)      # v: B x H x N x D

        attn_score = torch.matmul(q, k.transpose(-1, -2)) * scale ** -0.5       # B x H x N x N
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = F.dropout(attn_score, self.attn_dropout)       # This is weird, but it is from the original paper

        v = torch.matmul(attn_score, v)
        v = v.resize(original_size)
        out = self.c_proj(v)
        out = F.dropout(out, self.attn_dropout)
        return out


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x).chunk(3, dim=-1)     # q, k, v is of shape B x N x (HD) (H refers to number of heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)    # B x H x N x D
        scale = q.size(-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (scale ** -0.5)   # B x H x N x N

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.dropout)

        out = torch.matmul(attn, v)     # B x H x N x D
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.n_head)     # B x N x (HD)
        out = F.dropout(self.c_proj(out), self.dropout)
        return out


class QuickGLEU(nn.Module):
    """
    This module is originated from OpenAI's CLIP repo:
        - https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim: int,
                 hidden_dim: int,
                 dropout: float = 0.,
                 out_dim: int = None):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            QuickGLEU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn_dim: int, depth: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, n_head, ffn_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class StandardPatchProjection(nn.Module):
    def __init__(self, patch_height: int, patch_width: int, d_model: int, out_dim: int):
        super().__init__()
        self.patch_proj = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)',
                      ph=patch_height, pw=patch_width),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x) -> torch.Tensor:
        return self.patch_proj(x)


class ConvPatchProjection(nn.Module):
    def __init__(self, patch_size: int, d_model: int, in_channels: int, out_dim: int):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=d_model,
                      stride=patch_size, kernel_size=patch_size),
            Rearrange('b d ph pw -> b (ph pw) d'),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(x)
        return x


def cos_position_embedding(d_model: int, max_len: int = 1000):
    den = torch.exp(-torch.arange(0, d_model, 2) * math.log(1000) / d_model)
    pos = torch.arange(0, max_len).reshape(max_len, 1)
    pos_embedding = torch.zeros((max_len, d_model))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    return pos_embedding


class VisionTransformer(nn.Module):

    proj_choice = ['standard', 'conv']
    pool_choice = ['cls', 'mean']
    pos_embed_choice = ['random', 'cos']

    def __init__(self, image_size: Union[int, Tuple], patch_size: Union[int, Tuple], in_channels: int,
                 embed_dim: Union[int, None], n_head: int, ffn_dim: int, depth: int, dropout: float, num_classes: int,
                 pool: str = 'cls', pos_embed: str = 'cos',
                 patch_proj: str = 'standard', fixed_patch_proj: bool = False):
        super().__init__()
        self.image_height, self.image_width = make_pair(image_size)
        self.patch_height, self.patch_width = make_pair(patch_size)
        self.n_head = n_head

        d_model = self.patch_height * self.patch_width * in_channels
        num_patches = self.image_height // self.patch_height * self.image_width // self.patch_width
        if embed_dim is None:
            embed_dim = d_model

        assert self.image_height % self.patch_height == 0, \
            f"Image height {self.image_height} is not divisible by patch height {self.patch_height} "

        assert self.image_width % self.patch_width == 0, \
            f"Image width {self.image_height} is not divisible by patch width {self.patch_height} "

        assert patch_proj in self.proj_choice, f"patch_proj must be in {self.proj_choice}"
        assert pos_embed in self.pos_embed_choice, f"pos_embed must be in {self.pos_embed_choice}"
        assert pool in self.pool_choice, f"pool must be in {self.pool_choice}"
        self.pool = pool

        scale = embed_dim ** -0.5
        self.cls = nn.Parameter(scale * torch.randn(embed_dim))

        self.fixed_patch_proj = fixed_patch_proj
        if patch_proj == 'standard':    # Patchify the images like the original ViT paper.
            self.patch_proj = StandardPatchProjection(self.patch_height, self.patch_width, d_model, embed_dim)
        else:   # Patchify the images by convolution augmentation.
            self.patch_proj = ConvPatchProjection(patch_size, d_model, in_channels, embed_dim)

        if pos_embed == 'random':
            self.pos_embed = nn.Parameter(scale * torch.randn(num_patches + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(cos_position_embedding(embed_dim))
        self.ln_pre = nn.LayerNorm(embed_dim)

        self.transformer = Transformer(d_model=embed_dim, n_head=n_head, ffn_dim=ffn_dim, dropout=dropout, depth=depth)
        self.ln_post = nn.LayerNorm(embed_dim)

        self.mlp_head = FeedForward(dim=embed_dim, hidden_dim=ffn_dim, out_dim=num_classes)

    def forward(self, x) -> torch.Tensor:
        if self.fixed_patch_proj:
            with torch.no_grad():
                x = self.patch_proj(x)  # B x N x D
        else:
            x = self.patch_proj(x)  # B x N x D
        x = torch.cat([self.cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
                      dim=-2)    # B x (N + 1) x D
        x = x + Variable(self.pos_embed[:x.size(1), :], requires_grad=False)
        x = self.ln_pre(x)

        x = self.transformer(x)
        x = self.ln_post(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)



