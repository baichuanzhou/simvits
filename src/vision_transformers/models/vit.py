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
from dataclasses import dataclass
from functools import partial, reduce
from operator import mul
from typing import Optional

__all__ = ['VisionTransformer', 'ViT', 'vit_tiny', 'ViTConfig']


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
            *[ResidualAttentionBlock(d_model, n_head, ffn_dim, dropout) for _ in range(depth)]
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


def cos_position_embedding(d_model: int, max_len: int = 10000):
    den = torch.exp(-torch.arange(0, d_model, 2) * math.log(float(10000)) / d_model)
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


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    d_model: int = 768
    n_head: int = 12
    qkv_bias: bool = True
    feedforward_dim: int = 2048
    num_layers: int = 8
    attn_dropout: float = 0.1
    out_dropout: Optional[float] = 0.1
    num_classes: Optional[int] = None
    fix_patch_embedding: bool = True
    norm_layer = partial(nn.LayerNorm, eps=1e-6)


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = make_pair(config.image_size)
        self.patch_size = make_pair(config.patch_size)

        self.num_channels, self.d_model = config.num_channels, config.d_model
        self.num_patches = self.image_size[0] * self.image_size[1] // self.patch_size[0] * self.patch_size[1]
        if self.image_size[0] % self.patch_size[0] != 0 or self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError(f"image size and patch size doesn't match: {self.image_size}, {self.patch_size}")

        self.proj = nn.Conv2d(in_channels=self.num_channels, stride=self.patch_size,
                              kernel_size=self.patch_size, out_channels=self.d_model)
        # xavier_uniform initialization
        val = math.sqrt(6. / float(reduce(mul, self.patch_size, 1)) + self.d_model)
        nn.init.normal_(self.proj.weight, -val, val)
        nn.init.zeros_(self.proj.bias)

        self.proj.weight.requires_grad = not config.fix_patch_embedding
        self.proj.bias.requires_grad = not config.fix_patch_embedding

        self.linear = nn.Linear(config.d_model, config.d_model)
        self.ln = config.norm_layer(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x).flatten(2).transpose(1, 2)
        out = self.linear(self.ln(out))
        return out


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.pos_embeddings = nn.Parameter(self.build_cos_position_embedding(config.d_model), requires_grad=False)

    @classmethod
    def build_cos_position_embedding(cls, d_model: int, max_len: int = 10000):
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(float(max_len)) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pe_token = torch.zeros([1, d_model], dtype=torch.float32)
        pos_embedding = torch.cat([pe_token, pos_embedding], dim=0)
        return pos_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)

        x = torch.cat([self.cls_token +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=-2)
        x = x + self.pos_embeddings[:x.shape[-2], :]
        return x


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.c_proj = nn.Linear(config.d_model, config.d_model)

        self.n_head = config.n_head
        self.d_model = config.d_model
        self.attn_dropout = config.attn_dropout
        self.out_dropout = config.out_dropout if config.out_dropout is not None else config.attn_dropout
        assert config.d_model % config.n_head == 0, f"n_head: {config.n_head}, must be divisible by d_model: " \
                                                    f"{config.d_model}"

    def attn_transpose(self, x: torch.Tensor) -> torch.Tensor:
        transpose_size = x.size()[:-1] + (self.n_head, self.d_model // self.n_head)
        x = x.reshape(transpose_size).permute(0, 2, 1, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)      # q: B x N x (H x D)
        original_size = q.size()
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.attn_transpose(q)      # q: B x H x N x D
        k = self.attn_transpose(k)      # k: B x H x N x D
        v = self.attn_transpose(v)      # v: B x H x N x D
        scale = q.size(-1)

        attn_score = torch.matmul(q, k.transpose(-1, -2)) * scale ** -0.5       # B x H x N x N
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = F.dropout(attn_score, self.attn_dropout)       # This is weird, but it is from the original paper

        v = torch.matmul(attn_score, v)
        v = v.reshape(original_size)
        out = self.c_proj(v)
        out = F.dropout(out, self.out_dropout)
        return out


class ViTFeedForward(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        feedforward_dim = config.feedforward_dim or config.d_model * 4
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            QuickGLEU(),
            nn.Dropout(config.out_dropout, inplace=True),
            nn.Linear(feedforward_dim, config.d_model),
            nn.Dropout(config.out_dropout, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feedforward(x)


class ViTGenerator(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            QuickGLEU(),
            nn.Dropout(config.out_dropout, inplace=True),
            nn.Linear(config.feedforward_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_head(x)


class ViTEncoderBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.ln_pre = config.norm_layer(config.d_model)
        self.attn = ViTSelfAttention(config)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.d_model, config.feedforward_dim)),
            ("gleu", QuickGLEU()),
            ("fc_c", nn.Linear(config.feedforward_dim, config.d_model))
        ]))
        self.ln_post = config.norm_layer(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_pre(x))
        x = x + self.mlp(self.ln_post(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(*[
            ViTEncoderBlock(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViT(nn.Module):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.norm_layer = config.norm_layer(config.d_model)
        self.pooling = ViTPooler(config) if add_pooling_layer else None

        self.generator = ViTGenerator(config) if config.num_classes is not None else None

        self._init_param()

    def _init_param(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.embeddings.cls_token, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, image_height, image_width = x.size()
        if num_channels != self.config.num_channels:
            raise ValueError(f"Image's channels do not match. Yours: {num_channels}, "
                             f"config's: {self.config.num_channels}")
        if (image_height, image_width) != make_pair(self.config.image_size):
            raise ValueError(f"Image size does not match. Yours: {(image_height, image_width)}, "
                             f"config's: {make_pair(self.config.image_size)}")

        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.norm_layer(x)
        x = self.pooling(x) if self.pooling is not None else x[:, 0]
        x = self.generator(x) if self.generator is not None else x
        return x


def vit_tiny():
    tiny_config = ViTConfig(num_layers=12, d_model=192, feedforward_dim=768, n_head=3, num_classes=1000)
    return ViT(tiny_config)

