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
import torch.nn.functional as F
from .configuration_vit import ViTConfig
from ..layers import QuickGLEU


def make_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = make_pair(config.image_size)
        self.patch_size = make_pair(config.patch_size)

        self.num_channels, self.hidden_size = config.num_channels, config.hidden_size
        self.num_patches = self.image_size[0] * self.image_size[1] // self.patch_size[0] * self.patch_size[1]
        if self.image_size[0] % self.patch_size[0] != 0 or self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError(f"image size and patch size doesn't match: {self.image_size}, {self.patch_size}")

        self.proj = nn.Conv2d(in_channels=self.num_channels, stride=self.patch_size,
                              kernel_size=self.patch_size, out_channels=self.hidden_size)
        # Rearrange('b d ph pw -> b (ph pw) d'),
        self.norm_layer = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.proj.weight.requires_grad = not config.fix_patch_embedding
        self.proj.bias.requires_grad = not config.fix_patch_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        new_x_size = (x.size(0),) + (x.size(1) * x.size(2),) + (x.size(3),)
        x = x.view(*new_x_size)
        out = self.dense(self.norm_layer(x))
        return out


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * config.hidden_size ** -0.5)
        self.patch_embeddings = ViTPatchEmbeddings(config)
        self.pos_embeddings = nn.Parameter(
            self.build_cos_position_embedding(config.hidden_size), requires_grad=False
        )

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
        x = self.patch_embeddings(x)       # B x N x D
        x = torch.cat(
            [self.cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=-2
        )   # B x (N + 1) x D
        x = x + self.pos_embeddings[:x.size(1), :]      # B x (N + 1) x D
        return x


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

        # self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)

        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.num_heads = config.num_heads
        self.d_model = config.hidden_size
        self.attn_dropout = config.attn_dropout
        self.out_dropout = config.out_dropout if config.out_dropout is not None else config.attn_dropout
        assert config.hidden_size % config.num_heads == 0, f"num_heads: {config.num_heads}, " \
                                                           f"must be divisible by d_model: " \
                                                           f"{config.hidden_size}"

    def attn_transpose(self, x: torch.Tensor) -> torch.Tensor:
        transpose_size = x.size()[:-1] + (self.num_heads, self.d_model // self.num_heads)
        x = x.view(*transpose_size).permute(0, 2, 1, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)      # q: B x N x (H x D)
        all_head_size = q.size(-1)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.attn_transpose(q)      # q: B x H x N x D
        k = self.attn_transpose(k)      # k: B x H x N x D
        v = self.attn_transpose(v)      # v: B x H x N x D
        # (q, k, v) = self.qkv_proj(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))  # B x H x N x D
        scale = q.size(-1) ** 0.5

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / scale     # B x H x N x N
        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = F.dropout(attn_score, self.attn_dropout)       # This is weird, but it is from the original paper

        v = torch.matmul(attn_score, v)     # B x H x N x D
        # v = rearrange(v, 'b h n d -> b n (h d)', h=self.num_heads)
        v = v.permute(0, 2, 1, 3).contiguous()
        new_v_size = v.size()[:-2] + (all_head_size, )
        v = v.view(*new_v_size)
        out = self.c_proj(v)
        out = F.dropout(out, self.out_dropout)
        return out


class ViTFeedForward(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        intermediate_size = config.intermediate_size or config.hidden_size * 4
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            QuickGLEU(),
            nn.Linear(intermediate_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feedforward(x)


class ViTGenerator(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            QuickGLEU(),
            nn.Dropout(config.out_dropout, inplace=True),
            nn.Linear(config.intermediate_size, config.num_classes),
            nn.Dropout(config.out_dropout, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = ViTSelfAttention(config)
        self.mlp = ViTFeedForward(config)
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_pre(x))
        x = x + self.mlp(self.ln_post(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.encoder = nn.Sequential(*[
            ViTLayer(config)
            for _ in range(config.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder = ViTEncoder(config)

        self.post_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.pooling = ViTPooler(config) if add_pooling_layer else None

        self.generator = ViTGenerator(config) if config.num_classes is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, image_height, image_width = x.size()
        if num_channels != self.config.num_channels:
            raise ValueError(f"Image's channels do not match. Yours: {num_channels}, "
                             f"config's: {self.config.num_channels}")
        if (image_height, image_width) != make_pair(self.config.image_size):
            raise ValueError(f"Image size does not match. Yours: {(image_height, image_width)}, "
                             f"config's: {make_pair(self.config.image_size)}")
        x = self.embeddings(x)
        x = self.pre_norm(x)
        x = self.encoder(x)
        x = self.post_norm(x)
        x = self.pooling(x) if self.pooling is not None else x[:, 0]
        x = self.generator(x) if self.generator is not None else x
        return x


def vit_tiny():
    tiny_config = ViTConfig(num_layers=12, hidden_size=192, intermediate_size=768, num_heads=3, num_classes=1000)
    return ViTModel(tiny_config)

