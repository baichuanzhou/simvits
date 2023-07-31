from dataclasses import dataclass
from typing import Optional


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_heads: int = 12
    qkv_bias: bool = True
    intermediate_size: int = 2048
    num_layers: int = 8
    attn_dropout: float = 0.1
    out_dropout: Optional[float] = 0.1
    num_classes: Optional[int] = None
    fix_patch_embedding: bool = True
    initializer_range: float = 0.02
