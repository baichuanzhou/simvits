from dataclasses import dataclass
from typing import List


@dataclass
class ResNetConfig:
    hidden_sizes: List[int]
    depths: List[int]
    num_channels: int = 3
    embedding_size: int = 64
    layer_type: str = "bottleneck"
    downsample_in_first_stage: bool = True
    embedding_kernel_size: int = 7
    embedding_stride: int = 2
    embedding_pooling_with_downsample: bool = True
    reduction: int = 4
