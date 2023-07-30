from dataclasses import dataclass, field
from typing import List


@dataclass
class ResNetConfig:
    num_channels: int = 3
    embedding_size: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    layer_type: str = "bottleneck"
    downsample_in_first_stage: bool = True
    downsample_after_stage: bool = False
    embedding_kernel_size: int = 7
    embedding_stride: int = 2
    embedding_pooling_with_downsample: bool = True
    reduction: int = 4
    num_labels: int = 1000
