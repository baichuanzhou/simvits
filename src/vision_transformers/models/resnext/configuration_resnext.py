from dataclasses import dataclass, field
from typing import List


@dataclass
class ResNextConfig:
    num_channels: int = 3
    embedding_size: int = 64
    num_labels: int = 1000
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    depths: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    embedding_kernel_size: int = 7
    embedding_stride: int = 2
    cardinality: int = 32
    reduction: int = 4
    bottleneck_width: int = 4
    embedding_pooling_with_downsample: bool = True
    downsample_in_first_stage: bool = False
    downsample_after_stage: bool = False
