import torch
import torch.nn as nn
from .configuration_resnext import ResNextConfig
from typing import Optional


class ResNextConvLayer(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int = 3, stride: int = 1, groups: int = 1,
            activation: Optional[nn.Module] = nn.ReLU()
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=groups, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states) if self.activation is not None else hidden_states
        return hidden_states


class ResNextEmbeddings(nn.Module):
    def __init__(self, config: ResNextConfig):
        super().__init__()
        self.embedder = ResNextConvLayer(
            in_channels=config.num_channels, out_channels=config.embedding_size,
            kernel_size=config.embedding_kernel_size, stride=config.embedding_stride
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) \
            if config.embedding_pooling_with_downsample else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedder(pixel_values)
        hidden_states = self.pooler(hidden_states)
        return hidden_states


class ResNextResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        return hidden_states


class ResNextBottleNeckLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 bottleneck_width: int, cardinality: int, reduction: int = 4):
        super().__init__()
        bottleneck_channels = int((out_channels // reduction) * (bottleneck_width / 64)) * cardinality
        self.bottleneck_layer = nn.Sequential(
            ResNextConvLayer(in_channels, bottleneck_channels, kernel_size=1),
            ResNextConvLayer(bottleneck_channels, bottleneck_channels,
                             kernel_size=3, stride=stride, groups=cardinality),
            ResNextConvLayer(bottleneck_channels, out_channels, kernel_size=1, activation=None)
        )
        should_apply_residual = in_channels != out_channels or stride != 1
        self.residual_connection = ResNextResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        ) if should_apply_residual else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.bottleneck_layer(hidden_states) + self.residual_connection(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class ResNextStage(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            cardinality: int,
            bottleneck_width: int,
            depth: int = 3,
            stride: int = 2,
            reduction: int = 4
    ):
        super().__init__()
        self.layers = nn.Sequential(
            ResNextBottleNeckLayer(in_channels, out_channels, stride, bottleneck_width, cardinality, reduction),
            *[ResNextBottleNeckLayer(
                out_channels, out_channels, 1, bottleneck_width, cardinality, reduction
            ) for _ in range(depth - 1)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layers(hidden_states)
        return hidden_states


class ResNextEncoder(nn.Module):
    def __init__(self, config: ResNextConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        bottleneck_width = config.bottleneck_width
        self.stages.append(
            ResNextStage(
                in_channels=config.embedding_size,
                out_channels=config.hidden_sizes[0],
                cardinality=config.cardinality,
                bottleneck_width=bottleneck_width,
                depth=config.depths[0],
                stride=2 if config.downsample_in_first_stage else 1,
                reduction=config.reduction
            )
        )

        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(
                ResNextStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    cardinality=config.cardinality,
                    bottleneck_width=bottleneck_width,
                    depth=depth,
                    stride=2 if not config.downsample_after_stage else 1,
                    reduction=config.reduction
                )
            )
            if config.downsample_after_stage:
                self.stages.append(
                    ResNextConvLayer(out_channels, out_channels, stride=2)
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for staged_module in self.stages:
            hidden_states = staged_module(hidden_states)
        return hidden_states


class ResNextModel(nn.Module):
    def __init__(self, config: ResNextConfig):
        super().__init__()
        self.config = config
        self.embedder = ResNextEmbeddings(config)
        self.encoder = ResNextEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, ResNextBottleNeckLayer) and m.bottleneck_layer[2].normalization is not None:
                nn.init.constant_(m.bottleneck_layer[2].normalization.weight, 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedder(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.pooler(hidden_states)
        return hidden_states


class ResNextForImageClassification(nn.Module):
    def __init__(self, config: ResNextConfig):
        super().__init__()
        self.config = config
        self.model = ResNextModel(config)
        self.linear = nn.Linear(config.hidden_sizes[-1], config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(pixel_values).flatten(1)
        logits = self.linear(hidden_states)
        return logits
