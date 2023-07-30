import torch
import torch.nn as nn
from functools import partial
from configuration_resnet import ResNetConfig
from typing import Optional


class ResNetConvLayer(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
            activation: Optional[nn.Module] = nn.ReLU()
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state) if self.activation is not None else hidden_state
        return hidden_state


class ResNetEmbeddings(nn.Module):
    """
    The first convolution layer, as described in paper
    """

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.embedder = ResNetConvLayer(
            config.num_channels, config.embedding_size,
            kernel_size=config.embedding_kernel_size, stride=config.embedding_stride
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) \
            if config.embedding_pooling_with_downsample else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding = self.embedder(pixel_values)
        return self.pooler(embedding)


class ResNetResidual(nn.Module):
    """
    Residual connection, which is used to project features to correct sizes.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetBasicLayer(nn.Module):
    """
    The original resnet paper describes two types of block layer: BasicLayer and BottleNetLayer.
    The BasicLayer composes of two 3x3 ConvLayer.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        should_apply_residual = in_channels != out_channels or stride != 1
        self.basic_layer = nn.Sequential(
            ResNetConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride),
            ResNetConvLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )

        self.residual_connection = ResNetResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        ) if should_apply_residual else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.residual_connection(hidden_state) + self.basic_layer(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetBottleNetLayer(nn.Module):
    """
    BottleNeckLayer is described in the original ResNet paper. It is composed of three ConvLayers, first of which is a
    1x1 conv layer, used to decrease feature channels, followed by a 3x3 ConvLayers and a 1x1 ConvLayer to recover the
    input feature channels. Feature channels form the information bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 4):
        super().__init__()
        bottleneck_channels = out_channels // reduction
        should_apply_residual = in_channels != out_channels or stride != 1
        self.bottleneck_layer = nn.Sequential(
            ResNetConvLayer(in_channels=in_channels, out_channels=bottleneck_channels,
                            kernel_size=1),
            ResNetConvLayer(in_channels=bottleneck_channels, out_channels=bottleneck_channels,
                            stride=stride, kernel_size=3),
            ResNetConvLayer(in_channels=bottleneck_channels, out_channels=out_channels,
                            kernel_size=1, activation=None)
        )
        self.residual_connection = ResNetResidual(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        ) if should_apply_residual else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.residual_connection(hidden_state) + self.bottleneck_layer(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetStage(nn.Module):
    """
    This setup is from Huggingface:
    https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/resnet/modeling_resnet.py#L180
    """

    def __init__(
            self,
            config: ResNetConfig,
            in_channels: int,
            out_channels: int,
            depth: int = 2,
            stride: int = 2
    ):
        super().__init__()
        layer = partial(ResNetBottleNetLayer, reduction=config.reduction) \
            if config.layer_type == 'bottleneck' else ResNetBasicLayer

        # Downsampling is performed in the first layer of a ResNetStage layer. This is defined in the original paper.
        self.layers = nn.Sequential(
            layer(in_channels=in_channels, out_channels=out_channels, stride=stride),
            *[layer(in_channels=out_channels, out_channels=out_channels) for _ in range(depth - 1)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0]
            )
        )

        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(
                ResNetStage(config=config, in_channels=in_channels,
                            out_channels=out_channels, depth=depth,
                            stride=2 if not config.downsample_after_stage else 1)
            )
            if config.downsample_after_stage:
                self.stages.append(
                    ResNetConvLayer(out_channels, out_channels, stride=2)
                )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for staged_module in self.stages:
            hidden_state = staged_module(hidden_state)
        return hidden_state


class ResNetModel(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embedding = self.embedder(pixel_values)
        hidden_states = self.encoder(embedding)
        hidden_states = self.pooler(hidden_states)
        return hidden_states


class ResNetForImageClassification(nn.Module):
    def __init__(self, config: ResNetConfig, num_classes: int = 1000):
        super().__init__()
        self.model = ResNetModel(config)
        self.linear = nn.Linear(config.hidden_sizes[-1], num_classes)

    def forward(self, pixel_values):
        hidden_state = self.model(pixel_values).flatten(1)
        logits = self.linear(hidden_state)
        return logits
