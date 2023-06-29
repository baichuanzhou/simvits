from vision_transformers import (
    VisionTransformer, Trainer, TrainingArguments, compute_metrics, HfArgumentParser
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Tuple
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import sampler
from dataclasses import dataclass, field


@dataclass
class ModelArgument:
    num_classes: int = field(default=10, metadata={"help": "number of classes for ViT's MLP head"})
    image_size: int = field(default=32, metadata={"help": "size of input image"})
    patch_size: int = field(default=4, metadata={"help": "patch size"})
    in_channels: int = field(default=3, metadata={"help": "input image color channel"})
    ffn_dim: int = field(default=384, metadata={"help": "MLP head hidden dim"})
    depth: int = field(default=6, metadata={"help": "number of transformer block"})
    n_head: int = field(default=12, metadata={"help": "number of self-attention head"})
    dropout: float = field(default=0.1, metadata={"help": "dropout"})
    embed_dim: int = field(default=384, metadata={"help": "embedding dim for ViT"})
    pool: str = field(default="cls", metadata={"help": "select how to extract embedding every block"})
    patch_proj: str = field(default="standard", metadata={"help": "how to patchify input image"})
    pos_embed: str = field(default="cos", metadata={"help": "positional embedding"})
    fixed_patch_proj: bool = field(default=False, metadata={"help": "whether to freeze patch projection"})


def main():
    parser = HfArgumentParser((ModelArgument, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = VisionTransformer(**model_args)
    NUM_TRAIN = 45000
    NUM = 50000

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    )

    test_transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='../../pytorch-cifar10/datasets', train=True,
                            download=False, transform=train_transform)
    val_dataset = CIFAR10(root='../../pytorch-cifar10/datasets', train=True,
                          download=False, transform=train_transform)
    test_dataset = CIFAR10(root='../../pytorch-cifar10/datasets', train=False,
                           download=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False
    )

    trainer = Trainer(
        model=model,
        training_args=training_args,
        train_loader=train_loader,
        eval_loader=val_loader,
        test_loader=test_loader,
        criterion=F.cross_entropy,
        compute_metric=compute_metrics
    )
    trainer.train(resume_from_checkpoint=True, overwrite_output_dir=False)


if __name__ == '__main__':
    main()
