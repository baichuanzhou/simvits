from vision_transformers import (
    Trainer, TrainingArguments, compute_metrics, HfArgumentParser
)
from vision_transformers.data import RandomCropPaste, MixUp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torch.utils.data import sampler
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelArguments:
    num_classes: int = field(default=100, metadata={"help": "number of classes for ViT's MLP head"})
    image_size: int = field(default=256, metadata={"help": "size of input image"})
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
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)

    model = VisionTransformer(
        num_classes=model_args.num_classes,
        embed_dim=model_args.embed_dim,
        image_size=model_args.image_size,
        patch_size=model_args.patch_size,
        in_channels=model_args.in_channels,
        ffn_dim=model_args.ffn_dim,
        depth=model_args.depth,
        n_head=model_args.n_head,
        dropout=model_args.dropout,
        pool=model_args.pool,
        patch_proj=model_args.patch_proj,
        pos_embed=model_args.pos_embed,
        fixed_patch_proj=model_args.fixed_patch_proj
    )

    train_transforms = T.Compose([
        T.RandomResizedCrop(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root="../../datasets/imagenet100", transform=train_transforms)
    val_dataset = ImageFolder(root="../../datasets/imagenet100", transform=val_transforms)

    num_images = len(train_dataset)
    random_indices = np.random.randint(low=0, high=num_images, size=num_images)
    train_indices = random_indices[:int(0.95 * num_images)]
    val_indices = random_indices[int(0.95 * num_images):]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        sampler=sampler.SubsetRandomSampler(train_indices)
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=256,
        sampler=sampler.SubsetRandomSampler(val_indices)
    )
    trainer = Trainer(
        model=model,
        training_args=training_args,
        train_loader=train_loader,
        eval_loader=val_loader,
        criterion=F.cross_entropy,
        compute_metric=compute_metrics
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint,
                  overwrite_output_dir=training_args.overwrite_output_dir)


if __name__ == '__main__':
    main()
