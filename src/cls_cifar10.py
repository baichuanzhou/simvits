from vision_transformers import VisionTransformer, Trainer, TrainingArguments, compute_metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Tuple
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import sampler
import warmup_scheduler

if __name__ == '__main__':
    model = VisionTransformer(
        num_classes=10,
        image_size=32,
        patch_size=4,
        in_channels=3,
        ffn_dim=384,
        depth=6,
        n_head=12,
        dropout=0.1,
        embed_dim=384,
        pool="mean",
        patch_proj="conv"
    )
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

    training_args = TrainingArguments(
        output_dir="output/3",
        do_train=True,
        do_eval=True,
        do_predict=True,
        optim="adamw",
        logging_steps=100,
        learning_rate=1e-3,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_steps=3000,
        epoch=100
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
    trainer.train(resume_from_checkpoint=False, overwrite_output_dir=False)
