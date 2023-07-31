from vision_transformers import (
    ViTModel, Trainer, TrainingArguments, compute_metrics, ViTConfig,
)
from vision_transformers.data import CutMix, MixUp
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
import numpy as np


def main():
    model = ViTModel(ViTConfig(num_layers=6, hidden_size=384, intermediate_size=384, num_heads=12, num_classes=10,
                               image_size=32, patch_size=4, fix_patch_embedding=False), add_pooling_layer=True)
    NUM_TRAIN = 45000
    NUM = 50000

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=1),
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
        batch_size=512,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False
    )

    training_args = TrainingArguments(
        output_dir='output/8',
        do_train=True,
        do_eval=True,
        do_predict=True,
        optim='adamw',
        epoch=200,
        learning_rate=1e-3,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_steps=1500,
        logging_steps=100,
        resume_from_checkpoint=False,
        overwrite_output_dir=True
    )
    cutmix = CutMix(model.config.image_size, beta=1.)
    mixup = MixUp(alpha=1.)

    class ViTTrainer(Trainer):
        use_cutmix = False
        use_mixup = True

        def compute_loss(self, sample):
            img, label = sample
            if model.training and self.use_cutmix:

                img, label, rand_label, lambda_ = cutmix((img, label))
                out = self.model(img)
                loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            elif model.training and self.use_mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
                out = self.model(img)
                loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                out = self.model(img)
                loss = self.criterion(out, label)
            return loss

    trainer = ViTTrainer(
        model=model,
        training_args=training_args,
        train_loader=train_loader,
        eval_loader=val_loader,
        test_loader=test_loader,
        criterion=F.cross_entropy,
        compute_metric=compute_metrics
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint,
                  overwrite_output_dir=training_args.overwrite_output_dir)


if __name__ == '__main__':
    main()

