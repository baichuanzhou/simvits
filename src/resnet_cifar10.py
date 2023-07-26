from vision_transformers import (
    Trainer, TrainingArguments, compute_metrics
)
from vision_transformers import ResNetConfig, ResNetForImageClassification
from vision_transformers.data import CutMix, MixUp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from typing import Any, Tuple
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import sampler
from dataclasses import dataclass, field
import numpy as np


def main():
    model = ResNetForImageClassification(
        ResNetConfig(
            embedding_size=16, hidden_sizes=[16, 32, 64], depths=[9, 9, 9], embedding_stride=1,
            embedding_kernel_size=3, embedding_pooling_with_downsample=False,
            downsample_in_first_stage=False, layer_type='basic_layer'
        ), num_classes=10
    )
    NUM_TRAIN = 45000
    NUM = 50000

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.CenterCrop(32),
        T.RandomHorizontalFlip(),
        # T.RandAugment(num_ops=2, magnitude=1),
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
        output_dir='output/cifar10/resnet/resnet56',
        logging_dir='runs/ResNetForImageClassification_ResNet56',
        do_train=True,
        do_eval=True,
        do_predict=True,
        optim='sgd',
        max_steps=64000,
        learning_rate=1e-1,
        weight_decay=0.0001,
        lr_scheduler_type='constant',
        logging_steps=100,
        resume_from_checkpoint=True,
        overwrite_output_dir=False,
        optim_args="momentum = 0.9",
    )
    cutmix = CutMix(32, beta=1.)
    mixup = MixUp(alpha=1.)

    class ResNetTrainer(Trainer):
        use_cutmix = False
        use_mixup = False

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

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

    def lr_schedule(step: int):
        if 16000 <= step < 32000:
            return 0.1
        elif 32000 <= step < 48000:
            return 0.01
        elif step >= 48000:
            return 0.001
        return 1
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    trainer = ResNetTrainer(
        model=model,
        training_args=training_args,
        train_loader=train_loader,
        eval_loader=val_loader,
        test_loader=test_loader,
        criterion=F.cross_entropy,
        compute_metric=compute_metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint,
                  overwrite_output_dir=training_args.overwrite_output_dir)


if __name__ == '__main__':
    main()

