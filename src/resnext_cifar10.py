from vision_transformers import (
    Trainer, TrainingArguments, compute_metrics
)
from vision_transformers import ResNextConfig, ResNextForImageClassification
from vision_transformers.data import CutMix, MixUp
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import sampler
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--output_dir", type=str, default="output/cifar10/resnext/resnext29-1x64d")
parser.add_argument("--logging_dir", type=str, default="runs/ResNextForImageClassification_ResNext29-1x64d")
parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 3])
parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[256, 512, 1024])
parser.add_argument("--layer_type", type=str, default="basic_layer")
parser.add_argument("--num_labels", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
parser.add_argument("--overwrite_output_dir", type=bool, default=True)
parser.add_argument("--bottleneck_width", type=int, default=64)
parser.add_argument("--cardinality", type=int, default=1)


def main():
    args = parser.parse_args()
    model = ResNextForImageClassification(
        ResNextConfig(
            embedding_size=64, hidden_sizes=args.hidden_sizes, depths=args.depths, embedding_stride=1,
            embedding_kernel_size=3, embedding_pooling_with_downsample=False,
            downsample_after_stage=False, downsample_in_first_stage=False,
            num_labels=args.num_labels, bottleneck_width=args.bottleneck_width, cardinality=args.cardinality
        )
    )
    NUM_TRAIN = 50000

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
    val_dataset = CIFAR10(root='../../pytorch-cifar10/datasets', train=False,
                          download=False, transform=train_transform)
    test_dataset = CIFAR10(root='../../pytorch-cifar10/datasets', train=False,
                           download=False, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        epoch=300,
        logging_steps=100,
        resume_from_checkpoint=args.resume_from_checkpoint,
        overwrite_output_dir=args.overwrite_output_dir,
    )

    class ResNetTrainer(Trainer):

        def compute_loss(self, sample):
            img, label = sample

            out = self.model(img)
            loss = self.criterion(out, label)
            return loss

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def lr_schedule(step: int):
        # if 16000 <= step < 32000:
        #     return 0.1
        if 52800 <= step < 79200:
            return 0.1
        elif step >= 79200:
            return 0.01
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

