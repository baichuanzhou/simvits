from vision_transformers import VisionTransformer, Trainer, TrainingArguments
import torch
import torch.optim as optim
import torch.nn.functional as F

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
        embed_dim=384
    )
    NUM_TRAIN = 45000
    NUM = 50000

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,
                                                                eta_min=1e-5)
    lr_scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1,
                                                           total_epoch=5, after_scheduler=base_scheduler)

    torch.set_float32_matmul_precision('high')
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
        output_dir="../temp_output",
        do_train=True,
        optim="adamw",
        logging_steps=100,
        learning_rate=1e-3,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        epoch=200
    )
    trainer = Trainer(
        model=model,
        training_args=training_args,
        train_loader=train_loader,
        criterion=F.cross_entropy,
        # optimizer=optimizer,
        # lr_scheduler=lr_scheduler
    )
    trainer.train()
