import torch
import torch.optim as optim
from models import VisionTransformer
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import sampler
import warmup_scheduler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.vision_transformers.data import CutMix, MixUp


class ViT(pl.LightningModule):
    def __init__(self,
                 in_channels: int = 3,
                 image_size: int = 32,
                 patch_size: int = 4,
                 n_head: int = 12,
                 ffn_dim: int = 1024,
                 depth: int = 7,
                 num_classes: int = 10,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.vit = VisionTransformer(image_size, patch_size, in_channels, ffn_dim, n_head,
                                     ffn_dim, depth, dropout, num_classes, patch_proj='conv',
                                     pool='mean', pos_embed='random')
        self.cutmix = CutMix(32, beta=1.)
        self.mixup = MixUp(alpha=1.)

    def forward(self, x):
        x = self.vit(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        # x, label, rand_label, lambda_ = self.cutmix((x, y))

        y_hat = self.vit(x)
        # loss = F.cross_entropy(y_hat, y) * lambda_ + F.cross_entropy(y_hat, rand_label) * (1. - lambda_)
        loss = F.cross_entropy(y_hat, y)

        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.vit.parameters(), lr=1e-3, weight_decay=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,
                                                                    eta_min=1e-5)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1,
                                                            total_epoch=5, after_scheduler=base_scheduler)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    NUM_TRAIN = 45000
    NUM = 50000

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

    model = ViT(
        num_classes=10,
        image_size=32,
        patch_size=4,
        in_channels=3,
        ffn_dim=384,
        depth=6,
        n_head=12,
        dropout=0.1
    )

    logger = TensorBoardLogger("logs/", name="ViT")
    trainer = pl.Trainer(max_epochs=200, accelerator='gpu', devices=1, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)
