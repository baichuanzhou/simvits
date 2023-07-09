import math
import torch
import torch.nn as nn
import shutil
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Any, Callable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from .trainer_utils import (
    get_last_checkpoint, get_first_checkpoint,
    SchedulerType, OptimizerNames,
    set_seed, enable_full_determinism,
    get_parameter_names, get_model_param_count
)
from .training_args import TrainingArguments
from .trainer_state import TrainerState
from .optimization import get_scheduler
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"
MODEL_STATE_NAME = "model.pth"
OPTIMIZER_STATE_NAME = "optimizer.pth"
LR_SCHEDULER_STATE_NAME = "lr_scheduler.pth"
TRAINING_ARGS_NAME = "training_args.bin"


class Trainer:
    """
    TODO: Write instructions for this class
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: Optional[Union[nn.Module, Callable]] = None,
            training_args: TrainingArguments = None,
            train_loader: Optional[DataLoader] = None,
            eval_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
            compute_metric: Optional[Callable] = None
    ):
        if training_args is None:
            output_dir = "tmp_output_dir"
            training_args = TrainingArguments(output_dir=output_dir)
        self.args = training_args
        # Set the seed before training
        enable_full_determinism(self.args.seed) if self.args.enable_full_deterministic else set_seed(self.args.seed)

        if model is None:
            raise RuntimeError("Trainer must be provided with a model")

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        # For now, we only support one gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.state = TrainerState()
        self.compute_metric = compute_metric

        if self.args.max_steps > 0:
            logger.warning("max_steps is given, overriding num_train_epochs if given")

        if self.args.do_train and train_loader is None:
            raise RuntimeError("In order to do training, a train_loader must be specified")
        self.train_loader = train_loader
        if self.args.do_eval and eval_loader is None:
            raise RuntimeError("In order to do evaluation, an eval_loader must be specified")
        self.eval_loader = eval_loader
        if self.args.do_predict and test_loader is None:
            raise RuntimeError("In order to do test prediction, a test_loader must be specified")
        self.test_loader = test_loader
        self.scaler = GradScaler()  # Set up GradScaler for mixed precision

        if self.args.log_tensorboard:
            if os.path.exists(self.args.logging_dir):
                num_logs = len(os.listdir(self.args.logging_dir))
                if not self.args.resume_from_checkpoint:
                    self.tensorboard_writer = SummaryWriter(
                        log_dir=os.path.join(self.args.logging_dir, self.model._get_name() + f"_{num_logs}")
                    )

                else:
                    # If resume from last checkpoint, log data to the last log dir
                    self.tensorboard_writer = SummaryWriter(
                        log_dir=os.path.join(self.args.logging_dir, self.model._get_name() + f"_{num_logs - 1}")
                    )
            else:
                os.makedirs(self.args.logging_dir)
                self.tensorboard_writer = SummaryWriter(
                    log_dir=os.path.join(self.args.logging_dir, self.model._get_name())
                )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and scheduler.
            Overwrite this method if you don't want to use default optimizer or scheduler or if we don't provide
            the optimizers you need.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        """Create optimizer if optimizer is not specified

        """
        model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(model, nn.LayerNorm)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_param_groups = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0
                }
            ]
            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(optimizer_param_groups, **optim_kwargs)
        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value
        optim_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }

        if args.optim == OptimizerNames.SGD:
            from torch.optim.sgd import SGD
            optim_cls = SGD
            if optim_args.get("momentum") is not None:
                optim_kwargs.update({"momentum": optim_args["momentum"]})

        elif args.optim == OptimizerNames.ADAM:
            from torch.optim.adam import Adam
            optim_cls = Adam
            optim_kwargs.update(adam_kwargs)

        elif args.optim == OptimizerNames.ADAMW:
            from torch.optim.adamw import AdamW
            optim_cls = AdamW
            optim_kwargs.update(adam_kwargs)

        elif args.optim == OptimizerNames.ADAGRAD:
            from torch.optim.adagrad import Adagrad
            optim_cls = Adagrad

        else:
            raise NotImplementedError(f"Do not support optimizer: {args.optim}, currently support: {OptimizerNames}")

        return optim_cls, optim_kwargs

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Set up the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            num_training_steps:
            optimizer:
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def train(self, resume_from_checkpoint: bool = True, overwrite_output_dir: bool = False):
        num_steps_per_epoch = len(self.train_loader) // self.args.gradient_accumulation_steps
        steps_trained_in_current_epoch = 0
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            end_epoch = self.args.max_steps // num_steps_per_epoch

        else:
            max_steps = self.args.epoch * num_steps_per_epoch

            end_epoch = self.args.epoch
        print(f"max_steps: {max_steps}, end_epoch: {end_epoch}, num_steps_per_epoch: {num_steps_per_epoch}")
        self.create_optimizer_and_scheduler(max_steps)

        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        if os.path.isdir(self.args.output_dir) and self.args.do_train \
                and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and resume_from_checkpoint is False:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
                self.load_checkpoint(last_checkpoint)

        if overwrite_output_dir and resume_from_checkpoint:
            raise ValueError("overwrite_output_dir and resume_from_checkpoint cannot be set to True at the same time")

        if overwrite_output_dir:
            if os.path.exists(self.args.output_dir):
                shutil.rmtree(self.args.output_dir)

        if resume_from_checkpoint:
            if os.path.isdir(self.args.output_dir):
                last_checkpoint = get_last_checkpoint(self.args.output_dir)
                if last_checkpoint is not None:
                    logger.info(
                        f"Resume training from checkpoint: {last_checkpoint}"
                    )
                    self.load_checkpoint(last_checkpoint)
                else:
                    logger.info(
                        f"No checkpoint detected in {self.args.output_dir}, will train from scratch"
                    )
            else:
                raise FileNotFoundError(f"{self.args.output_dir} does not exist")

        start_epoch = math.ceil(self.state.num_train_epochs)

        if self.args.do_train:
            logger.info("***** Running training *****")
            with logging_redirect_tqdm():
                for epoch in tqdm.trange(start_epoch, end_epoch):
                    self.training_loop(self.train_loader, max_steps, epoch)
                    if self.state.global_step >= max_steps:
                        break
        if self.args.do_predict:
            if self.args.do_predict:
                logger.info("***** Running test *****")
                self.test_loop(self.test_loader)

        self.tensorboard_writer.close()

    def training_loop(self, loader, num_train_steps: int, epoch: int):
        self.model.train()

        accu_loss = 0  # accumulated loss for gradient accumulation
        for step, sample in enumerate(loader):
            loss = self.training_step(sample)

            accu_loss += loss.item()

            if ((step + 1) % self.args.gradient_accumulation_steps) == 0 or (step + 1 == len(loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if self.args.log_tensorboard:
                    self.log("train_loss", accu_loss)
                    if self.args.log_lr:
                        self.log("learning_rate", self.optimizer.param_groups[0]['lr'])

                if self.state.global_step % self.args.logging_steps == 0:
                    logger.info(f"Loss: {accu_loss: .2f}, Epoch: {epoch + step / len(loader): .2f}, "
                                f"Steps: {self.state.global_step}, LR: {self.optimizer.param_groups[0]['lr']: .6f}")

                if self.state.global_step % self.args.eval_steps == 0 and self.args.do_eval:
                    logger.info("***** Running eval *****")
                    self.eval_loop(self.eval_loader)

                if self.state.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()
                self.state.global_step += 1
                self.state.num_train_epochs = epoch + step / len(loader)
                accu_loss = 0

                if self.state.global_step >= num_train_steps:
                    if self.args.save_last:
                        self.save_checkpoint()
                    return

    def training_step(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """Steps for training,
        If not overwritten, it treats training as a classification problem
        """
        X, y = sample
        X, y = X.to(device=self.device), y.to(device=self.device)

        with autocast():  # using autocast for automatic mixed precision
            output = self.model(X)
            loss = self.compute_loss(output, y) / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()  # scale the loss before backward()

        return loss.detach()  # return Tensor, not item

    def eval_loop(self, loader):
        accuracy = 0
        losses = 0.0
        num = len(loader)
        for sample in loader:
            correct, loss = self.eval_step(sample)
            losses += loss
            accuracy += correct

        eval_accuracy, eval_loss = accuracy / num, losses / num
        logger.info(f"Eval Loss: {eval_loss: .2f}, Eval Acc: {eval_accuracy * 100: .2f}%")
        if self.args.log_tensorboard:
            self.log("eval_accuracy", eval_accuracy)
            self.log("eval_loss", eval_loss)
        self.model.train()

    def eval_step(self, sample: Tuple[Any, Any]):
        if (self.args.do_eval or self.args.do_predict) and self.compute_metric is None:
            raise NotImplementedError("do_eval or do_predict requires implementing compute_metric to do evaluation")

        self.model.eval()
        with torch.no_grad():
            predictions, y = self.compute_metric(self.model, sample, self.device)
            loss = self.compute_loss(predictions, y)
            _, prediction_index = predictions.max(dim=1)
            num_correct = (prediction_index == y).sum().data
            num_sample = y.size(0)
            return num_correct / num_sample, loss.item()

    def test_loop(self, loader):
        num_correct = 0
        losses = 0.0
        num = len(loader)
        for sample in loader:
            correct, loss = self.eval_step(sample)
            losses += loss
            num_correct += correct

        self.model.train()
        test_accuracy, test_loss = num_correct / num, losses / num
        logger.info(f"Test Loss: {test_loss: .2f}, Test Acc: {test_accuracy * 100: .2f}%")
        if self.args.log_tensorboard:
            self.log("test_accuracy", test_accuracy)
            self.log("test_loss", test_loss)

    def compute_loss(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes loss for training.
        If not overwritten, it takes prediction and ground truth and use `self.criterion` to compute loss
        """
        if self.criterion is None:
            raise NotImplementedError("Trainer must define a criterion for computing loss")
        else:
            return self.criterion(output, y)

    def log(self, stat_name, stat, stat_type="scalar"):
        add_stat_method = getattr(self.tensorboard_writer, "add_" + stat_type)
        add_stat_method(stat_name, stat, self.state.global_step)

    def load_checkpoint(self, checkpoint: str):
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint)
        state_file = os.path.join(checkpoint_dir,  TRAINER_STATE_NAME)
        model_file = os.path.join(checkpoint_dir, MODEL_STATE_NAME)
        optimizer_file = os.path.join(checkpoint_dir, OPTIMIZER_STATE_NAME)
        lr_scheduler_file = os.path.join(checkpoint_dir, LR_SCHEDULER_STATE_NAME)

        self.state = TrainerState.load_from_json(state_file)
        self.model.load_state_dict(torch.load(model_file))
        self.optimizer.load_state_dict(torch.load(optimizer_file))
        self.lr_scheduler.load_state_dict(torch.load(lr_scheduler_file))

    def save_checkpoint(self):
        checkpoint = f"checkpoint-{self.state.global_step}"
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        if len(os.listdir(self.args.output_dir)) < self.args.max_save or self.args.max_save is None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self._save_checkpoint(checkpoint_dir)

        else:
            first_checkpoint = get_first_checkpoint(self.args.output_dir)
            shutil.rmtree(os.path.join(self.args.output_dir, first_checkpoint))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self._save_checkpoint(checkpoint_dir)

    def _save_checkpoint(self, checkpoint_dir: str):
        state_file = os.path.join(checkpoint_dir, TRAINER_STATE_NAME)
        model_file = os.path.join(checkpoint_dir, MODEL_STATE_NAME)
        optimizer_file = os.path.join(checkpoint_dir, OPTIMIZER_STATE_NAME)
        lr_scheduler_file = os.path.join(checkpoint_dir, LR_SCHEDULER_STATE_NAME)
        training_args_file = os.path.join(checkpoint_dir, TRAINING_ARGS_NAME)

        self.state.save_to_json(state_file)
        torch.save(self.model.state_dict(), model_file)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        torch.save(self.lr_scheduler.state_dict(), lr_scheduler_file)
        torch.save(self.args, training_args_file)


class ClassificationTrainer(Trainer):

    @classmethod
    def accuracy(cls, y_hat: torch.Tensor, y: torch.Tensor, averaged: bool = True) -> float:
        _, prediction_index = y_hat.max(dim=1)
        num_correct = (prediction_index == y).sum().data
        return num_correct / y_hat.size(0) if averaged else num_correct
