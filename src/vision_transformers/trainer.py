import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Any, Callable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from .trainer_utils import (
    get_last_checkpoint,
    SchedulerType, OptimizerNames,
    set_seed, enable_full_determinism,
    get_parameter_names, get_model_param_count
)
from .training_args import TrainingArguments
from .trainer_state import TrainerState
from .optimization import get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TRAINER_STATE_NAME = "trainer_state.json"


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
        if self.args.log_tensorboard:
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
        num_steps_per_epoch = len(self.train_loader)
        start_epoch = math.ceil(self.state.global_step / num_steps_per_epoch)
        steps_trained_in_current_epoch = 0
        model = self.model
        logger.info("***** Running training *****")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        last_checkpoint = None
        if os.path.isdir(self.args.output_dir) and self.args.do_train \
                and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        if resume_from_checkpoint and last_checkpoint is not None and os.path.isfile(
                os.path.join(last_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(last_checkpoint, TRAINER_STATE_NAME))
            steps_trained_in_current_epoch = self.state.global_step % num_steps_per_epoch

        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            end_epoch = self.args.max_steps // num_steps_per_epoch

        else:
            max_steps = self.args.epoch * num_steps_per_epoch

            end_epoch = self.args.epoch
        print(f"max_steps: {max_steps}, end_epoch: {end_epoch}, num_steps_per_epoch: {num_steps_per_epoch}")
        self.create_optimizer_and_scheduler(max_steps)
        start_epoch = math.ceil(self.state.num_train_epochs)
        for epoch in range(start_epoch, end_epoch):
            self.training_loop(self.train_loader, max_steps, epoch)
            if self.state.global_step >= max_steps:
                break
        self.tensorboard_writer.close()

    def training_loop(self, loader, num_train_steps: int, epoch: int):
        self.model.train()

        for step, sample in enumerate(loader):
            loss = self.training_step(sample)
            if self.state.global_step % self.args.logging_steps == 0:
                logger.info(f"Loss: {loss: .2f}, Epoch: {epoch + step / len(loader): .2f}, "
                            f"Steps: {self.state.global_step}, LR: {self.optimizer.param_groups[0]['lr']: .6f}")

            if self.state.global_step % self.args.eval_steps == 0 and self.args.do_eval:
                logger.info("***** Running eval *****")
                self.eval_loop(self.eval_loader)
            self.state.global_step += 1
            self.state.epoch_trained = epoch + step / len(loader)

            if self.state.global_step >= num_train_steps:
                if self.args.do_predict:
                    logger.info("***** Running test *****")
                    self.test_loop(self.test_loader)
                return
            self.lr_scheduler.step()

    def training_step(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        """Steps for training,
        If not overwritten, it treats training as a classification problem
        """
        X, y = sample
        X, y = X.to(self.device), y.to(self.device)
        output = self.model(X)
        loss = self.compute_loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.args.log_tensorboard:
            self.log("train_loss", loss.item())
        return loss.item()

    def eval_loop(self, loader):
        accuracy = 0
        losses = 0.0
        num = len(loader)
        for sample in loader:
            correct, loss = self.eval_step(sample)
            losses += loss
            accuracy += correct

        self.model.train()
        eval_accuracy, eval_loss = accuracy / num, losses / num
        logger.info(f"Eval Loss: {eval_loss: .2f}, Eval Acc: {eval_accuracy * 100: .2f}%")
        if self.args.log_tensorboard:
            self.log("eval_accuracy", eval_accuracy)
            self.log("eval_loss", eval_loss)

    def eval_step(self, sample: Tuple[Any, Any]):
        if (self.args.do_eval or self.args.do_predict) and self.compute_metric is None:
            raise NotImplementedError("do_eval or do_predict requires implementing compute_metric to do evaluation")
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

    def compute_loss(self, output, y):
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

    def _load_checkpoint(self, checkpoint: str):
        pass

    def _save_checkpoint(self, checkpoint: str):
        pass
