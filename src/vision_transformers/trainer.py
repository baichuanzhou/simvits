import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from torch.utils.data import Dataset
import logging
import os
from .trainer_utils import (
    get_last_checkpoint,
    SchedulerType, OptimizerNames,
    set_seed, enable_full_determinism
)
from .training_args import TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    TODO: Write instructions for this class
    """
    def __init__(
            self,
            model: nn.Module,
            training_args: TrainingArguments = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,

            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        if training_args is None:
            output_dir = "tmp_output_dir"
            training_args = TrainingArguments(output_dir=output_dir)
        self.args = training_args
        # Set the seed before training
        enable_full_determinism(self.args.seed) if self.args.enable_full_deterministic else set_seed(self.args.seed)

        if os.path.isdir(self.args.output_dir) and self.args.do_train \
                and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self.args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        if model is None:
            raise RuntimeError("Trainer must be provided with a model")

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        self.model = model
        self.optimizer, self.lr_scheduler = optimizers

        if self.args.max_steps > 0:
            logger.warning("max_steps is given, overriding num_train_epochs if given")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset


    def training_steps(self):
        pass
