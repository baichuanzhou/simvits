import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from torch.utils.data import Dataset
from enum import Enum


class ExplicitEnum(str, Enum):
    """
    This code is from huggingface implementation
        - https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/utils/generic.py#L341
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class OptimizerNames(ExplicitEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


@dataclass
class TrainingArguments:
    """
    TODO: Write instructions for this class
    """
    output_dir: str = field(metadata={"help": "The output directory for checkpoints and outputs"})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training on training set"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on validation set"})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on test set"})

    learning_rate: float = field(default=1e-3, metadata={"help": "Learning rate for the model"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay rate for AdamW optimizer"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    epoch: int = field(default=100, metadata={"help": "Total number of training epoch"})
    max_steps: int = field(default=-1, metadata={"help": "If > 0, overwrite epoch and "
                                                         "change training to steps counting strategy"})
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "Learning rate scheduler for training"}
    )
    warmup_ratio: float = field(default=0.0, metadata={"help": "Linear warm up ratio over number of steps"})
    warmup_steps: int = field(default=0, metadata={"help": "Number of linear warm up steps"
                                                           "Overwrite warmup_ratio"})

    logging_dir: Optional[str] = field(default=None, metadata={"help": "Log dir for Tensorboard"})
    logging_steps: int = field(default=500, metadata={"help": "Log data every logging_steps"})

    eval_steps: int = field(default=500, metadata={"help": "Eval model every eval_steps"})
    save_steps: int = field(default=500, metadata={"help": "Save model to output_dir every save_steps"})

    max_save: int = field(default=None, metadata={"help": "Maximum number of save to your output_dir"
                                                          "Delete older checkpoints like a stack"
                                                          "Default to None"})
    save_last: bool = field(default=True, metadata={"help": "Whether to save the last checkpoint"})


class Trainer:
    """
    TODO: Write instructions for this class
    """
    def __init__(
            self,
            model: nn.Module,
            train_args: TrainingArguments = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        if train_args is None:
            output_dir = "tmp_output_dir"
            train_args = TrainingArguments(output_dir=output_dir)
        self.train_args = train_args

        if model is None:
            raise RuntimeError("Trainer must be provided with a model")
