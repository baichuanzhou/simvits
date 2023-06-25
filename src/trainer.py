import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import Dataset


@dataclass
class TrainingArguments:
    """
    TODO: Write instructions for this class
    """
    do_train: bool = field(default=False, metadata={"help": "Whether to run training on training set"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on validation set"})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on test set"})

    learning_rate: float = field(default=1e-3, metadata={"help": "Learning rate for the model"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay rate for AdamW optimizer"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})



class Trainer:
    """
    TODO: Write instructions for this class
    """
    def __init__(self, model: nn.Module, train_args: TrainingArguments = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, test_dataset: Optional[Dataset] = None):
        self.model = model
        self.args = train_args
