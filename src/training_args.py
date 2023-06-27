from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from .trainer_utils import SchedulerType, OptimizerNames


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

    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Whether to resume from last checkpoint"})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Whether to overwrite output directory"})

    per_device_train_batch: int = field(default=256, metadata={"help": "Batch size per device for training"})
    per_device_eval_batch: int = field(default=256, metadata={"help": "Batch size per device for validation"})
    per_device_test_batch: int = field(default=256, metadata={"help": "Batch size per device for testing"})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
