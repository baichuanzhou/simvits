import os
import re
from enum import Enum
import torch
import torch.nn as nn
import random
import numpy as np


def get_last_checkpoint(folder: str) -> str:
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _RE_CHECKPOINT = r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$"

    files = os.listdir(folder)
    checkpoints = [
        checkpoint for checkpoint in files
        if re.match(checkpoint, _RE_CHECKPOINT) is not None and os.path.isdir(os.path.join(folder, checkpoint))
    ]
    max_checkpoints = max(checkpoints, key=lambda x: int(re.match(x, _RE_CHECKPOINT).groups()[0]))
    return os.path.join(folder, max_checkpoints)


def enable_full_determinism(seed: int):
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    ADAGRAD = "adagrad",


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


def get_parameter_names(model, skip_module):
    results = []
    for name, child in model.named_children():
        results += [
            f"{name}.{n}"
            for n in get_parameter_names(child, skip_module)
            if not isinstance(child, skip_module)
        ]
    results += list(model._parameters.keys())
    return results


def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """

    def numel(p):
        return p.numel()
    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)

