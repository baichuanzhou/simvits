from dataclasses import dataclass
import dataclasses
import json
from typing import Optional


@dataclass
class TrainerState:
    epoch: float = 0.0
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    best_model_checkpoint: Optional[str] = None

    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
