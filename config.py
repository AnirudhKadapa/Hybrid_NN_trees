from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    epochs: int = 150
    weight_decay: float = 1e-4
    patience: int = 15
    lr : float = 3e-3
    batch_size: int = 4096
    

    
