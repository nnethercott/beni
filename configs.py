from dataclasses import dataclass, field
from typing import Optional, List

import torch

from utils import RMSNorm


@dataclass 
class Config:
    linear_cls: torch.nn.Module = torch.nn.Linear # add option later for bitlinear, lora 
    vocab_size: int = 30000
    bias: bool = False
    dropout: float = 0.0
    n_layer: int = 12 
    n_head: int =  12
    n_embd: int = 786 
    block_size: int = 1024 
    n_query_groups: int = 12
    rope_theta: int = 10000
    norm_cls: torch.nn.Module = RMSNorm
    norm_eps: float = 1e-05
    tie_weights: bool = True
    neftune_noise_alpha: float = 0.0

    @property 
    def head_size(self):
        return self.n_embd//self.n_head


@dataclass 
class TrainConfig:
    n_epochs: int = 1 
    warmup_ratio: float = 0.03
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    save_steps: int = 100
    log_steps: int = 1
    grad_clip: Optional[float] = None
    weight_decay: float = 0.0
    lr: float = 1e-03
    betas: List = field(default_factory = lambda: [0.9, 0.999]),
    min_lr: float = 1e-05
    distill_temperature: float = 1.0
    top_k: Optional[int] = None,
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None 
    wandb_report: bool = False
    ckpt_path: Optional[str] = None
    save_path: Optional[str] = None
    ddp: bool = False
    teacher_model_id: Optional[str] = None
