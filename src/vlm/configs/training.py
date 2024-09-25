# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainConfig:
    n_epochs: int = 1
    warmup_ratio: float = 0.03
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    save_steps: int = 100
    eval_steps: Optional[int] = None
    do_eval: bool = False
    log_steps: int = 1
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    mm_connector_lr: float = 1e-05
    llm_lr: float = 1e-04
    min_lr: float = 1e-05
    betas: List = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "cosine_with_warmup"
    ckpt_path: Optional[str] = None
    save_path: Optional[str] = None
    fsdp: bool = True
    enable_peft: bool = False
    tf32: bool = False
    fp16: bool = False
    mixed_precision: bool = False
    bfloat16: bool = False


@dataclass
class train_config:
    model_name: str = "PATH/to/Model"
    tokenizer_name: str = None
    enable_fsdp: bool = True
    low_cpu_fsdp: bool = True
    log_steps: int = 100
    run_validation: bool = False
    batch_size_training: int = 1
    batching_strategy: str = "padding"  # alternative: padding
    context_length: int = 512
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 1
    max_train_step: int = 0
    max_eval_step: int = 0
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    optimizer: str = "AdamW"
    seed: int = 42
    use_fp16: bool = False
    use_fp32: bool = False
    pure_bf16: bool = False
    mixed_precision: bool = False
    val_batch_size: int = 1
    peft_method: str = "lora"  # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool = False
    from_peft_checkpoint: str = ""  # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "LAST_MODEL/"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = False
    dist_checkpoint_root_folder: str = "FSDP_MODEL/"  # will be used if using FSDP
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False  # Enable wandb for experient tracking
    save_metrics: bool = (
        False  # saves training metrics to a json file for later plotting
    )
    flop_counter: bool = False  # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3  # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False  # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = (
        "PATH/to/save/profiler/results"  # will be used if using profiler
    )
