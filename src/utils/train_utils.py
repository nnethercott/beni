# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import contextlib
from functools import partial
import math

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers import LlamaTokenizer

# not making this pip installable so we'll add relative paths
import sys

sys.path.insert(1, "../checkpointing")
sys.path.insert(1, "../policies")

from policies import fpSixteen, bfSixteen, get_llama_wrapper
from accelerate.utils import is_xpu_available, is_ccl_available
from .flop_utils import FlopMeasure


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(
                f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            )
        print(
            f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}"
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait_step, warmup=warmup_step, active=active_step, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(
                f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            )
        with FlopMeasure(
            rank=local_rank, warmup_step=cfg.flop_counter_start
        ) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print("--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and torch.version.cuda >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    ) or (is_xpu_available())

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
        else:
            print("bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


# https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/optimization.py#L135
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_ratio: float = 0.0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_ratio=float(min_lr / (optimizer.param_groups[0]["lr"] + 1e-07)),
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
