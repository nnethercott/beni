# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement. = functools.partial(sft_collate_fn, tok=tok)


import os
import functools
from pprint import pprint

import dataclasses
import random
import torch
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
)

# import model layers here
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from configs import (
    fsdp_config,
    train_config,
    quantization_config,
    lora_config,
    wandb_config,
)
from policies import AnyPrecisionAdamW, apply_fsdp_checkpointing, fsdp_auto_wrap_policy

from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

from d import tiny_shakespeare, sft_collate_fn

# Nate
# torch.bfloat16 if torch.cuda.get_device_capability()[0]>= 8 else torch.float16,  #bfloat16 only supported on GPUs with compute capability of at least 8.0

################### To fix #######################
# - add option for float32
# - replace model load with rank0 + meta device & sync_module_states
# - generalize the fsdp auto wrap policy to non-llamadecoder layers
# - add dataloader kwargs to train config
# - float16->bfloat16 when we get to better gpus
# - add param groups to optimizer to control weight decay terms
# - look into device meshes in fsdp
# - add cosine schedule instead of step lr


def get_dtype_from_config(cfg):
    if cfg.use_fp16:
        return torch.float16
    elif cfg.use_fp32:
        return torch.float32
    else:
        return torch.bfloat16


def main(
    train_config, fsdp_config, lora_config=None, quant_config=None, wandb_config=None
):
    wandb_run = None
    # register train config if wandbd exists
    if train_config.use_wandb:
        wandb_run = wandb.init(**dataclasses.asdict(wandb_config))
        wandb_run.update(train_config)
        wandb_run.update(fsdp_config)

        if train_config.use_peft:
            wandb_run.update(lora_config)

        if train_config.quantization:
            wandb_run.update(quant_config)

    # set seeds
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun env variables
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        quantization_config=bnb_config,
        use_cache=use_cache,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        device_map="auto"
        if train_config.quantization and not train_config.enable_fsdp
        else None,
        torch_dtype=get_dtype_from_config(train_config),
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    if rank == 0:
        print(model)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name
        if train_config.tokenizer_name is None
        else train_config.tokenizer_name,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if (
        train_config.enable_fsdp
        and train_config.pure_bf16
        and not train_config.quantization
    ):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True
            )
            model.peft_config()
        # Generate the peft config and start fine-tuning from original model
        else:
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(train_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        # fsdp
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy
            if train_config.use_peft
            else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True)
            if fsdp_config.fsdp_cpu_offload
            else None,
            mixed_precision=mixed_precision_policy
            if not train_config.pure_bf16
            else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(
                lambda module: module.to_empty(
                    device=torch.device("cuda"), recurse=False
                )
            )
            if train_config.low_cpu_fsdp and rank != 0
            else None,
            use_orig_params=True,
        )

        # activation checkpointing
        if fsdp_config.fsdp_activation_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)

    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    # NATE
    dataset_train = tiny_shakespeare(tokenizer, slen=128)
    collate_fn = functools.partial(sft_collate_fn, tok=tokenizer)

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    # if train_config.batching_strategy == "packing":
    #    dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        batch_size=train_config.batch_size_training,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    eval_dataloader = None
    # if train_config.run_validation:
    #    if train_config.batching_strategy == "packing":
    #        dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

    #    val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

    #    eval_dataloader = torch.utils.data.DataLoader(
    #        dataset_val,
    #        num_workers=train_config.num_workers_dataloader,
    #        pin_memory=True,
    #        **val_dl_kwargs,
    #    )
    #    if len(eval_dataloader) == 0:
    #        raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
    #    else:
    #        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    # Initialize the optimizer and learning rate scheduler
    if train_config.pure_bf16 and train_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v


if __name__ == "__main__":
    parser = HfArgumentParser(
        (train_config, fsdp_config, lora_config, quantization_config, wandb_config)
    )
    t_conf, f_conf, l_conf, q_conf, w_conf = parser.parse_args_into_dataclasses()

    if os.environ["LOCAL_RANK"] == "0":
        pprint(t_conf)

    main(t_conf, f_conf, l_conf, q_conf, w_conf)
