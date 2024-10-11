import functools
import random
import os
from dataclasses import asdict
import copy
import json
import uuid

from accelerate.hooks import attach_align_device_hook
import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

# fsdp layers
from transformers import (
    SiglipVisionModel,
    get_constant_schedule_with_warmup,
)

from peft import get_peft_model, LoraConfig, PeftModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import (
    SiglipEncoderLayer,
    SiglipVisionTransformer,
)
from transformers.models.vit.modeling_vit import ViTLayer

# local
from model.vision import VisionTowerConfig
from policies import fsdp_auto_wrap_policy
import policies
from utils.train_utils import (
    clear_gpu_cache,
    setup_environ_flags,
    get_mixed_precision_policy,
)
from transformers import get_cosine_schedule_with_warmup
from configs import TrainConfig, WandbConfig, FSDPConfig
from checkpointing import load_model
from data import get_train_dataloader
from model import (
    VLM,
    VLMConfig,
    PerceiverResamplerConfig,
    PerceiverResampler,
    VisionTower,
)
from model.vision.sparsity import BilinearConfig
from train import train
from policies.anyprecision_optimizer import AnyPrecisionAdamW

# possibly needed for training models like llama3
TOKEN = os.getenv("HF_TOKEN", None)


def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def fsdp_main(model_config, **kwargs):
    # TODO
    # parse kwargs from launch script
    # parser = HfArgumentParser((TrainConfig, FSDPConfig, WandbConfig))
    # train_config, fsdp_config, wandb_config = parser.parse_args_into_dataclasses()

    train_config = kwargs["train_config"]
    wandb_config = kwargs["wandb_config"]
    fsdp_config = kwargs["fsdp_config"]
    lora_config = kwargs.get("lora_config", None)

    rank = int(os.getenv("LOCAL_RANK", 0))
    int(os.getenv("WORLD_SIZE", 1))
    seed_everything(rank)

    model_config_copy = model_config.to_dict()

    configs = {
        "model_config": model_config_copy,
        "train_config": train_config,
        "fsdp_config": fsdp_config,
        "lora_config": lora_config,
    }

    wandb_run = wandb_config.build_run(configs, rank == 0)

    # setup each cuda device ('device' aliased to cuda:n)
    if dist.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(rank)
        setup_environ_flags(rank)

    if train_config.fsdp:
        if rank == 0:
            # save model config
            s = train_config.save_path
            if s is not None:
                if not os.path.exists(s):
                    os.makedirs(s, exist_ok=True)

                with open(f"{s}/model_config.json", "w") as f:
                    f.write(json.dumps(model_config_copy))

        # quantized
        if model_config.llm_quantization_config is not None:
            model = VLM(model_config, hf_token=TOKEN)
        else:
            # load on cpu:0 only
            if rank == 0:
                model = VLM(model_config, hf_token=TOKEN)

                if train_config.save_path is not None:
                    model.tokenizer.save_pretrained(
                        f"{train_config.save_path}/tokenizer"
                    )

                # load from checkpoint; merges any loras into base llm
                if train_config.ckpt_path is not None:
                    print(f"loading state dict from {train_config.ckpt_path}...")
                    model = load_model(
                        model,
                        train_config.ckpt_path,
                        trainable=True,
                        merge_and_unload=False,  # keep lora from previous train iter
                    )
                    # print(model)

            else:
                with torch.device("meta"):
                    model = VLM(model_config, hf_token=TOKEN)

        # TODO: maybe switch to unified lora for whole model instead
        if train_config.enable_peft_vision and not isinstance(model.vision, PeftModel):
            assert (
                lora_config is not None
            ), "Either disable `enable_peft` or provide a valid lora config!"
            if rank == 0:
                print("creating vision loras...")
            if train_config.enable_peft_vision:
                model.vision = get_peft_model(model.vision, lora_config)

        if train_config.enable_peft_llm and not isinstance(model.llm, PeftModel):
            assert (
                lora_config is not None
            ), "Either disable `enable_peft` or provide a valid lora config!"
            if rank == 0:
                print("creating llm loras...")
            if train_config.enable_peft_llm:
                model.llm = get_peft_model(model.llm, lora_config)

        # model precision
        if train_config.bfloat16:
            model.to(torch.bfloat16)
        elif train_config.fp16:
            model.to(torch.float16)

        if rank == 0:
            params = sum((p.numel() for p in model.parameters()))
            trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
            print(
                f"VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\n"
            )

        my_auto_wrapping_policy = fsdp_auto_wrap_policy(
            model,
            fsdp_config.transformer_cls,
        )

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            mixed_precision=get_mixed_precision_policy(train_config, rank=rank),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=True,
            param_init_fn=lambda module: module.to_empty(  # type: ignore
                device=torch.device("cuda"), recurse=False
            )
            if rank != 0
            else None,
            use_orig_params=False,
        )
        # model.to(torch.float16) #nan

        if fsdp_config.fsdp_activation_checkpointing:
            # might need to add stuff for lora with checkpointing
            # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/finetuning.py#L193
            policies.apply_fsdp_checkpointing(model)

    else:
        # TODO add non fsdp training
        model = VLM(model_config)

    if rank == 0:
        print(model)  # type: ignore
        print("trainable params:")
        for n, p in model.named_parameters():  # type: ignore
            if p.requires_grad:
                print(n)

    dist.barrier()

    # data
    train_dl = get_train_dataloader(
        model.tokenizer, model.config, train_config
    )  # FIXME:
    # eval_dl = get_eval_dataloader(model.tokenizer, model.config, train_config)
    eval_dl = None

    # optimizer and scheduler
    optimizer_parameter_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "llm" not in n],
            "weight_decay": train_config.weight_decay,
            "lr": train_config.mm_connector_lr,
        },
        {
            "params": [p for p in model.llm.parameters()],
            "weight_decay": train_config.weight_decay,
            "lr": train_config.llm_lr,
        },
    ]

    if train_config.fp16:
        optimizer = AnyPrecisionAdamW(
            optimizer_parameter_groups,
            lr=train_config.mm_connector_lr,
            betas=train_config.betas,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            compensation_buffer_dtype=torch.bfloat16,
        )
    else:
        optimizer = torch.optim.AdamW(  # type: ignore
            optimizer_parameter_groups,
            weight_decay=train_config.weight_decay,
            betas=train_config.betas,
        )

    # huggingface one
    if train_config.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(train_config.warmup_ratio * len(train_dl) * train_config.n_epochs),
            len(train_dl) * train_config.n_epochs,
            # min_lr=train_config.min_lr,
        )
    elif train_config.scheduler == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                train_config.warmup_ratio * len(train_dl) * train_config.n_epochs
            ),
            last_epoch=-1,
        )
    else:
        # constant lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # sanity check to make sure we added new tokens
    # print(model.tokenizer.added_tokens_decoder)

    # launch train
    train(
        model,
        optimizer,
        scheduler,
        train_dl_factory=lambda: get_train_dataloader(
            model.tokenizer, model_config, train_config
        ),
        train_config=train_config,
        eval_dl=eval_dl,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    setup()
    from transformers import AddedToken

    ATTN_IMPLEMENTATION = "eager"

    vision_tower_config = VisionTowerConfig(
        r=8,
        feature_select_index=-1,
        use_cls=True,
        img_size=448,
        grid=(1, 1),
        # sparsity_plugins=[BilinearConfig(size=(28, 28))],
        use_global_crop=False,
    )

    # some additional tokens
    # special_tokens = ["<|vision_start|>", "<|vision_end|>"]
    # special_tokens += ["<|box_start|>", "<|box_end|>"] # change in datasets
    # special_tokens += [f"0.{i}" for i in range(10)]
    # special_tokens += [f"0.{i}{j}" for i in range(10) for j in range(1, 10)]
    # special_tokens += ["1.0"]

    model_config = VLMConfig(
        vision_name_or_path="google/siglip-so400m-patch14-384",
        text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # text_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        vision_tower_config=vision_tower_config,
        vision_cls="SiglipVisionModel",
        freeze=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        bos_token="<|im_start|>user:\n",
        instruction_template="<|im_start|>user:\n{instruction}<|im_end|>\n<|im_start|>assistant:\n",
        response_template="{response}<|im_end|><|endoftext|>",
        unfreeze_lm_head=False,
        # special_tokens=special_tokens,
        # lm_loss_lambda=1.0,
    )

    train_config = TrainConfig(
        warmup_ratio=0.03,
        batch_size=1,
        gradient_accumulation_steps=6,
        mm_connector_lr=2e-04,
        llm_lr=2e-04,
        weight_decay=0.0,
        grad_clip=1.0,
        save_steps=2,
        do_eval=False,
        eval_steps=500,
        log_steps=2,
        # ckpt_path="/mnt/nate/model_checkpoints/ocr_pretrain_clone/epoch3_step10418",
        save_path="/mnt/nate/model_checkpoints/test",
        betas=[0.9, 0.999],
        scheduler="cosine_with_warmup",  # "constant_with_warmup",  # "cosine_with_warmup"
        fsdp=True,
        enable_peft_llm=True,
        enable_peft_vision=False,
        n_epochs=4,
        bfloat16=False,
        fp16=True,
        mixed_precision=True,
    )

    # need this for weight tying: torch.nn.Embedding, # if we upscale images we can't fsdp the positional embeddings ?

    fsdp_config = FSDPConfig(
        transformer_cls=(
            SiglipEncoderLayer,
            Qwen2DecoderLayer,
            LlamaDecoderLayer,
            # PerceiverResampler,
            # torch.nn.modules.sparse.Embedding,
            # VisionTower,
        ),
        fsdp_activation_checkpointing=False,
        fsdp_cpu_offload=False,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )

    wandb_config = WandbConfig(
        enable=False,
        project="ocr-theory",
        entity="nnethercott",
        name=model_config.text_name_or_path.split("/")[-1].lower()
        + "_"
        + model_config.vision_name_or_path.split("/")[-1].lower()
        + "-"
        + str(uuid.uuid1()).split("-")[1],  # model-archi-uuid
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,
        target_modules=[
            "key",
            "query",
            "value",
            "dense",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        ],
        bias="none",
    )
    # lora_config = None

    kwargs = {
        "train_config": train_config,
        "fsdp_config": fsdp_config,
        "wandb_config": wandb_config,
        "lora_config": lora_config,
    }

    # launch train
    fsdp_main(model_config, **kwargs)

    cleanup()
