import random
import os
from dataclasses import asdict
import copy
import json
import uuid

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

# fsdp layers
from transformers import get_constant_schedule_with_warmup

from peft import get_peft_model, LoraConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

# local
from model.vision import VisionTowerConfig
from policies.wrapping import fsdp_auto_wrap_policy
import policies
from utils.train_utils import (
    clear_gpu_cache,
    setup_environ_flags,
    get_cosine_schedule_with_warmup,
)
from configs import TrainConfig, WandbConfig, FSDPConfig
from checkpointing import load_model
from data import get_train_dataloader
from model import Beni, BeniConfig
from train import train

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

    model_config_copy = copy.deepcopy(asdict(model_config))

    configs = {
        "model_config": model_config_copy,
        "train_config": train_config,
        "fsdp_config": fsdp_config,
        "lora_config": lora_config,
    }

    # save environment
    if rank == 0:
        os.system("pipreqs ../")
        with open("../requirements.txt", "r") as f:
            configs["pip_env"] = f.read().splitlines()

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

                # move pip_env to model run
                os.system(f"mv ../requirements.txt {s}")

        # quantized
        if model_config.llm_quantization_config is not None:
            model = Beni(model_config, hf_token=TOKEN)
        else:
            # load on cpu:0 only
            if rank == 0:
                model = Beni(model_config, hf_token=TOKEN)

                # load from checkpoint; merges any loras into base llm
                if train_config.ckpt_path is not None:
                    print(f"loading state dict from {train_config.ckpt_path}...")
                    model = load_model(model, train_config.ckpt_path, trainable=True)

            else:
                with torch.device("meta"):
                    model = Beni(model_config, hf_token=TOKEN)

        if train_config.enable_peft:
            assert (
                lora_config is not None
            ), "Either disable `enable_peft` or provide a valid lora config!"
            model.llm = get_peft_model(model.llm, lora_config)

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
            policies.apply_fsdp_checkpointing(model)

    else:
        # TODO add non fsdp training
        model = Beni(model_config)

    if rank == 0:
        print(model)  # type: ignore
        print("trainable params:")
        for n, p in model.named_parameters():  # type: ignore
            if p.requires_grad:
                print(n)

    dist.barrier()

    # data
    train_dl = get_train_dataloader(model.tokenizer, model.config, train_config)
    # eval_dl = get_eval_dataloader(model.tokenizer, model.config, train_config)
    eval_dl = None

    # optimizer and scheduler
    optimizer_parameter_groups = [
        {
            "params": [p for p in model.connector.parameters()],
            "weight_decay": train_config.weight_decay,
            "lr": train_config.mm_connector_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "connector" not in n],
            "weight_decay": train_config.weight_decay,
            "lr": train_config.llm_lr,
        },
    ]

    optimizer = torch.optim.AdamW(  # type: ignore
        optimizer_parameter_groups,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas,
    )

    if train_config.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(train_config.warmup_ratio * len(train_dl) * train_config.n_epochs),
            len(train_dl) * train_config.n_epochs,
            min_lr=train_config.min_lr,
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

    # launch train
    train(
        model,
        optimizer,
        scheduler,
        train_dl,
        train_config,
        eval_dl=eval_dl,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    setup()
    vision_tower_config = VisionTowerConfig(
        r=9,
        feature_select_index=-1,
        use_cls=True,
        img_size=384,
        grid=(1, 1),  # 2x2 + 1 crops
        # sparsity_plugins=[BilinearConfig(size=(22, 22))],
        perceiver_config=None,
    )

    # <|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n

    model_config = BeniConfig(
        vision_name_or_path="google/siglip-so400m-patch14-384",
        # vision_name_or_path="openai/clip-vit-large-patch14-336",
        text_name_or_path="HuggingFaceTB/SmolLM-135M-Instruct",
        vision_tower_config=vision_tower_config,
        vision_cls="SiglipVisionModel",
        vision_processor_cls="SiglipImageProcessor",
        freeze=True,
        attn_implementation="eager",
        bos_token="<|im_start|>user\n",
        instruction_template="<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        response_template="{response}<|im_end|>",
    )

    train_config = TrainConfig(
        warmup_ratio=0.03,
        batch_size=9,
        gradient_accumulation_steps=1,
        mm_connector_lr=0.00001,
        # llm_lr=6e-04,
        weight_decay=0.1,
        grad_clip=1.0,
        save_steps=1000,
        do_eval=False,
        eval_steps=500,
        log_steps=1,
        ckpt_path=None,
        save_path="/mnt/nate/model_checkpoints/smol",
        betas=[0.9, 0.999],
        scheduler="constant_with_warmup",
        # scheduler="cosine_with_warmup",
        fsdp=True,
        enable_peft=False,
        n_epochs=1,
    )

    # need this for weight tying: torch.nn.Embedding, # if we upscale images we can't fsdp the positional embeddings ?

    fsdp_config = FSDPConfig(
        transformer_cls=(
            SiglipEncoderLayer,
            Qwen2DecoderLayer,
            LlamaDecoderLayer,
            # LlamaDecoderLayer,
            # StableLmDecoderLayer,
            # SiglipEncoderLayer,  # PerceiverResampler,
            # VisionTower,
            torch.nn.modules.sparse.Embedding,
        ),
        fsdp_activation_checkpointing=False,
        fsdp_cpu_offload=False,
    )
    wandb_config = WandbConfig(
        enable=True,
        project="vlm",
        entity="nnethercott",
        name=model_config.text_name_or_path.split("/")[-1].lower()
        + "-"
        + str(uuid.uuid1()).split("-")[-1],  # model-archi-uuid
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=48,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    lora_config = None

    kwargs = {
        "train_config": train_config,
        "fsdp_config": fsdp_config,
        "wandb_config": wandb_config,
        "lora_config": lora_config,
    }

    # launch train
    fsdp_main(model_config, **kwargs)

    cleanup()

    def nate():
        return 1
