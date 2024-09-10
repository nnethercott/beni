import random
import functools
import os
import time
from dataclasses import asdict
import copy
import json
import uuid

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

# fsdp layers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from peft import get_peft_model

# local
from model.vision import VisionTower, VisionTowerConfig, BilinearConfig
from data import (
    MinHashLSHDeduplicator,
    load_recap,
    MultiDataLoader,
    sft_collate_fn,
)
from policies.wrapping import fsdp_auto_wrap_policy
import policies
from utils.train_utils import (
    clear_gpu_cache,
    setup_environ_flags,
    get_cosine_schedule_with_warmup,
)
from configs import TrainConfig, WandbConfig, FSDPConfig
from checkpointing import save_model, load_model

from model import Beni, BeniConfig

TOKEN = os.getenv("HF_TOKEN", None)


def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def fuzzy_filter(*datasets, tokenizer: AutoTokenizer):
    """
    use dataset.data so we avoid loading images
    """
    datasets = list(datasets)

    # for some reason dataset.Dataset.to_list() converts pil images to bytes so we need
    # to update the CustomDataset.data attribute AFTER creation

    text_only = []
    for d in datasets:
        if isinstance(d.data, dict):
            d.data = [dict(zip(d.data, t)) for t in zip(*d.data.values())]
        text_only.append([item["response"] for item in d.data])

    # fuzzy deduplication with minhash
    minhash = MinHashLSHDeduplicator(tokenizer, *text_only)
    duplicate_ids = minhash.deduplicate(jaccard_sim=0.85, num_perm=128)

    print(f"{len(duplicate_ids)} duplicates detected!\nremoving them now...")

    # clean original datasets
    for e, id_list in enumerate(duplicate_ids):
        for idx in id_list[::-1]:
            _ = datasets[e].data.pop(idx)

    dist.barrier()
    return datasets


def get_dataloader(tokenizer, model_config, train_config):
    instruction_template = model_config.instruction_template
    response_template = model_config.response_template

    # recap
    print("loading recap...")
    recap = load_recap(
        tokenizer,
        n=200000,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    # optimized ordering of samples for uniform seq lengths
    def sort_batch_shuffle(data, winsize):
        data = sorted(data, key=lambda x: len(x["input_ids"]))
        data = [
            data[winsize * i : winsize * (i + 1)]
            for i in range(len(data) // winsize + 1)
        ]
        random.shuffle(data)
        data = [x for xs in data for x in xs]  # thanks stack
        return data

    for dataset in [recap]:
        dataset.data = sort_batch_shuffle(dataset.data, train_config.batch_size)

    # multidataloader
    loaders = (
        DataLoader(
            d,
            batch_size=train_config.batch_size,
            collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )
        for d in [recap]
    )

    dl = MultiDataLoader(
        *loaders, seed=42 + dist.get_rank()
    )  # each rank should see same modality when seed=int (but not when we use rank info)

    return dl


def fsdp_main(model_config, **kwargs):
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

                # load from checkpoint
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
            model, fsdp_config.transformer_cls
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
    dl = get_dataloader(model.tokenizer, model.config, train_config)

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
            int(train_config.warmup_ratio * len(dl) * train_config.n_epochs),
            len(dl) * train_config.n_epochs,
            min_lr=train_config.min_lr,
        )
    elif train_config.scheduler == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                train_config.warmup_ratio * len(dl) * train_config.n_epochs
            ),
            last_epoch=-1,
        )
    else:
        pass

    # launch train
    train(model, optimizer, scheduler, dl, train_config, wandb_run=wandb_run)


def train(model, optimizer, scheduler, dl, train_config, wandb_run=None):
    print("training")
    c = train_config
    model.train()

    # recover rank
    rank = int(os.getenv("LOCAL_RANK", 0))

    total_len = len(dl)
    start_time = time.time()
    prev_batch = {}

    for i in range(c.n_epochs):
        for e, batch in enumerate(dl):
            tik = time.time()
            if batch is None:  # if we can't download imgs use previous data
                print(f"Encountered empty batch on rank {dist.get_rank()}")
                batch = prev_batch
            else:
                prev_batch = batch

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda")  # assumes gpu training

            # forward pass
            out = model(**batch, output_attentions=False)
            loss = out["loss"]

            loss = loss / c.gradient_accumulation_steps
            loss.backward()

            if (e + 1) % c.gradient_accumulation_steps == 0:
                if c.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            if (e + 1) % c.log_steps == 0:
                dt = time.time() - tik

                # might block
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                if os.environ["LOCAL_RANK"] == "0":
                    print(
                        f"iter: {e+1}/{total_len}  loss: {c.gradient_accumulation_steps*loss.item():.2f}  lr: {scheduler.get_last_lr()[0]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*total_len/60):.2f}, {dt:.2f}s/it]"
                    )

            if wandb_run is not None:
                data = {
                    "loss": c.gradient_accumulation_steps * loss.item(),
                    "ppl": 2 ** (c.gradient_accumulation_steps * loss.item()),
                    "lr": scheduler.get_last_lr()[0],
                }

                if rank == 0:
                    wandb_run.log(data)

            if (e + 1) % c.save_steps == 0:
                dist.barrier()
                save_dir = os.path.join(c.save_path, f"step{e+1}")
                save_model(model, save_dir=save_dir, rank=dist.get_rank())
                dist.barrier()

    # training done
    dist.barrier()

    # save on epoch end
    save_dir = os.path.join(c.save_path, f"step{total_len}")
    save_model(model, save_dir=save_dir, rank=dist.get_rank())


if __name__ == "__main__":
    setup()
    vision_tower_config = VisionTowerConfig(
        r=10,
        feature_select_index=-2,
        use_cls=True,
        img_size=384,
        grid=(2, 2),  # 2x2 + 1 crops
        sparsity_plugins=[BilinearConfig(size=(16, 16))],
        perceiver_config=None,
    )

    model_config = BeniConfig(
        vision_name_or_path="google/siglip-so400m-patch14-384",
        text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # text_name_or_path="stabilityai/stablelm-2-1_6b-chat",
        vision_tower_config=vision_tower_config,
        vision_cls="SiglipVisionModel",
        vision_processor_cls="SiglipImageProcessor",
        freeze=True,
        attn_implementation="eager",
        # bos_token="<|user|>\n",  # offset needed for img token insert
        # instruction_template="<|user|>\n{instruction}<|endoftext|>\n<|assistant|>\n",  # no loss part
        # response_template="{response}<|endoftext|>\n",  # loss part
        bos_token="<|user|>",
        instruction_template="<|user|>\n{instruction}</s>\n<|assistant|>\n",
        response_template="{response}</s>",
        # llm_quantization_config = BitsAndBytesConfig(
        #    load_in_4bit = True,
        #    bnb_4bit_compute_dtype=torch.float16,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_quant_storage=torch.float16,
        # ),
    )

    train_config = TrainConfig(
        warmup_ratio=0.03,
        batch_size=4,
        gradient_accumulation_steps=4,
        mm_connector_lr=3e-04,
        weight_decay=0.1,
        min_lr=4e-05,
        grad_clip=1.0,
        save_steps=1000,
        log_steps=1,
        ckpt_path=None,
        save_path="/mnt/nate/model_checkpoints/grid_2x2",
        betas=[0.9, 0.95],
        scheduler="constant_with_warmup",
        fsdp=True,
        enable_peft=False,
    )

    # need this for weight tying: torch.nn.Embedding, # if we upscale images we can't fsdp the positional embeddings ?
    fsdp_config = FSDPConfig(
        transformer_cls=(
            LlamaDecoderLayer,
            # StableLmDecoderLayer,
            SiglipEncoderLayer,
            # PerceiverResampler,
            # nn.Embedding,
            VisionTower,
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
    # lora_config = LoraConfig(
    #    r=4,
    #    lora_alpha=32,
    #    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #    bias="none",
    # )
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
