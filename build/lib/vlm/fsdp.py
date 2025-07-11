import random
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# local
from d import *
from policies.wrapping import fsdp_auto_wrap_policy
from utils.train_utils import clear_gpu_cache, setup_environ_flags


# make this a constant
def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def fsdp_main(model_config, train_config, fsdp_config):
    rank = int(os.getenv("LOCAL_RANK"))
    int(os.getenv("WORLD_SIZE"))

    # setup each cuda device ('device' aliased to cuda:n)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

        seed_everything(rank)

    # load model & tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    setattr(config, "num_hidden_layers", 11)

    if rank == 0:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="sdpa",
        )
        # model.load_state_dict(torch.load("tinyllama0.pt"))
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")

    sum(p.numel() for p in model.parameters())

    # LORA
    # peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, (LlamaDecoderLayer,))
    # my_auto_wrapping_policy = llama_autowrap_policy()

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrapping_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # SHARD_GRAD_OP
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
        if rank != 0
        else None,
    )

    # model.to(torch.float16) #should cause nan

    # if fsdp_config.fsdp_activation_checkpointing:
    #    policies.apply_fsdp_checkpointing(model)

    dist.barrier()
    if rank == 0:
        print(model)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-04,
        weight_decay=0.1,
    )

    # do rank-specific shuffling before train launch
    dl = DataLoader(
        tiny_shakespeare(tok, slen=128),
        batch_size=1,
        num_workers=1,
        collate_fn=functools.partial(sft_collate_fn, tok=tok),
        shuffle=True,
        pin_memory=True,
    )

    train(model, optimizer, dl)


def train(model, optimizer, dl, config):
    model.train()

    for i in range(epochs):
        for e, batch in enumerate(dl):
            tik = time.time()
            for key in batch.keys():
                batch[key] = batch[key].to(rank)

            loss = model(**batch)["loss"]
            loss = loss / g
            loss.backward()

            # if train_config.use_fp16:
            #     # if fp16 is enabled, use gradient scaler to handle gradient update
            #     scaler.scale(loss).backward()
            #     if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            #         scaler.step(optimizer)
            #         scaler.update()
            #         optimizer.zero_grad()
            #     loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (e + 1) % g == 0:
                optimizer.step()
                optimizer.zero_grad()

            dt = time.time() - tik
            batch["input_ids"].shape

            if os.environ["LOCAL_RANK"] == "0":
                print(
                    f"iter: {e}/{len(dl)} | loss: {loss.item()} | left: {(dt*(len(dl)-e)/60):.2f} [{dt:.2f}s/it]"
                )

    # training done
    dist.barrier()

    if False:  # replace based on config later
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, "shakespeare.pt")

    else:
        model.save_pretrained("./")


if __name__ == "__main__":
    setup()
    # fsdp_main()
    cleanup()
