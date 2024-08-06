import random
import functools
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List 

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    #enable_wrap,
    #wrap,
)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

from peft import LoraConfig, get_peft_model

# local 
from d import *
from policies.wrapping import fsdp_auto_wrap_policy, get_llama_wrapper
from utils.train_utils import clear_gpu_cache, setup_environ_flags

from model import Beni, BeniConfig

def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


@dataclass
class TrainConfig:
    n_epochs: int = 1 
    warmup_ratio: float = 0.03
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    save_steps: int = 100
    log_steps: int = 1
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    lr: float = 1e-04
    min_lr: float = 1e-05
    betas: List = field(default_factory = lambda: [0.9, 0.999]),
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None 
    wandb_report: bool = False
    ckpt_path: Optional[str] = None
    save_path: Optional[str] = None


def fsdp_main(model_config: BeniConfig, train_config: TrainConfig, fsdp_config):
    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    # setup each cuda device ('device' aliased to cuda:n)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(rank)
        setup_environ_flags(rank)

        seed_everything(rank)

    # load from rank 0 and broadcast weights
    if rank == 0:
        model = Beni(model_config) # cpu

        if train_config.ckpt_path is not None:
            model.load_state_dict(torch.load(train_config.ckpt_path))
    else:
        with torch.device("meta"):
            model = Beni(model_config)    
            
    
    # LORA
    #peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')
    #model = get_peft_model(model, peft_config)
    #model.print_trainable_parameters()
    
    # TODO: make the below use fsdp_config
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, (LlamaDecoderLayer, CLIPEncoderLayer))
    #my_auto_wrapping_policy = llama_autowrap_policy()

    model = FSDP(
        model,
        auto_wrap_policy= my_auto_wrapping_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, #SHARD_GRAD_OP
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False) if rank != 0 else None,
    )
    
    # model.to(torch.float16) #should cause nan

    #if fsdp_config.fsdp_activation_checkpointing:
    #    policies.apply_fsdp_checkpointing(model)

    dist.barrier()
    model.pretty_print()
        
    # optimizer
    params = [{
        "params": [p for p in model.connector.parameters() if p.requires_grad],
        "weight_decay": train_config.weight_decay,
        "lr": train_config.lr,
    },]
    optimizer = torch.optim.AdamW(
        params,
       #betas=train_config.betas,
    )

    # do rank-specific shuffling before train launch
    ds = load_recap(model.tok, 100)
    dl = DataLoader(ds, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tok))
    #dl = None

    train(model, optimizer, dl, train_config)


def train(model, optimizer, dl, train_config):
    c = train_config
    model.train()

    for i in range(c.n_epochs):
        for e, batch in enumerate(dl):
            tik = time.time()
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda") #should be cuda:rank

            loss = model(**batch)['loss']
            loss = loss / c.gradient_accumulation_steps
            loss.requires_grad=True
            loss.backward()
            
            if os.environ["LOCAL_RANK"] == '0':
                for param in model.connector.parameters():
                    print(param.grad)
           # if train_config.use_fp16:
           #     # if fp16 is enabled, use gradient scaler to handle gradient update
           #     scaler.scale(loss).backward()
           #     if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
           #         scaler.step(optimizer)
           #         scaler.update()
           #         optimizer.zero_grad()
           #     loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)

            if (e+1)%c.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            dt = time.time()-tik 
            shape = batch['input_ids'].shape

            if os.environ["LOCAL_RANK"] == '0':
                print(f"iter: {e}/{len(dl)} | loss: {loss.item()} | left: {(dt*(len(dl)-e)/60):.2f} [{dt:.2f}s/it]")


    # training done
    dist.barrier()

    if False: #replace based on config later
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

    VISION_MODEL_ID = "openai/clip-vit-large-patch14"
    TEXT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model_config = BeniConfig(
        vision_name_or_path = VISION_MODEL_ID,
        llm_name_or_path = TEXT_MODEL_ID,
        r = 16,
        freeze = True,
    )
    train_config = TrainConfig(
        batch_size = 2,
    )
    fsdp_config = None

    # launch train
    fsdp_main(model_config, train_config, fsdp_config)

    cleanup()
