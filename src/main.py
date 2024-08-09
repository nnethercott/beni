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
)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HfArgumentParser

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from peft import LoraConfig, get_peft_model

# local 
from d import *
from policies.wrapping import fsdp_auto_wrap_policy, get_llama_wrapper
from utils.train_utils import clear_gpu_cache, setup_environ_flags, get_cosine_schedule_with_warmup
from configs import TrainConfig, WandbConfig, FSDPConfig

from model import Beni, BeniConfig

def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def setup():
    dist.init_process_group("nccl")
def cleanup():
    dist.destroy_process_group()


def fsdp_main(model_config, **kwargs):
    # parse kwargs from launch script 
    #parser = HfArgumentParser((TrainConfig, FSDPConfig, WandbConfig))
    #train_config, fsdp_config, wandb_config = parser.parse_args_into_dataclasses()
    train_config = kwargs['train_config']
    wandb_config = kwargs['wandb_config']
    fsdp_config = kwargs['fsdp_config']

    wandb_run = wandb_config.build_run()
    
    rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    seed_everything(rank)

    # setup each cuda device ('device' aliased to cuda:n)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(rank)
        setup_environ_flags(rank)

    if train_config.fsdp:
        if rank == 0:
            model = Beni(model_config) # cpu
            if train_config.ckpt_path is not None:
                print(f"loading state dict from {train_config.ckpt_path}...")
                model.connector.load_state_dict(torch.load(train_config.ckpt_path))
        else:
            with torch.device("meta"):
                model = Beni(model_config)    
                
        # LORA
        #peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')
        #model = get_peft_model(model, peft_config)
        #model.print_trainable_parameters()
        
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, fsdp_config.transformer_cls)
        #my_auto_wrapping_policy = llama_autowrap_policy()

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=True,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False) if rank != 0 else None,
        )
        
        # model.to(torch.float16) #nan

        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)

        dist.barrier()

    else:
        model = Beni(model_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if train_config.ckpt_path is not None:
            model.load_state_dict(torch.load(train_config.ckpt_path))
        model.to(device)

    model.pretty_print()
        
    # data -- hardcoded here
    ds = load_recap(model.tok, 10000)
    #ds = tiny_shakespeare(model.tok, slen = 512)
    dl = DataLoader(ds, 
                    batch_size = train_config.batch_size, 
                    collate_fn = functools.partial(sft_collate_fn, tok=model.tok),
                    shuffle=True,
                    num_workers = 1,
                    pin_memory=True)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas, 
    )
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                int(train_config.warmup_ratio*len(dl)*train_config.n_epochs),
                                                len(dl)*train_config.n_epochs,
                                                min_lr = train_config.min_lr,
                                                )
                                                

    # launch train
    train(model, optimizer, scheduler, dl, train_config, wandb_run=wandb_run)


def prepare_batch(batch):
    return batch


def train(model, optimizer, scheduler, dl, train_config, wandb_run=None):
    c = train_config
    model.train()

    # recover rank 
    rank = int(os.getenv("LOCAL_RANK", 0))

    start_time = time.time()
    for i in range(c.n_epochs):
        for e, batch in enumerate(dl):
            if batch is None:
                continue 
            tik = time.time()

            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda") 

            out = model(**batch, output_attentions=False) 
            loss = out['loss']
            loss = loss / c.gradient_accumulation_steps
            #loss.requires_grad = True  # debug
            loss.backward()

            if (e+1)%c.gradient_accumulation_steps==0:
                if c.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            # step scheduler
            scheduler.step()

            if (e+1)%c.log_steps == 0: 
                dt = time.time()-tik 

                if os.environ["LOCAL_RANK"] == '0':
                    print(f"iter: {e+1}/{len(dl)}  loss: {loss.item():.2f}  lr: {scheduler.get_last_lr()[0]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*len(dl)/60):.2f}, {dt:.2f}s/it]")

            if wandb_config is not None:
                data = {'loss': loss.item(), 'ppl': 2**(loss.item()), 'lr': scheduler.get_last_lr()[0]}
                        

    # training done
    dist.barrier()

    # save on epoch end 
    # TODO: add saving at save_steps
    if c.save_path is not None:
        if rank == 0:
            print(f"saving model to {c.save_path}...")
        if c.fsdp:
            with FSDP.state_dict_type(
                    model, 
                    fsdp_config.checkpoint_type,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    cpu_state = model.connector.state_dict()
                    if rank == 0:
                        torch.save(cpu_state, c.save_path)
        else:
            torch.save(model.connector.state_dict(), c.save_path)



if __name__ == "__main__":
    setup()

    VISION_MODEL_ID = "openai/clip-vit-large-patch14"
    TEXT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #TEXT_MODEL_ID = "TinyLlama/TinyLlama_v1.1"

    model_config = BeniConfig(
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        r = 8,
        freeze = True,
        attn_implementation = "sdpa",
    )

    train_config = TrainConfig(
        warmup_ratio = 0.03,
        batch_size = 8,
        gradient_accumulation_steps = 1,
        lr = 1e-04,
        grad_clip = 1.0,
        save_steps = 100,
        log_steps = 1,
        save_path = 'test.pt',
        ckpt_path = None,
        betas = [0.9, 0.999],
        fsdp=True,
    )
    fsdp_config = FSDPConfig(
        transformer_cls=(LlamaDecoderLayer, CLIPEncoderLayer),
    )
    wandb_config = WandbConfig(
        enable = False,
        project = "vlm",
        entity = "nnethercott",
    )

    kwargs = {'train_config': train_config, 'fsdp_config': fsdp_config, 'wandb_config': wandb_config}

    # launch train
    fsdp_main(model_config, **kwargs)

    cleanup()
