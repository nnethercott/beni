import random
import functools
import os
import time
from dataclasses import dataclass, field, asdict
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
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

from peft import LoraConfig, get_peft_model

# local 
from data import *
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
    lora_config = kwargs.get("lora_config", None)

    # TODO: run a pip freeze on the venv and upload that too
    """
    try: 
        from pip._internal.operations import freeze
    except ImportError: # pip < 10.0
        from pip.operations import freeze

    pkgs = freeze.freeze() # add this to config dict
    """
    configs = {'model_config': model_config, 'train_config': train_config, 'fsdp_config': fsdp_config, 'lora_config': lora_config}
    wandb_run = wandb_config.build_run(configs)
    
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
        # TODO: make beniconfig subclass hf config so we can use model.save_pretrained
        if train_config.enable_peft:
            assert lora_config is not None 
            model.llm = get_peft_model(model.llm, lora_config)


        if rank == 0:
            # print stuff
            params = sum((p.numel() for p in model.parameters()))
            trainable = sum((p.numel() for p in model.parameters() if p.requires_grad)) 
            print(f'VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\n')

        
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

        #dist.barrier()

    else:
        model = Beni(model_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if train_config.ckpt_path is not None:
            model.load_state_dict(torch.load(train_config.ckpt_path))
        model.to(device)

    #model.pretty_print()
    if rank == 0:
        print(model)
    
    # data -- # TODO: move later
    dst = load_allava_text(model.tok, n=500000)
    dsv = load_allava_laion(model.tok, n=500000)
    dsr = load_recap(model.tok, n = 500000)

    # shuffle per rank before multiloader creation 
    #witness something crazy:
    def sort_batch_shuffle(data, winsize):
        data = sorted(data, key = lambda x: len(x['input_ids']))
        data = [data[winsize*i: winsize*(i+1)] for i in range(len(data)//winsize + 1)]
        random.shuffle(data)
        data = [x for xs in data for x in xs] #thanks stack
        return data

    dst.data = sort_batch_shuffle(dst.data, train_config.batch_size)
    dsv.data = sort_batch_shuffle(dsv.data, train_config.batch_size)
    dsr.data = sort_batch_shuffle(dsr.data, train_config.batch_size)

    dlt = DataLoader(dst, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tok), num_workers=1, pin_memory=True, shuffle=False)
    dlv = DataLoader(dsv, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tok), num_workers=1, pin_memory=True, shuffle=False)
    dsr = DataLoader(dsr, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tok), num_workers=1, pin_memory=True, shuffle=False)

    dl = MultiDataLoader(dlt, dlv, dsr, seed = 42) #each rank should see same modality

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
    print("training")
    c = train_config
    model.train()

    # recover rank 
    rank = int(os.getenv("LOCAL_RANK", 0))

    total_len = len(dl)
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
                    print(f"iter: {e+1}/{total_len}  loss: {loss.item():.2f}  lr: {scheduler.get_last_lr()[0]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*total_len/60):.2f}, {dt:.2f}s/it]")

            if wandb_run is not None:
                data = {'loss': loss.item(), 'ppl': 2**(loss.item()), 'lr': scheduler.get_last_lr()[0]}
                # all print :(
                wandb_run.log(data)

            # move later
            if (e+1)%c.save_steps == 0:
                if c.save_path is not None:
                    # llm 
                    if c.enable_peft:
                        print(f"we are about to save the PEFT modules to {c.save_path}")
                        model.llm.save_pretrained(f"{c.save_path}-step{e+1}")

                    # connector
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
                                    torch.save(cpu_state, f"{c.save_path}-connector-step{e+1}.pt")
                    else:
                        torch.save(model.connector.state_dict(), f"{c.save_path}-connector-step{e+1}.pt")
                        

    # training done
    dist.barrier()

    # save on epoch end 
    # TODO: add saving at save_steps
    if c.save_path is not None:
        # llm 
        if c.enable_peft:
            print(f"we are about to save the PEFT modules to {c.save_path}")
            model.llm.save_pretrained(f"{c.save_path}-step{e+1}")

        # connector
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
                        torch.save(cpu_state, f"{c.save_path}-connector-step{e+1}.pt")
        else:
            torch.save(model.connector.state_dict(), f"{c.save_path}-connector-step{e+1}.pt")
                        



if __name__ == "__main__":
    setup()

    model_config = BeniConfig(
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        #vision_name_or_path = "openai/clip-vit-large-patch14",
        text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        #text_name_or_path = "HuggingFaceTB/SmolLM-360M",
        #text_name_or_path = "microsoft/Phi-3-mini-4k-instruct",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        freeze = True,
        attn_implementation = "sdpa",
        img_size = {'height': 448, 'width': 448},
        r = 11,
    )

    train_config = TrainConfig(
        warmup_ratio = 0.03,
        batch_size = 8,
        gradient_accumulation_steps = 1,
        lr = 3e-05,
        weight_decay = 0.1,
        min_lr = 1e-05,
        grad_clip = 1.0,
        save_steps = 1200,
        log_steps = 1,
        save_path = './model_checkpoints/finetune/tinyllama1b-siglip400m-ft',
        ckpt_path = './model_checkpoints/tinyllama1b-2-siglip400m.pt',
        betas = [0.9, 0.95],
        fsdp=True,
        enable_peft=True,
    )
    fsdp_config = FSDPConfig(
            #transformer_cls=(LlamaDecoderLayer, CLIPEncoderLayer),
            transformer_cls=(LlamaDecoderLayer, SiglipEncoderLayer), # need this for weight tying: torch.nn.Embedding, # if we upscale images we can't fsdp the positional embeddings ?
    )
    wandb_config = WandbConfig(
        enable = True,
        project = "vlm",
        entity = "nnethercott",
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')

    kwargs = {'train_config': train_config, 'fsdp_config': fsdp_config, 'wandb_config': wandb_config, 'lora_config': lora_config}

    # launch train
    fsdp_main(model_config, **kwargs)

    cleanup()
