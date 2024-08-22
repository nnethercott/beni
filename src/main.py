import random
import functools
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List 
import copy
import json

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

# fsdp layers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from model.vision import *

from peft import LoraConfig, get_peft_model, PeftModel

# local 
from data import *
from policies.wrapping import fsdp_auto_wrap_policy, get_llama_wrapper
from utils.train_utils import clear_gpu_cache, setup_environ_flags, get_cosine_schedule_with_warmup
from configs import TrainConfig, WandbConfig, FSDPConfig
from checkpointing import save_model, load_model

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
    rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    seed_everything(rank)

    model_config_copy = copy.deepcopy(asdict(model_config))
    _ = model_config_copy.pop("text_config") 
    _ = model_config_copy.pop("vision_config")

    configs = {'model_config': model_config_copy, 'train_config': train_config, 'fsdp_config': fsdp_config, 'lora_config': lora_config}
    wandb_run = wandb_config.build_run(configs, rank==0)

    # setup each cuda device ('device' aliased to cuda:n)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        clear_gpu_cache(rank)
        setup_environ_flags(rank)

    if train_config.fsdp:
        # load on cpu:0 only
        if rank == 0:
            model = Beni(model_config) # cpu

            # write model config
            s = train_config.save_path
            if s is not None:
                if not os.path.exists(s):
                    os.makedirs(s, exist_ok=True)
                with open(f"{s}/model_config.json", 'w') as f:
                    f.write(json.dumps(model_config_copy))

            # load from checkpoint
            if train_config.ckpt_path is not None:
                print(f"loading state dict from {train_config.ckpt_path}...")
                model = load_model(model, train_config.ckpt_path, trainable=True)

        else:
            with torch.device("meta"):
                model = Beni(model_config)    
                
        if train_config.enable_peft:
            assert lora_config is not None, "Either disable `enable_peft` or provide a valid lora config!"
            model.llm = get_peft_model(model.llm, lora_config)

        if rank == 0:
            params = sum((p.numel() for p in model.parameters()))
            trainable = sum((p.numel() for p in model.parameters() if p.requires_grad)) 
            print(f'VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\n')


        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, fsdp_config.transformer_cls)
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

    else:
        model = Beni(model_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if train_config.ckpt_path is not None:
            model.load_state_dict(torch.load(train_config.ckpt_path))
        model.to(device)

    if rank == 0:
        print(model)
        print("trainable params:")
        for n,p in model.named_parameters():
            if p.requires_grad:
                print(n)
    
    # data -- # TODO: move later
    template = model.config.chat_template
    dsr = load_recap(model.tokenizer, n = 300000, skip=400000, template = template)
    dsl = load_allava_laion(model.tokenizer, n = 300000, template = template)
    dst = load_allava_text(model.tokenizer, n = 300000, template = template)

    # shuffle per rank before multiloader creation 
    #witness something crazy:
    def sort_batch_shuffle(data, winsize):
        data = sorted(data, key = lambda x: len(x['input_ids']))
        data = [data[winsize*i: winsize*(i+1)] for i in range(len(data)//winsize + 1)]
        random.shuffle(data)
        data = [x for xs in data for x in xs] #thanks stack
        return data

    dsr.data = sort_batch_shuffle(dsr.data, train_config.batch_size)
    dsl.data = sort_batch_shuffle(dsl.data, train_config.batch_size)
    dst.data = sort_batch_shuffle(dst.data, train_config.batch_size)

    dsr = DataLoader(dsr, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tokenizer), num_workers=1, pin_memory=True, shuffle=False)
    dsl = DataLoader(dsl, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tokenizer), num_workers=1, pin_memory=True, shuffle=False)
    dst = DataLoader(dst, batch_size = train_config.batch_size, collate_fn = functools.partial(sft_collate_fn, tok=model.tokenizer), num_workers=1, pin_memory=True, shuffle=False)
    dl = MultiDataLoader(dsr, dsl, dst, seed = 42) #each rank should see same modality

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=train_config.betas, 
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                int(train_config.warmup_ratio*len(dl)*train_config.n_epochs),
                                                len(dl)*train_config.n_epochs,
                                                min_lr = train_config.min_lr,
                                                )

    # launch train
    train(model, optimizer, scheduler, dl, train_config, wandb_run=wandb_run)


def train(model, optimizer, scheduler, dl, train_config, wandb_run=None):
    print("training")
    c = train_config
    model.train()

    # recover rank 
    rank = int(os.getenv("LOCAL_RANK", 0))
    offset = 16710

    total_len = len(dl)
    start_time = time.time()
    for i in range(c.n_epochs):
        for e, batch in enumerate(dl):
            tik = time.time()
            if batch is None:  
                continue 

            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to("cuda") # assumes gpu training

            # forward pass
            out = model(**batch, output_attentions=False) 
            llm_loss = out['loss']
            vis_loss = model.vision.get_losses()
            
            loss = llm_loss
            if vis_loss is not None:
                loss += vis_loss  # weight lambdas defined in plugins; vis_loss is rly lam*vis_loss

            loss = loss / c.gradient_accumulation_steps
            loss.backward()

            if (e+1)%c.gradient_accumulation_steps==0:
                if c.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            if (e+1)%c.log_steps == 0: 
                dt = time.time()-tik 

                if os.environ["LOCAL_RANK"] == '0':
                    print(f"iter: {e+1}/{total_len}  loss: {c.gradient_accumulation_steps*loss.item():.2f}  lr: {scheduler.get_last_lr()[0]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*total_len/60):.2f}, {dt:.2f}s/it]")

            if wandb_run is not None:
                data = {'loss': llm_loss.item(), 'ppl': 2**(llm_loss.item()), 'lr': scheduler.get_last_lr()[0], }
                #_data = {'p_drop': model.vision.p, 'vis_loss': vis_loss.item()*c.gradient_accumulation_steps}
                #data = {**data, **_data}

                if rank == 0:
                    wandb_run.log(data)

            if (e+1)%c.save_steps == 0:
                save_dir = os.path.join(c.save_path, f"step{e+1}")
                save_model(model, save_dir=save_dir, rank=rank, fsdp_checkpoint_type = fsdp_config.checkpoint_type) # will break if fsdp config is none

    # training done
    dist.barrier()

    # save on epoch end 
    save_dir = os.path.join(c.save_path, f"step{total_len}")
    save_model(model, save_dir=save_dir, rank=rank, fsdp_checkpoint_type = fsdp_config.checkpoint_type)

if __name__ == "__main__":
    setup()
    perceiver_config = PerceiverResamplerConfig(
        hidden_size = 1152, # from siglip.config
        depth = 1, 
        n_latents = 64,
        n_query_groups=1,
        n_heads = 16,
        head_dim = 96,
        concat_latents_kv = True,
        attention_dropout = 0.0,
    )

    model_config = BeniConfig(
        perceiver_config = None,
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        #text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        text_name_or_path = "HuggingFaceTB/SmolLM-360M-Instruct",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        freeze = True,
        attn_implementation = "sdpa",
        img_size = 384,
        r = 9, 
        feature_select_index = -1,
        use_cls = True, # use first token with siglip
        #sparsity_plugins = [GumbelConfig(hidden_size=1152, temperature=0.1, p=0.5, lam=1.0)],
        sparsity_plugins = None,
        chat_template = "<|im_start|>user\n{prompt}<|im_end|>assistant\n", # smol chat template
    )

    train_config = TrainConfig(
        warmup_ratio = 0.03,
        batch_size = 8,
        gradient_accumulation_steps = 4,
        lr = 4e-04,
        weight_decay = 0.1,
        min_lr = 1e-05,
        grad_clip = 1.0,
        save_steps = 900,
        log_steps = 1,
        ckpt_path = "./model_checkpoints/smol-360M/step12500/",
        save_path = "./model_checkpoints/smol-360M-instruct/",
        betas = [0.9, 0.999],
        fsdp=True,
        enable_peft=True,
    )

    # need this for weight tying: torch.nn.Embedding, # if we upscale images we can't fsdp the positional embeddings ?
    fsdp_config = FSDPConfig(
            transformer_cls=(LlamaDecoderLayer, SiglipEncoderLayer, PerceiverResampler, nn.Embedding), 
    )
    wandb_config = WandbConfig(
        enable = True,
        project = "smol",
        entity = "nnethercott",
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')
    #lora_config = None

    kwargs = {'train_config': train_config, 'fsdp_config': fsdp_config, 'wandb_config': wandb_config, 'lora_config': lora_config}

    # launch train
    fsdp_main(model_config, **kwargs)

    cleanup()
