import random
import functools
import os
import time

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
import datasets


# previous train repo 
import sys 
sys.path.insert(1, "./train/utils")
from fsdp_utils import fsdp_auto_wrap_policy #this adds any peft child classes


# make this a constant 
def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


#<|FSDP utils|>
def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def llama_autowrap_policy():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            (LlamaDecoderLayer, )
        },
    )
    return auto_wrap_policy


def fsdp_main():
    seed_everything(0)

    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    # setup each cuda device ('device' aliased to cuda:n)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)

    # load model & tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    
    if rank == 0:
        model = AutoModelForCausalLM.from_config(
            config,
        )
        # ok 
        model.load_state_dict(torch.load("tinyllama0.pt"))
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)

    #my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, (LlamaDecoderLayer, ))
    my_auto_wrapping_policy = llama_autowrap_policy()

    model = FSDP(
        model,
        auto_wrap_policy= my_auto_wrapping_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, #SHARD_GRAD_OP
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False) if rank != 0 else None,
    )

    #if fsdp_config.fsdp_activation_checkpointing:
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
    train(model, tok, optimizer, rank)


def load_data(tok):
    class N(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    data = datasets.load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    data = data.take(1000)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))
    data = data.map(
        lambda x: {
            **x,
            "input_ids": [
                tok.encode(
                    y
                )
                + [tok.eos_token_id]
                for y in x["content"]
            ],
        },
        batched=True,
    )
    data = data.filter(lambda x: [len(y) <= 512 for y in x["input_ids"]], batched=True)

    data = [{"prompt_len": 0, "input_ids": i} for i in data["input_ids"]]
    ds = N(data)

    return ds

def sft_collate_fn(inputs, tok):
    # LD -> DL
    inputs = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    input_ids = inputs["input_ids"]
    prompt_len = inputs["prompt_len"]

    # needed for masking PAD loss in train
    seq_len = [len(i) for i in input_ids]
    max_len = max(seq_len)

    # pad inputs and create attention mask
    input_ids_t = [
        torch.tensor(i + [tok.pad_token_id] * (max_len - len(i))).unsqueeze(0)
        for i in input_ids
    ]
    input_ids_t = torch.cat(input_ids_t, 0)

    pos = torch.arange(max_len).unsqueeze(0).repeat((input_ids_t.shape[0], 1))
    seq_end = (
        torch.tensor([len(i) for i in input_ids]).unsqueeze(1).repeat((1, max_len))
    )
    attn_mask = (pos < seq_end).to(dtype=torch.float32)

    labels = input_ids_t.masked_fill(attn_mask == 0, -100)

    return {
        "input_ids": input_ids_t,
        "attention_mask": attn_mask,
        "labels": labels,
    }

def train(model, tok, optimizer, rank):
    seed_everything(rank)

    model.train()
    dl = DataLoader(load_data(tok), batch_size=1, collate_fn = functools.partial(sft_collate_fn, tok = tok), shuffle=True)
    g = 4

    for e, batch in enumerate(dl):
        tik = time.time()
        for key in batch.keys():
            batch[key] = batch[key].to(rank)

        loss = model(**batch)['loss']
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

        if (e+1)%g==0:
            optimizer.step()
            optimizer.zero_grad()


        dt = time.time()-tik 
        shape = batch['input_ids'].shape
        dx = shape[0]
        if rank == 0 and e%g == 0:
            print(f'iter: [{e+1}/{len(dl)}] | loss: {g*loss.item():.3f} | samples/s: {dx/(dt+1e-05):3f}')
    
    # training done
    dist.barrier()
    with FSDP.state_dict_type(
            model, 
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, "tinyllama0.pt")


if __name__ == "__main__":
    setup()
    fsdp_main()
    cleanup()
