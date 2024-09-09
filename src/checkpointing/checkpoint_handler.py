import os
from peft import PeftModel

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)


def load_model(model, ckpt_dir, trainable: bool = True):
    """
    replaces model weights from those stored in  ckpt_dir
    """
    assert os.path.exists(ckpt_dir), f"checkpoint directory {ckpt_dir} not found!"
    weight_paths = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]

    for f in weight_paths:
        if "connector.pt" in f:
            print("loading connector...")
            model.connector.load_state_dict(torch.load(f))
        elif "loras" in f:
            print("loading loras...")
            model.llm = PeftModel.from_pretrained(
                model.llm, f, is_trainable=trainable
            )  # `is_trainable` is for training
            model.llm = model.llm.merge_and_unload()
        else:
            pass

    return model


def save_model(model, save_dir, rank):
    """
    checkpointing for fsdp training
    """
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # loras
    if isinstance(model.llm, PeftModel):
        if rank == 0:
            print(f"saving LoRAs to {save_dir}...")
        model.llm.save_pretrained(f"{save_dir}/loras")

    # connector
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.connector.state_dict()

        if rank == 0:
            print(f"\nSAVING CHECKPOINT TO: {save_dir} ...\n")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(cpu_state, f"{save_dir}/connector.pt")
