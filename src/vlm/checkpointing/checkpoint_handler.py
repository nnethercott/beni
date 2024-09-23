import os
from peft import PeftModel

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)


def load_model(model, ckpt_dir, trainable: bool = True, merge_and_unload: bool = False):
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
            model.vision = PeftModel.from_pretrained(
                model.vision, f, is_trainable=trainable
            )  # `is_trainable` is for training

            if merge_and_unload:
                model.vision = (
                    model.vision.merge_and_unload()
                )  # NOTE: potentially problematic in training
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
    # if isinstance(model.llm, PeftModel):
    if isinstance(model.vision, PeftModel):
        if rank == 0:
            print(f"SAVING LoRAs CHECKPOINT TO {save_dir}...")
        model.vision.save_pretrained(f"{save_dir}/loras")

    # connector
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.connector.state_dict()

        if rank == 0:
            print(f"SAVING CONNECTOR CHECKPOINT TO: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(cpu_state, f"{save_dir}/connector.pt")

        if model.config.vision_tower_config.perceiver_config is not None:
            perceiver_cpu_state = model.vision.resampler.state_dict()

            if rank == 0:
                print(f"SAVING PERCEIVER CHECKPOINT TO: {save_dir} ...")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(perceiver_cpu_state, f"{save_dir}/resampler.pt")
