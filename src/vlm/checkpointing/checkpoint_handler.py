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

        # loras
        elif "vision_loras" in f:
            print("loading vision loras...")
            model.vision = PeftModel.from_pretrained(
                model.vision, f, is_trainable=trainable
            )  # `is_trainable` is for training

            if merge_and_unload:
                model.vision = (
                    model.vision.merge_and_unload()
                )  # NOTE: potentially problematic in training

        elif "llm_loras" in f:
            print("loading text loras...")
            model.llm = PeftModel.from_pretrained(
                model.llm, f, is_trainable=trainable
            )  # `is_trainable` is for training

            if merge_and_unload:
                model.llm = (
                    model.llm.merge_and_unload()
                )  # NOTE: potentially problematic in training

        elif "lm_head.pt" in f:
            print("loading lm_head...")
            model.llm.lm_head.load_state_dict(torch.load(f))

    return model


def save_model(model, save_dir, rank):
    """
    checkpointing for fsdp training
    """
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # loras
    if isinstance(model.vision, PeftModel):
        if rank == 0:
            print(f"SAVING VISION LoRAs CHECKPOINT TO {save_dir}...")
        model.vision.save_pretrained(f"{save_dir}/vision_loras")

    if isinstance(model.llm, PeftModel):
        if rank == 0:
            print(f"SAVING TEXT LoRAs CHECKPOINT TO {save_dir}...")
        model.llm.save_pretrained(f"{save_dir}/llm_loras")

    # connector
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = model.connector.state_dict()

        if rank == 0:
            print(f"SAVING CONNECTOR CHECKPOINT TO: {save_dir}")
            torch.save(cpu_state, f"{save_dir}/connector.pt")

        # if model.config.vision_tower_config.perceiver_config is not None:
        #     perceiver_cpu_state = model.vision.resampler.state_dict()

        #     if rank == 0:
        #         print(f"SAVING PERCEIVER CHECKPOINT TO: {save_dir} ...")
        #         os.makedirs(save_dir, exist_ok=True)
        #         torch.save(perceiver_cpu_state, f"{save_dir}/resampler.pt")

        if model.config.unfreeze_lm_head:
            lm_head_cpu_state = model.llm.lm_head.state_dict()
            if rank == 0:
                print(f"SAVING LM_HEAD CHECKPOINT TO: {save_dir} ...")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(lm_head_cpu_state, f"{save_dir}/lm_head.pt")
