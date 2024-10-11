import functools
from typing import Optional
import os
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from accelerate import FullyShardedDataParallelPlugin
from peft import get_peft_model, LoraConfig, PeftModel
from transformers.trainer import _is_peft_model

from model.vision import *
from data import load_allava_laion, sft_collate_fn
from policies import fsdp_auto_wrap_policy, fpSixteen, bfSixteen
from model import VLM, VLMConfig, VisionTowerConfig

TOKEN = os.getenv("HF_TOKEN", None)

ATTN_IMPLEMENTATION = "eager"
SHARDING_STRATEGY = {
    "no_shard": ShardingStrategy.NO_SHARD,
    "full_shard": ShardingStrategy.FULL_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}
MIXED_PRECISION_POLICIES = {
    "fp16": fpSixteen,
    "bf16": bfSixteen,
}


@dataclass
class FSDPConfig:
    fsdp_enable: bool = True
    fsdp_activation_checkpointing: bool = False
    fsdp_cpu_offload: bool = False
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, no_shard, hybrid_shard


@dataclass
class LoraConfigWrapper:
    lora_target_modules: str = ""
    lora_r: int = 4
    lora_bias: str = "none"
    lora_alpha: int = 32
    enable_peft: bool = False

    def into(self):
        lora_dict = asdict(self)
        lora_dict["lora_target_modules"] = lora_dict["lora_target_modules"].split(",")
        return lora_dict


class MyTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        lora_config = kwargs.pop("lora_config", None)

        super().__init__(*args, **kwargs)
        self.lora_config = lora_config


# TODO: add neftune registering here
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add post init hooks here
        self.maybe_load_peft()

        if self.args.local_rank == 0:
            print(self.model)
            params = sum((p.numel() for p in self.model.parameters()))
            trainable = sum(
                (p.numel() for p in self.model.parameters() if p.requires_grad)
            )
            print(
                f"VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\n"
            )

    def maybe_load_peft(self):
        # TODO: add check for vision peft as well
        if self.args.lora_config is not None:
            self.model.llm = get_peft_model(self.model.llm, self.args.lora_config)

    def get_train_dataloader(self):
        # TODO: override this with multidataloader and write new dataset object
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=functools.partial(sft_collate_fn, tok=model.tokenizer),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # loads on all devices
        connector_state_dict = self.model.connector.state_dict()

        if self.args.local_rank == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # self.model.config.save_pretrained(output_dir)
            print(f"saving connector to {output_dir}/connector.pt")
            torch.save(connector_state_dict, os.path.join(output_dir, "connector.pt"))

        if isinstance(self.model.llm, PeftModel):
            if self.args.local_rank == 0:
                print(f"saving loras to {output_dir}/llm_loras")

            self.model.llm.save_pretrained(
                os.path.join(output_dir, "llm_loras"),
                save_embedding_layers=False,  # TODO: change this later if we resize
            )


def get_model():
    # some additional tokens
    special_tokens = ["<|vision_start|>", "<|vision_end|>"]
    special_tokens += ["<|box_start|>", "<|box_end|>"]  # change in datasets
    special_tokens += [f"0.{i}" for i in range(10)]
    special_tokens += [f"0.{i}{j}" for i in range(10) for j in range(1, 10)]
    special_tokens += ["1.0"]

    vision_tower_config = VisionTowerConfig(
        r=9,
        feature_select_index=-1,
        use_cls=True,
        img_size=384,
        grid=(1, 1),
        use_global_crop=False,
    )
    model_config = VLMConfig(
        vision_name_or_path="google/siglip-so400m-patch14-384",
        text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # text_name_or_path="Qwen/Qwen2-0.5B",
        vision_tower_config=vision_tower_config,
        vision_cls="SiglipVisionModel",
        freeze=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        unfreeze_lm_head=False,
        # special_tokens=[],
    )
    model = VLM(model_config, hf_token=TOKEN)
    return model


if __name__ == "__main__":
    # parse from env
    parser = HfArgumentParser(
        (MyTrainingArguments, FSDPConfig, LoraConfigWrapper, VisionTowerConfig)
    )
    train_config, fsdp_config, lora_config, vision_tower_config = (
        parser.parse_args_into_dataclasses()
    )

    # inject some stuff
    train_config.local_rank = dist.get_rank()

    if lora_config.enable_peft:
        lora_config = LoraConfig(
            r=lora_config.lora_r,
            target_modules=lora_config.into()["lora_target_modules"],  # to dict rq
            bias=lora_config.lora_bias,
            lora_alpha=lora_config.lora_alpha,
        )
        # add to train
        train_config.lora_config = lora_config

    model = get_model()

    # update internals
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(
        model,
        (
            SiglipEncoderLayer,
            Qwen2DecoderLayer,
            LlamaDecoderLayer,
            # nn.Embedding,
        ),
    )

    if train_config.fp16:
        precision = MIXED_PRECISION_POLICIES["fp16"]
    elif train_config.bf16:
        precision = MIXED_PRECISION_POLICIES["bf16"]
    else:
        precision = None

    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy=my_auto_wrapping_policy,
        sharding_strategy=SHARDING_STRATEGY[fsdp_config.fsdp_sharding_strategy],
        limit_all_gathers=True,
        use_orig_params=False,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        ),
        activation_checkpointing=fsdp_config.fsdp_activation_checkpointing,
        sync_module_states=True,
        # for saving the state dict all gather on cuda:0 and offload to cpu
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=dist.fsdp.fully_sharded_data_parallel.FullStateDictConfig(
            offload_to_cpu=False, rank0_only=True
        ),
        mixed_precision_policy=precision,
    )

    # data -- change to a bucket link ?
    model_config = model.config
    it = model_config.instruction_template
    rt = model_config.response_template

    train_dataset = load_allava_laion(
        model.tokenizer,
        n=200,
        instruction_template=it,
        response_template=rt,
    )

    if dist.get_rank() == 0:
        print(train_config)

    trainer = MyTrainer(
        model=model,
        tokenizer=model.tokenizer,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    # https://huggingface.co/docs/peft/en/accelerate/fsdp
    if fsdp_config.fsdp_enable:
        trainer.accelerator.state.fsdp_plugin = fsdp_plugin

    trainer.train()
