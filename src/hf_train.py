import functools
import os
import time
import json

import torch
import torch.distributed as dist
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

from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

# new
from trl import SFTTrainer
from accelerate import FullyShardedDataParallelPlugin, Accelerator

from model.vision import *
from data import *
from policies.wrapping import fsdp_auto_wrap_policy, get_llama_wrapper
import policies
from utils.train_utils import (
    clear_gpu_cache,
    setup_environ_flags,
    get_cosine_schedule_with_warmup,
)
from configs import TrainConfig, WandbConfig, FSDPConfig
from checkpointing import save_model, load_model
from model import Beni, BeniConfig

TOKEN = os.getenv("HF_ACCESS_TOKEN", None)

model_config = BeniConfig(
    perceiver_config=None,
    vision_name_or_path="google/siglip-so400m-patch14-384",
    text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    vision_cls="SiglipVisionModel",
    vision_processor_cls="SiglipImageProcessor",
    freeze=True,
    attn_implementation="eager",
    img_size=384,
    r=9,
    feature_select_index=-1,
    use_cls=True,
)
model = Beni(model_config, hf_token=TOKEN)
print(model)

# NOTE: we can do the llava-style custom save hook and use standard torch code for saving ckpts

training_args = TrainingArguments(
    output_dir="./hf_test",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=4e-04,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    max_grad_norm=1.0,
    num_train_epochs=1,
    lr_scheduler_type="cosine_with_min_lr",  # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/trainer_utils.py#L410
    lr_scheduler_kwargs={"min_lr": 4e-05},
    warmup_ratio=0.03,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="no",  # steps
    # save_steps = 10,
    # save_only_model=True,
    seed=42,
    # local_rank = -1, # don't know what to put here
    dataloader_num_workers=1,
    label_names=["labels"],
    fsdp=True,
    optim="adamw_torch",
    report_to="none",
    dataloader_pin_memory=True,
)

my_auto_wrapping_policy = fsdp_auto_wrap_policy(
    model,
    (
        LlamaDecoderLayer,
        SiglipEncoderLayer,
        PerceiverResampler,
        nn.Embedding,
        VisionTower,
    ),
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=dist.fsdp.fully_sharded_data_parallel.FullStateDictConfig(
        offload_to_cpu=False, rank0_only=True
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    state_dict_type=StateDictType.FULL_STATE_DICT,
    limit_all_gathers=True,
    use_orig_params=False,
    param_init_fn=lambda module: module.to_empty(
        device=torch.device("cuda"), recurse=False
    ),
    auto_wrap_policy=my_auto_wrapping_policy,
    sync_module_states=True,
    activation_checkpointing=False,
)


def formatting_prompts_func(example):
    return example["text"]


train_dataset = load_recap(
    model.tokenizer,
    n=1000,
)


class MyTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=functools.partial(sft_collate_fn, tok=model.tokenizer),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


trainer = MyTrainer(
    model=model,
    tokenizer=model.tokenizer,
    args=training_args,
    # max_seq_length=512,
    train_dataset=train_dataset,
    eval_dataset=None,
    # formatting_func=formatting_prompts_func,
    # callbacks=[EfficiencyCallback()],
)
# https://huggingface.co/docs/peft/en/accelerate/fsdp
trainer.accelerator.state.fsdp_plugin = fsdp_plugin

trainer.train()
input()
