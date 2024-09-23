import functools
import os
import uuid

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
)

from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

# new
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model

from model.vision import *
from data import *
from policies.wrapping import fsdp_auto_wrap_policy
from configs import WandbConfig
from checkpointing import save_model
from model import Beni, BeniConfig

TOKEN = os.getenv("HF_TOKEN", None)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

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
    bos_token="<s><|user|>",
    instruction_template="<s><|user|>\n{instruction}</s>\n<|assistant|>\n",  # all tokens we want no loss on
    response_template="{response}</s>",
)
model = Beni(model_config, hf_token=TOKEN)
model.to(torch.float16)

# init wandb before trainer
wandb_config = WandbConfig(
    enable=False,
    project="vlm",
    entity="nnethercott",
    name=model_config.text_name_or_path.split("/")[-1].lower()
    + "-"
    + str(uuid.uuid1()).split("-")[-1],  # model-archi-uuid
)
_ = wandb_config.build_run()
os.environ["WANDB_PROJECT"] = wandb_config.project
os.environ["WANDB_NAME"] = wandb_config.name


class MyTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        lora_config = kwargs.pop("lora_config", None)

        super().__init__(*args, **kwargs)
        self.lora_config = lora_config


training_args = MyTrainingArguments(
    output_dir="/mnt/nate/beni/hf_test",
    per_device_train_batch_size=10,
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
    save_strategy="steps",  # steps
    save_steps=250,
    save_only_model=True,
    seed=42,
    local_rank=os.getenv("LOCAL_RANK", -1),
    dataloader_num_workers=1,
    label_names=["labels"],
    fsdp=True,
    optim="adamw_torch",
    report_to="wandb",
    run_name=os.environ["WANDB_NAME"],
    dataloader_pin_memory=True,
    fp16=True,
    half_precision_backend="auto",
    # lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none'),
)

# fsdp
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


# dataset(s)
it = model_config.instruction_template
rt = model_config.response_template

# train_dataset = load_recap(
#    model.tokenizer,
#    n=50000,
#    instruction_template=it,
#    response_template=rt,
# )

train_dataset = load_allava_laion(
    model.tokenizer,
    n=200000,
    instruction_template=it,
    response_template=rt,
)


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add post init hooks here
        self.maybe_load_peft()

        if self.args.local_rank == 0:
            params = sum((p.numel() for p in self.model.parameters()))
            trainable = sum(
                (p.numel() for p in self.model.parameters() if p.requires_grad)
            )
            print(
                f"VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\n"
            )
            print(self.model)

    def maybe_load_peft(self):
        if self.args.lora_config is not None:
            self.model.llm = get_peft_model(self.model.llm, self.args.lora_config)

    def get_train_dataloader(self):
        # todo; override this with multidataloader and write new dataset object
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

        save_model(
            self.model,
            save_dir=output_dir,
            rank=self.args.local_rank,
            fsdp_checkpoint_type=self.accelerator.state.fsdp_plugin.state_dict_type,
        )


trainer = MyTrainer(
    model=model,
    tokenizer=model.tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
)

# https://huggingface.co/docs/peft/en/accelerate/fsdp
trainer.accelerator.state.fsdp_plugin = fsdp_plugin

trainer.train()
