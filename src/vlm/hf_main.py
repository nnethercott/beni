import functools
from typing import Optional
import os
from dataclasses import asdict, dataclass
import torch
import json
from torch import mode, nn
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
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
from accelerate import FullyShardedDataParallelPlugin
from peft import get_peft_model, LoraConfig, PeftModel
from transformers.trainer import _is_peft_model

from model.vision import *
import data
from data import load_allava_laion, sft_collate_fn, prepare
import data.datasets as local_datasets
from model import VLM, VLMConfig, VisionTowerConfig

from logging import getLogger, Formatter, StreamHandler
import logging

log_formatter = Formatter("%(asctime)s [%(levelname)-8.8s] %(message)s")
console_handler = StreamHandler()
console_handler.setFormatter(log_formatter)
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

TOKEN = os.getenv("HF_TOKEN", None)
ATTN_IMPLEMENTATION = "eager"


# remove later in favor of bucket-based datasets
@dataclass
class VLMDataset:
    dataset_name: str
    split: str = "train"
    n_samples: int = 100

    def load(self):
        name = "load_" + self.dataset_name
        loader = getattr(data, name)
        return loader(n=self.n_samples, split=self.split)


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
# TODO: custom logger ?
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add post init hooks here
        self.maybe_load_peft()

        if self.args.local_rank == 0:
            # save model config and tokenizer
            with open(f"{self.args.output_dir}/config.json", "w") as f:
                f.write(json.dumps(self.model.config.to_dict()))

            self.model.tokenizer.save_pretrained(self.args.output_dir)

            # print model
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
            collate_fn=functools.partial(sft_collate_fn, tok=self.model.tokenizer),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            shuffle=True,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # loads on all devices
        connector_state_dict = self.model.connector.state_dict()

        if self.args.local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)

            # self.model.config.save_pretrained(output_dir)
            print(f"saving connector to {output_dir}/connector.pt")
            torch.save(connector_state_dict, os.path.join(output_dir, "connector.pt"))

        if isinstance(self.model.llm, PeftModel):
            if self.args.local_rank == 0:
                print(f"saving loras to {output_dir}/llm_loras")

            # NOTE: should make save_embedding_layers=True when adding new tokens
            self.model.llm.save_pretrained(
                os.path.join(output_dir, "llm_loras"),
                save_embedding_layers=False,
            )


def get_data(dataset_config, model_config, tokenizer):
    data = dataset_config.load()
    train_dataset = prepare(
        data,
        tokenizer,
        model_config.instruction_template,
        model_config.response_template,
    )
    return train_dataset


# is this okay?
def download_model_weights(llm: str, vit: str, rank: int = 0):
    if rank == 0:
        print(f"Downloading model weights for {llm} and {vit}...")
        try:
            os.system(f"huggingface-cli login --token={os.environ['HF_TOKEN']}")
        except:
            print("No huggingface token provided; only access to unrestricted models.")

        os.system(f"huggingface-cli download {llm}")
        os.system(f"huggingface-cli download {vit}")


def initialize_wandb(train_config, rank):
    import wandb

    if rank == 0:
        try:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            print("Wandb login successful !")
        except:
            print("no api key provided; setting `report_to`=none")
            train_config.report_to = "none"


def main():
    # parses from --args
    parser = HfArgumentParser(
        (
            MyTrainingArguments,
            VLMConfig,
            VisionTowerConfig,
            LoraConfigWrapper,
            VLMDataset,
        )
    )
    train_config, model_config, vision_tower_config, lora_config, dataset_config = (
        parser.parse_args_into_dataclasses()
    )

    RANK = dist.get_rank()
    train_config.local_rank = RANK

    if train_config.report_to == "wandb":
        initialize_wandb(train_config, RANK)

    # if on google machine
    if os.getenv("AIP_MODEL_DIR") is not None:
        gs_prefix = "gs://"
        gcsfuse_prefix = "/gcs/"
        output_dir = os.getenv("AIP_MODEL_DIR")
        output_dir = output_dir.replace(gs_prefix, gcsfuse_prefix)
        train_config.output_dir = output_dir

    # if lora
    if lora_config.enable_peft:
        lora_config = LoraConfig(
            r=lora_config.lora_r,
            target_modules=lora_config.into()["lora_target_modules"],  # to dict rq
            bias=lora_config.lora_bias,
            lora_alpha=lora_config.lora_alpha,
        )
        train_config.lora_config = lora_config

    # add some special tokens here before model init
    # special_tokens = ["<|vision_start|>", "<|vision_end|>"]
    # special_tokens += ["<|box_start|>", "<|box_end|>"]  # change in datasets
    # special_tokens += [f"0.{i}" for i in range(10)]
    # special_tokens += [f"0.{i}{j}" for i in range(10) for j in range(1, 10)]
    # special_tokens += ["1.0"]
    # model_config.special_tokens = special_tokens

    download_model_weights(
        llm=model_config.text_name_or_path,
        vit=model_config.vision_name_or_path,
        rank=RANK,
    )
    dist.barrier()

    model_config.vision_tower_config = vision_tower_config
    model = VLM(model_config)

    train_dataset = get_data(dataset_config, model_config, model.tokenizer)

    trainer = MyTrainer(
        model=model,
        tokenizer=model.tokenizer,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    if dist.get_rank() == 0:
        print(train_config)

    trainer.train()


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    # some stuff
    print(sys.argv)
    os.system("ifconfig")

    setup()
    main()
    cleanup()
