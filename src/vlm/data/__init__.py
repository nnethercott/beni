import datasets
import functools
from peft.tuners.p_tuning import model
from torch.utils.data import DataLoader
import torch.distributed as dist
from .utils import (
    StreamingDataset,
    apply_chat_template,
    CustomDataset,
    LazyCustomDatasetForImages,
    sft_collate_fn,
    MultiDataLoader,
)
from .d import (
    load_lmms_lab_recap,
    load_recap,
    load_allava_laion,
    load_allava_text,
    load_gpt4v_long,
    load_gpt4v_short,
    load_synthdog,
    load_synthdog_nate,
    load_textocr,
)
from .minhash import *
import random


# train dataloader
def _get_train_dataloader(tokenizer, model_config, train_config):
    instruction_template = model_config.instruction_template
    response_template = model_config.response_template

    # for data replay
    # recap = load_lmms_lab_recap(
    #     tokenizer,
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    # )
    # recap = StreamingDataset(
    #     recap,
    #     batch_size=train_config.batch_size,
    #     n=20000,
    #     skip=50000,
    #     collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
    # )

    laion_instruct = load_allava_laion(
        tokenizer,
        split="instruct",
        n=1000,
        instruction_template=instruction_template,
        response_template=response_template,
    )
    # laion_caption = load_allava_laion(
    #     tokenizer,
    #     split="caption",
    #     n=500,
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    # )

    train_datasets = [laion_instruct]

    # optimized ordering of samples for uniform seq lengths
    def sort_batch_shuffle(data, winsize):
        data = sorted(data, key=lambda x: len(x["input_ids"]))
        data = [
            data[winsize * i : winsize * (i + 1)]
            for i in range(len(data) // winsize + 1)
        ]
        random.shuffle(data)
        data = [x for xs in data for x in xs]  # thanks stack
        return data

    for d in train_datasets:
        d.data = sort_batch_shuffle(d.data, train_config.batch_size)

    # multidataloader construct
    loaders = [
        DataLoader(
            d,
            batch_size=train_config.batch_size,
            collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )
        for d in train_datasets
    ]

    # add StreamingDatasets
    # loaders += [recap]

    dl = MultiDataLoader(
        *loaders, seed=42
    )  # each rank should see same modality when seed=int (but not when we use rank info)

    return dl


def get_train_dataloader(tokenizer, model_config, train_config):
    seed = 42 + dist.get_rank() if dist.is_initialized() else 42  # hope this helps
    print(f"seeding shuffle with seed: {seed}")

    # data = load_synthdog(
    #     tokenizer,
    #     instruction_template=model_config.instruction_template,
    #     response_template=model_config.response_template,
    # )
    # data = data.shuffle(seed=seed, buffer_size=1000)
    # data = StreamingDataset(
    #     data,
    #     n=50000,
    #     # skip=8000 * 2,
    #     batch_size=train_config.batch_size,
    #     collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
    # )

    # cropped and grounding
    nate_data = load_synthdog_nate(
        tokenizer,
        instruction_template=model_config.instruction_template,
        response_template=model_config.response_template,
    )
    nate_data = nate_data.shuffle(seed=seed, buffer_size=1000)
    nate_data = StreamingDataset(
        nate_data,
        n=50000,
        # skip=0,
        batch_size=train_config.batch_size,
        collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
    )

    # random seed
    dl = MultiDataLoader(nate_data, seed=seed)

    return dl


def get_eval_dataloader(tokenizer, model_config, train_config):
    # use synthdog as an eval set
    data = load_synthdog(
        tok=tokenizer,
        split="validation",
        n=32,
        instruction_template=model_config.instruction_template,
        response_template=model_config.response_template,
    )

    # # consume
    # def dataset_generator(dataset):
    #     yield from dataset

    # data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    dl = DataLoader(
        data,
        batch_size=train_config.batch_size,
        collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )

    return dl
