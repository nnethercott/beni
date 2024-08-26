import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Callable
import functools
import random
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
import io

"""
NOTES:
    * should sort samples by length and guesstimate boundaries based on # tokens 
        * e.g. batches of varying sizes depending on number of total tokens 

IDEA:
    * stateful DataLoader subclass which modifies batch size based on last seen sequence length of an ordered dataset
        * can adjust batch size dynamically for us
"""


# torch.utils.data.Dataset subclass
class CustomDataset(Dataset):
    def __init__(self, data):
        # TODO: sorted(data, key=lambda x: len(x['input_ids']))
        if torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            block_size = len(data) // world_size
            self.data = data[
                rank * block_size : (rank + 1) * block_size
            ]  # drop len(data)%block_size samples
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# TODO: use this to download images instead of dynamic get
class CustomDatasetForImages(CustomDataset):
    def __init__(self, data, cache=True, request_timeout: float = 3.0):
        super().__init__(data)
        self.offset = 0
        self.cache = cache
        self.timeout = request_timeout

        self.build()
        self.trim()

    def get_image(self, url):
        try:
            return Image.open(
                io.BytesIO(requests.get(url, timeout=self.timeout).content)
            ).convert("RGB")
        except:
            raise RuntimeError(f"image: {url} could not be downloaded !")

    # @override
    def __getitem__(self, i):
        if self.cache:
            return self.data[i]

        img = self.get_image(i)
        return {**self.data[i], "images": img}

    def build(self):
        urls_enumed = [(e, d["url"]) for e, d in enumerate(self.data)]

        def download(idx, url):
            try:
                img = self.get_image(url)
                if not len(img.size) == 2 or img.size[:2] <= (1, 1):  # edge cases
                    return (idx, None)

                if self.cache:
                    self.data[idx] = {**self.data[idx], "images": img}
                return (idx, img)
            except:
                return (idx, None)

        with ThreadPoolExecutor(
            max_workers=int(os.environ.get("OMP_NUM_THREADS", 16))
        ) as executor:
            res = list(
                tqdm(
                    executor.map(lambda args: download(*args), urls_enumed),
                    total=len(urls_enumed),
                )
            )

        # remove bad ids from data
        # nice leet code stuff
        bad_ids = [p[0] for p in res if p[1] is None]
        bad_ids = sorted(bad_ids)[::-1]
        for idx in bad_ids:
            self.data.pop(idx)

        return self

    def trim(self):
        if torch.distributed.is_initialized():
            # broadcast self.data lengths and choose min
            pass


# no build at init
class LazyCustomDatasetForImages(CustomDatasetForImages):
    def __init__(self, data, request_timeout: float = 3):
        CustomDataset.__init__(self, data)
        self.timeout = request_timeout

    def __getitem__(self, idx):
        try:
            img = self.get_image(self.data[idx]["url"])
            if len(img.size) > 2 or img.size[:2] == (1, 1):
                img = None
        except:
            img = None
        return {**self.data[idx], "images": img}


# can use this for diverse batch sizes in normal dataset ordered by context length too
# can nest these as well
class MultiDataLoader:
    """
    random sample between several dataloaders, allowing for multimodality training
    """

    def __init__(self, *loaders, **kwargs):
        self.loaders = list(loaders)
        self.loader_iters = [iter(loader) for loader in self.loaders]
        self.loader_lens = [len(loader) for loader in self.loaders]
        self.buffer = list(range(len(self.loaders)))
        self.weights = [l / sum(self.loader_lens) for l in self.loader_lens]

        # hacky
        seed = kwargs.get("seed", None)
        if seed:
            self.sampler = random.Random(seed)
        else:
            self.sampler = None

    def __len__(self):
        return sum(self.loader_lens)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:  # If the buffer is empty, all loaders are exhausted
            raise StopIteration

        if self.sampler is not None:
            idx = self.sampler.choices(self.buffer, weights=self.weights, k=1)[0]
        else:
            idx = random.choices(self.buffer, weights=self.weights, k=1)[0]

        try:
            batch = next(self.loader_iters[idx])
            self.loader_lens[idx] -= 1

            if self.loader_lens[idx] == 0:
                # remove data loader
                self.buffer.pop(idx)
                self.loader_iters.pop(idx)
                self.loader_lens.pop(idx)
                self.weights.pop(idx)

                # fix buffer indices
                self.buffer = list(range(len(self.loader_iters)))

            return batch

        except StopIteration:
            # should be gone already, but if there remove it and attempt to fetch from another loader
            self.buffer.remove(idx)
            return self.__next__()


def sft_collate_fn(inputs, tok):
    """
    dynamically pads input tensors and constructs attn mask
    """
    # print(inputs)
    # remove samples which we couldn't download image for
    bad_ids = [
        i for i, sample in enumerate(inputs) if sample.get("images", True) is None
    ]
    bad_ids = bad_ids[::-1]
    for i in bad_ids:
        inputs.pop(i)

    if len(inputs) == 0:
        return None

    # LD -> DL
    inputs = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    input_ids = inputs["input_ids"]
    prompt_len = inputs["prompt_len"]

    # needed for masking PAD loss in train
    seq_len = [len(i) for i in input_ids]
    max_len = max(seq_len)

    # pad inputs and create attention mask
    input_ids_t = [
        torch.tensor(i + [tok.eos_token_id] * (max_len - len(i))).unsqueeze(0)
        for i in input_ids
    ]
    input_ids_t = torch.cat(input_ids_t, 0)

    pos = torch.arange(max_len).unsqueeze(0).repeat((input_ids_t.shape[0], 1))
    seq_end = (
        torch.tensor([len(i) for i in input_ids]).unsqueeze(1).repeat((1, max_len))
    )
    prompt_end = torch.tensor(prompt_len).unsqueeze(1).repeat((1, max_len))
    attn_mask = pos < seq_end

    # TODO: zero out loss on prompt tokens for labels
    labels_t = input_ids_t.clone()
    labels_t.masked_fill_(attn_mask == 0, -100)

    if prompt_len != [0] * len(prompt_len):
        labels_t.masked_fill_(pos < prompt_end, -100)

    return {
        "input_ids": input_ids_t,
        "attention_mask": attn_mask.to(torch.long),
        "labels": labels_t,
        "images": inputs.get("images", None),
        "url": inputs.get("url"),  # debug
    }


def apply_chat_template(
    data: datasets.Dataset,
    tok,
    ctx_len: int = 384,
    truncate: bool = False,
    instruction_template=None,
    response_template=None,
):
    """
    Arguments:
        - data: lazy dataset loaded with `streaming=True`. dataset should have column 'response'. if no 'prompt' present we use ''
        - tok: tokenizer
        - template: optional syntaxing
        - n: how many samples to load
        - preprocess: optional preprocessing mapped before the boilerplate
        - ctx_len: max length for input ids
        - trim: if true we truncate samples exceeding max length, else we filter them out

    features:
        - adds EOS token to the end of each text entry -> formats like SOS+encoded+EOS
            - ensures free lunch with pure-text datasets, image ones handled by vlm
    """
    if instruction_template is None:
        instruction_template = (
            tok.bos_token + "{prompt}" + tok.eos_token
        )  # <s>prompt</s>
    if response_template is None:
        response_template = "{response}" + tok.eos_token

    def process(samples):
        prompts = [instruction_template.format(prompt=p) for p in samples["prompt"]]
        responses = [response_template.format(response=r) for r in samples["response"]]
        inputs = [p + r for p, r in zip(prompts, responses)]

        prompt_len = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]

        if truncate:
            input_ids = [
                tok.encode(i, add_special_tokens=False)[:ctx_len] for i in inputs
            ]
        else:
            input_ids = [tok.encode(i, add_special_tokens=False) for i in inputs]

        samples["input_ids"] = input_ids
        samples["prompt_len"] = prompt_len

        return samples

    # map
    data = data.map(
        process,
        batched=True,
    )

    # filter
    if not truncate:
        data = data.filter(
            lambda x: [len(y) <= ctx_len for y in x["input_ids"]], batched=True
        )

    # loadable in torch dataloader
    return data
