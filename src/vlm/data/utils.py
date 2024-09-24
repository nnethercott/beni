import datasets
import torch
from torch.utils.data import Dataset
import os
import random
from urllib.parse import urlparse
from transformers import AutoTokenizer
from typing import List
import requests
from PIL import Image
import io


from .minhash import MinHashLSHDeduplicator


# or use wordllama: https://huggingface.co/dleemiller/word-llama-l2-supercat
def fuzzy_filter(*datasets, tokenizer: AutoTokenizer):
    """
    use dataset.data so we avoid loading images
    """
    datasets = list(datasets)

    # for some reason dataset.Dataset.to_list() converts pil images to bytes so we need
    # to update the CustomDataset.data attribute AFTER creation

    text_only = []
    for d in datasets:
        if isinstance(d.data, dict):
            d.data = [dict(zip(d.data, t)) for t in zip(*d.data.values())]
        text_only.append([item["response"] for item in d.data])

    # fuzzy deduplication with minhash
    minhash = MinHashLSHDeduplicator(tokenizer, *text_only)
    duplicate_ids = minhash.deduplicate(jaccard_sim=0.85, num_perm=128)

    print(f"{len(duplicate_ids)} duplicates detected!\nremoving them now...")

    # clean original datasets
    for e, id_list in enumerate(duplicate_ids):
        for idx in id_list[::-1]:
            _ = datasets[e].data.pop(idx)

    return datasets


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


class LazyCustomDatasetForImages(CustomDataset):
    def __init__(self, data, request_timeout: float = 3):
        CustomDataset.__init__(self, data)
        self.timeout = request_timeout

    def get_image(self, url):
        # local images (should work on gcs too) #https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse
        if urlparse(url).scheme == "":
            return Image.open(url).convert("RGB")

        # fetch from online
        try:
            return Image.open(
                io.BytesIO(requests.get(url, timeout=self.timeout).content)
            ).convert("RGB")
        except:
            raise RuntimeError(f"image: {url} could not be downloaded !")

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert "url" in sample.keys(), "sample with no `url` field detected!"

        try:
            img = self.get_image(sample["url"])
            if len(img.size) != 2 or img.size[:2] == (1, 1):
                img = None
        except:
            img = None

        return {**sample, "images": img}


class StreamingDataset(Dataset):
    """
    torch.DataLoader-like
    """

    def __init__(
        self, data, batch_size: int, n: int, skip: int = 0, collate_fn=None
    ) -> None:
        self.data_iter = data.skip(skip).take(n).iter(batch_size)
        self.index = 0
        self.n = n
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __len__(self):
        indicator = 1 if self.n % self.batch_size > 0 else 0
        return self.n // self.batch_size + indicator

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.n:
            self.index += 1
            batch = next(iter(self.data_iter))

            if self.collate_fn is not None:
                batch = self.collate_fn(batch)

            return batch
        else:
            raise StopIteration


class MultiDataLoader:
    """
    Randomly sample between several data loaders, allowing for multimodality training.
    """

    def __init__(self, *loaders, **kwargs) -> None:
        self.loaders = list(loaders)  # Current loaders in use (modifiable)
        self.loader_iters = [iter(loader) for loader in self.loaders]
        self.loader_lens = [len(loader) for loader in self.loaders]
        self.buffer = list(range(len(self.loaders)))
        self.weights = [l / sum(self.loader_lens) for l in self.loader_lens]

        # Seeding for deterministic behavior
        seed = kwargs.get("seed", None)
        if seed:
            self.sampler = random.Random(seed)
        else:
            self.sampler = None

    def __len__(self) -> int:
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
                # Remove exhausted data loader
                self.buffer.pop(idx)
                self.loader_iters.pop(idx)
                self.loader_lens.pop(idx)
                self.weights.pop(idx)

                # Recalculate buffer indices
                self.buffer = list(range(len(self.loader_iters)))

            return batch

        except StopIteration:
            # If a loader is exhausted in between, remove it and fetch from another loader
            self.buffer.remove(idx)
            return self.__next__()


def sft_collate_fn(inputs, tok):
    """
    dynamically pads input tensors and constructs attn mask
    """
    # sometimes a slice is a DL instead of LD
    if isinstance(inputs, dict):
        inputs = [dict(zip(inputs, t)) for t in zip(*inputs.values())]

    # check for bad images
    bad_ids = []
    for i, sample in enumerate(inputs):
        image = sample.get("images", True)
        if image is None or len(image.size) != 2 or image.size[:2] == (1, 1):
            bad_ids.append(i)

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
    if instruction_template is None:
        instruction_template = ("<s>user:\n{instruction}</s>assistant:\n",)
    if response_template is None:
        response_template = "{response}</s>"

    # apply chat template to samples
    def process(samples):
        prompts = [
            instruction_template.format(instruction=p) for p in samples["prompt"]
        ]
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


def grounded_qa(coords_and_labels: List[dict]):
    """
    return a grounded prompt-response pair
    """
    # read contents of bbox(es)
    # generate bbox given object(s)
    n_queries = random.choice(
        list(range(1, min(8, len(coords_and_labels))))
    )  # lower bound 1 sample

    items = random.sample(coords_and_labels, k=n_queries)

    templates = [
        (
            "provide bounding box coordinates for the following text span(s):\n{texts}",
            "{boxes}",  # maybe change to named_boxes
        ),
        ("read the text contained in the provided bounding boxes:\n{boxes}", "{texts}"),
    ]

    # some string formatting
    chunks = [i["chunk"] for i in items]
    boxes = [[round(x, 2) for c in i["coord"] for x in c] for i in items]
    boxes = [str(box) for box in boxes]
    named_boxes = [f"{c}: {b}" for c, b in zip(chunks, boxes)]

    kwargs = {
        "boxes": "\n".join(boxes),
        "named_boxes": "\n".join(named_boxes),
        "texts": "\n".join(chunks),
    }
    template = random.choices(templates, weights=[0.66, 0.33], k=1)[
        0
    ]  # more bbox than ocr

    return template[0].format(**kwargs), template[1].format(**kwargs)
