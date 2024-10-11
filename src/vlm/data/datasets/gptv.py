import datasets
import functools
from ..prompts import DETAILED_CAPTION_PROMPT, CAPTION_PROMPT


def load_gpt4v_long(
    n=100,
    split="train",
):
    data = datasets.load_dataset(
        "laion/gpt4v-dataset",
        split=f"{split}[:{n}]",
    )

    # preprocess
    def preprocess(samples):
        samples["prompt"] = [DETAILED_CAPTION_PROMPT] * len(samples["caption"])
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"link": "url", "caption": "response"})

    return data


def load_gpt4v_short(
    n=100,
    split="train",
):
    data = datasets.load_dataset(
        "laion/220k-GPT4Vision-captions-from-LIVIS",
        split=split,
        streaming=True,
    )
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    # preprocess
    def preprocess(samples):
        samples["prompt"] = [CAPTION_PROMPT] * len(samples["caption"])
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"caption": "response"})

    return data
