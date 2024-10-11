import datasets
import functools


def dataset_generator(dataset):
    yield from dataset


def load_recap(
    n=100,
    split="train",
    skip=0,
):
    data = datasets.load_dataset(
        "UCSC-VLAA/Recap-DataComp-1B",
        split=split,
        streaming=True,
    )
    data = data.rename_columns({"re_caption": "response"})

    data = data.skip(skip).take(n)
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    # no prompt
    def preprocess(samples):
        samples["prompt"] = [""] * len(samples["response"])
        return samples

    data = data.map(preprocess, batched=True)

    # filter out samples containing  "The first image" or "second image"
    data = data.filter(
        lambda x: "first image" not in x["response"].lower()
        and "second image" not in x["response"].lower()
    )

    return data


def load_lmms_lab_recap(
    n: int = 100,
    split: str = "train",
    skip: int = 0,
):
    data = datasets.load_dataset(
        "lmms-lab/LLaVA-ReCap-558K",
        split=split,
        streaming=True,
    )
    data = data.skip(skip).take(n)
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    # lazy map
    def preprocess(samples):
        samples["prompt"] = [""] * len(samples["conversations"])
        samples["response"] = [c[1]["value"] for c in samples["conversations"]]
        samples["image"] = [image.convert("RGB") for image in samples["image"]]
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"image": "images"})

    return data
