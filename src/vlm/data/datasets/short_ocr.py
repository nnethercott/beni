import datasets
from ..prompts import OCR_PROMPT
import functools


# TODO: dynamically paste samples together into images
def load_short_ocr_sentences(
    n=100,
    split="train",
    skip=0,
):
    data = datasets.load_dataset(
        "e-val/short_ocr_sentences",
        split=split,
        streaming=True,
    )

    # load n
    data = data.skip(skip).take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def batch_fn(samples):
        samples["prompt"] = [OCR_PROMPT] * len(samples["cropped_image"])
        samples["images"] = [im.convert("RGB") for im in samples["cropped_image"]]
        return samples

    data = data.map(batch_fn, batched=True)
    data = data.rename_columns({"answer": "response"})

    # NOTE: may need this
    # data.data = [dict(zip(data.data, t)) for t in zip(*data.data.values())]
    return data
