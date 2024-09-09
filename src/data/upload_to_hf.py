import datasets
import functools
import os

data = datasets.load_dataset(
    "UCSC-VLAA/Recap-DataComp-1B",
    split="train",
    streaming=True,
    token=os.environ["HF_ACCESS_TOKEN"],
)


def dataset_generator(dataset):
    yield from dataset


data = data.take(100000)
data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

# data = CustomDatasetForImages(data.to_list(), cache=True)
# data.build()

# resolve downloaded images
# import json
#
# with open("./data/image_cache.json") as f:
#    mapping = json.loads(f.read())
#
## i'm dumb
# mapping = {v: k for k, v in mapping.items()}
# feature = datasets.Image()
#
# urls = list(mapping.keys())
# data = data.filter(lambda d: d["url"] in urls)
#
#
# def image_to_bytes(image_path):
#    with open(image_path, "rb") as f:
#        return f.read()
#
#
# def add_images(samples):
#    urls = samples["url"]
#    images = [image_to_bytes(f"./data/data/{mapping[u]}.jpg") for u in urls]
#    samples["bytes"] = images
#    return samples


# data = data.map(add_images, batched=True)

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

df = pd.DataFrame(data.to_list())
table = pa.Table.from_pandas(df)
pq.write_table(table, "./data/data/train.parquet")

# upload
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_ACCESS_TOKEN"])

api.upload_folder(
    folder_path="./data/data",
    repo_id="nnethercott/Recap-DataComp-100K",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
)
