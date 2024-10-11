from .utils import (
    LazyCustomDatasetForImages,
    CustomDataset,
    apply_chat_template,
)
import functools
import datasets
import os
from transformers import AutoTokenizer

"""
Available dataset loaders:

* tiny_shakespeare [https://huggingface.co/datasets/karpathy/tiny_shakespeare]
    * use for testing max context length on hardware before cuda OOM
* load_allava_text [https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V]
    * rich instruction tuning for stage 2 (text only)
* load_allava_laion [https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V]
    * insanely detailed captions done with gpt4v
    * rich instruction tuning for stage 2
* load_recap [https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B]
    * basic image captioning done by llava for stage 1
* load_synthdog [https://huggingface.co/datasets/naver-clova-ix/synthdog-en]
    * stage 1 ocr possibly
* load_textocr [https://huggingface.co/datasets/jimmycarter/textocr-gpt4v]
    * stage 1/2 captioning with ocr rich captions
* load_gpt4v_long [https://huggingface.co/datasets/laion/gpt4v-dataset]
    * stage 2 captioning, 25k higher quality
* load_gpt4v_short [https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS]
    * stage 1 captioning, 200k lower quality
* load_blip3_ocr
* load_blip3_grounded
* load_synthdog_nate
* load_recap
* load_lmms_lab_recap
"""

MAX_LEN = 384
TRUNCATE = True
TOKEN = os.getenv("HF_TOKEN")


def prepare(
    data: datasets.Dataset,
    tokenizer: AutoTokenizer,
    instruction_template="{instruction}\n",
    response_template="{response}",
):
    data = apply_chat_template(
        data,
        tokenizer,
        instruction_template=instruction_template,
        response_template=response_template,
        truncate=TRUNCATE,
        ctx_len=MAX_LEN,
    )

    if "images" in data.column_names:
        data = CustomDataset(data)
        # data.data = [dict(zip(data.data, t)) for t in zip(*data.data.values())]
    elif "url" in data.column_names:
        data = data.to_list()
        data = LazyCustomDatasetForImages(data)

    return data


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from .utils import sft_collate_fn
    from torch.utils.data import DataLoader
    from .datasets import load_blip3_grounded

    special_tokens = [f"0.{i}" for i in range(10)]
    special_tokens += [f"0.{i}{j}" for i in range(10) for j in range(1, 10)]
    special_tokens += ["0.0", "1.0"]
    special_tokens += ["<img>", "</img>"]
    special_tokens += ["<bbox>", "</bbox>"]

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tok.add_special_tokens({"additional_special_tokens": special_tokens})

    data = load_blip3_grounded(400)
    data = prepare(data, tok)

    dl = DataLoader(
        data, batch_size=1, collate_fn=functools.partial(sft_collate_fn, tok=tok)
    )

    dlit = iter(dl)
    batch = next(dlit)

    def foo():
        sample = next(dlit)
        print(sample["images"])
        if "url" in sample.keys():
            print(sample["url"])
        print(tok.batch_decode(sample["input_ids"])[0])
