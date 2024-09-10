from .utils import CustomDataset, LazyCustomDatasetForImages, apply_chat_template
import functools
import datasets
import re
import os
import json
import torch

NUM_IMG_TOKENS = 96
MAX_LEN = 224
TOKEN = os.getenv("HF_TOKEN")

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
"""


# a test dataset for llms to see max context length on available hardware before OOM
def tiny_shakespeare(tok, slen=512):
    data = datasets.load_dataset(
        "karpathy/tiny_shakespeare",
        split="train",
        trust_remote_code=True,
    )
    # one big string
    text = data["text"][0]
    tokens = tok.encode(text, add_special_tokens=False)
    tokens = torch.tensor(tokens[: (slen - 2) * (len(tokens) // (slen - 2))]).reshape(
        (-1, slen - 2)
    )
    bos = tok.bos_token_id * torch.ones(len(tokens)).unsqueeze(1).to(torch.long)
    eos = tok.eos_token_id * torch.ones(len(tokens)).unsqueeze(1).to(torch.long)

    # concat
    tokens = torch.cat((bos, tokens, eos), dim=1)
    tokens = tokens.tolist()[:128]

    data = [{"prompt_len": 0, "input_ids": t} for t in tokens]
    ds = CustomDataset(data)

    return ds


def load_recap(
    tok, n=100, split="train", skip=0, instruction_template=None, response_template=None
):
    data = datasets.load_dataset(
        "UCSC-VLAA/Recap-DataComp-1B",
        split=split,
        streaming=True,
        token=TOKEN,
    )

    data = data.rename_columns({"re_caption": "response"})

    # load
    data = data.skip(skip).take(n)

    def dataset_generator(dataset):
        yield from dataset

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

    data = apply_chat_template(
        data,
        tok,
        ctx_len=MAX_LEN - NUM_IMG_TOKENS,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_allava_laion(
    tok, n=100, split="instruct", instruction_template=None, response_template=None
):
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        "allava_laion",
        split=split,
        streaming=True,
        token=TOKEN,
    )

    # load n
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def batch_fn(samples):
        # build prompt and answer cols
        convo = samples["conversations"]
        human = [c[0]["value"] for c in convo]
        ai = [c[1]["value"] for c in convo]

        # remove llava image tag
        human = [re.sub("<image>", "", h).strip() for h in human]

        samples["prompt"] = human
        samples["response"] = ai
        return samples

    data = data.map(batch_fn, batched=True)

    data = apply_chat_template(
        data,
        tok,
        ctx_len=MAX_LEN - NUM_IMG_TOKENS,
        instruction_template=instruction_template,
        response_template=response_template,
    )
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_allava_text(
    tok, n=100, split="allava_text", instruction_template=None, response_template=None
):
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        split=split,
        streaming=True,
        token=TOKEN,
    )

    # load n
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        # build prompt and answer cols
        convo = samples["conversations"]
        human = [c[0]["value"] for c in convo]
        ai = [c[1]["value"] for c in convo]
        samples["prompt"] = human
        samples["response"] = ai
        return samples

    # preprocess
    data = data.map(preprocess, batched=True)

    data = apply_chat_template(
        data,
        tok,
        ctx_len=MAX_LEN,
        instruction_template=instruction_template,
        response_template=response_template,
    )
    data = data.to_list()
    return CustomDataset(data)


def load_synthdog(
    tok, n=100, split="train", instruction_template=None, response_template=None
):
    data = datasets.load_dataset(
        "naver-clova-ix/synthdog-en", split=split, streaming=True
    )
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        responses = list(
            map(
                lambda x: json.loads(x)["gt_parse"]["text_sequence"],
                samples["ground_truth"],
            )
        )
        # prompts = ["read the text in this image."] * len(responses)
        prompts = [""] * len(responses)

        samples["response"] = responses
        samples["prompt"] = prompts

        return samples

    # preprocess
    data = data.map(preprocess, batched=True)  # remove_columns=data.column_names
    data = data.rename_columns({"image": "images"})

    data = apply_chat_template(
        data,
        tok,
        truncate=True,
        ctx_len=MAX_LEN,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    # comes with images by default
    return CustomDataset(data)


def load_textocr(
    tok,
    local_path="train_data",
    n=100,
    split="train",
    instruction_template=None,
    response_template=None,
):
    data = datasets.load_dataset("jimmycarter/textocr-gpt4v", split=f"{split}[:{n}]")

    # preprocess
    def preprocess(samples):
        filenames = [f"{local_path}/{f}" for f in samples["filename"]]
        samples["prompt"] = [""] * len(samples["filename"])
        samples["filename"] = filenames
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"filename": "url", "caption_condensed": "response"})

    data = apply_chat_template(
        data,
        tok,
        truncate=True,
        ctx_len=MAX_LEN,
        instruction_template=instruction_template,
        response_template=response_template,
    )

    # comes with images by default
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_gpt4v_long(
    tok, n=100, split="train", instruction_template=None, response_template=None
):
    data = datasets.load_dataset(
        "laion/gpt4v-dataset",
        split=split,
    )

    # preprocess
    def preprocess(samples):
        samples["prompt"] = [""] * len(samples["caption"])
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"link": "url", "caption": "response"})

    data = apply_chat_template(
        data,
        tok,
        ctx_len=MAX_LEN - NUM_IMG_TOKENS,
        instruction_template=instruction_template,
        response_template=response_template,
    )
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_gpt4v_short(
    tok, n=100, split="train", instruction_template=None, response_template=None
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
        samples["prompt"] = [""] * len(samples["caption"])
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"caption": "response"})

    data = apply_chat_template(
        data,
        tok,
        truncate=True,
        ctx_len=MAX_LEN - NUM_IMG_TOKENS,
        instruction_template=instruction_template,
        response_template=response_template,
    )
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")

    # syn = load_synthdog(tok, n=100)
    ocr = load_textocr(tok, "/mnt/nate/datasets/textocr/train_images/", n=100)
