import datasets
import re
import functools


def load_allava_laion(
    n=100,
    split="instruct",
):
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        "allava_laion",
        split=split,
        streaming=True,
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

    # filter out samples with high ppl
    data = data.filter(lambda x: x["llava-1.5-7b-PPL"] <= 14.0)

    return data


def load_allava_text(
    n=100,
    split="allava_text",
):
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        split=split,
        streaming=True,
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

    # data = apply_chat_template(
    #     data,
    #     tok,
    #     truncate=True,
    #     ctx_len=MAX_LEN,
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    # )
    # data = data.to_list()
    # return CustomDataset(data)
    return data
