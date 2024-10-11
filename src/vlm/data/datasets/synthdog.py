import json
import datasets
import random
import functools
from ..prompts import OCR_PROMPT
from ..utils import grounded_qa


def dataset_generator(dataset):
    yield from dataset


def load_synthdog(
    n: int = 100,
    split="train",
    skip: int = 0,
    repo_name="naver-clova-ix/synthdog-en",
):
    data = datasets.load_dataset(repo_name, split=split, streaming=True)

    data = data.skip(skip).take(n)
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        images = [im.convert("RGB") for im in samples["image"]]
        responses = list(
            map(
                lambda x: json.loads(x)["gt_parse"]["text_sequence"],
                samples["ground_truth"],
            )
        )
        prompts = [OCR_PROMPT] * len(responses)
        # prompts = [""] * len(responses)

        samples["response"] = responses
        samples["prompt"] = prompts
        samples["images"] = images

        return samples

    # preprocess
    data = data.map(preprocess, batched=True)

    return data


# synthdog with bounding boxes
def load_synthdog_nate(
    n: int = 100,
    split="train",
    skip: int = 0,
):
    data = datasets.load_dataset(
        "nnethercott/synthdog-en-detection", split=split, streaming=True
    )

    data = data.skip(skip).take(n)
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        # randomly choose between pure full image ocr, bbox coord generation, and grounded ocr
        ocr = list(
            map(
                lambda x: json.loads(x)["gt_parse"]["text_sequence"],
                samples["ground_truth"],
            )
        )
        coords_and_labels = samples["2_coord_norm"]
        images = samples["image"]

        # FIXME: change this later
        prompts, responses = [], []
        for e, (o, cl) in enumerate(zip(ocr, coords_and_labels)):
            choice = random.choices([0, 1, 2], weights=[0.0, 12, 1])[0]
            # grounded qa
            if choice == 0 and len(cl) > 0:
                prompt, response = grounded_qa(cl)  # sometimes this won't work :/
            # pure ocr
            elif choice == 1 or len(cl) == 0:
                # prompt = "Read the text in this image."
                prompt = ""
                response = o
            # ocr on cropped region
            else:
                index = random.choice(
                    list(range(len(cl)))
                )  # didn't need lower bound 1 here
                region = cl[index]
                image = images[e]

                # prompt = "Read the text in this image crop."
                prompt = ""
                response = region["chunk"]
                w, h = image.size
                coord = [round(x, 2) for c in region["coord"] for x in c]
                coord = [
                    int(w * c) if e % 2 == 0 else int(h * c)
                    for e, c in enumerate(coord)
                ]
                image = image.crop(coord)
                samples["image"][e] = image

            prompts.append(prompt)
            responses.append(response)

        samples["response"] = responses
        samples["prompt"] = prompts

        return samples

    # preprocess
    data = data.map(preprocess, batched=True)  # remove_columns=data.column_names
    data = data.rename_columns({"image": "images"})

    return data
