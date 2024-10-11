import datasets
import functools
import ast
import re
import json
from ..prompts import CAPTION_PROMPT, GROUNDED_PROMPTS, OCR_PROMPT


def relative_area(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    return min(box1_area, box2_area) / (max(box1_area, box2_area) + 1e-06)


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xx1, yy1, xx2, yy2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
    w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
    inter_area = w * h

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def load_blip3_grounded(
    n: int = 100,
    split="train",
):
    # "Salesforce/blip3-grounding-50m"
    data = datasets.load_dataset(
        "Salesforce/blip3-grounding-50m", split=split, streaming=True
    )
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    # super messy
    def preprocess(samples):
        def replace_spans_with_bbox(text):
            # Regex pattern to capture the whole span (starts at [x1, y1] and extends upto [x2, y2])
            pattern = r"\(starts at \[([\d\.\-]+),\s*([\d\.\-]+)\] and extends upto \[([\d\.\-]+),\s*([\d\.\-]+)\]\)\w*"

            # ad-hoc rule to remove multiple bounding boxes
            boxes = []

            # Function to process each match
            def replace_match(match):
                x1, y1 = float(match.group(1)), float(match.group(2))
                x2, y2 = float(match.group(3)), float(match.group(4))

                # some checks
                if x2 < x1 or y2 < y1:
                    return ""

                if (x2 - x1) * (y2 - y1) > 0.90:
                    return ""

                # low effort nms
                box = [x1, y1, x2, y2]
                if any((iou(box, b) > 0.2 for b in boxes)):
                    return ""

                boxes.append(box)

                # Otherwise, format it in <bbox> tag
                eps = 1e-04
                return f"<bbox>[{round(x1+eps,2)},{round(y1+eps,2)},{round(x2+eps,2)},{round(y2+eps,2)}]</bbox>"

            # Replace all occurrences of the pattern in the text
            new_text = re.sub(pattern, replace_match, text)
            # new_text = re.sub(r"-([.][0-9]+)", r"\1", new_text)
            new_text = re.sub(" +", " ", new_text)

            # super ugly
            indices = []
            for seq in [
                "The image contains",
                "Although",
                "Despite",
                "This sentence",
                "The detailed",
            ]:
                if seq in new_text:
                    indices.append(new_text.find(seq))
                else:
                    indices.append(len(new_text))

            new_text = new_text[: min(indices)].strip()

            return new_text

        def prepare_response(x):
            try:
                parsed = ast.literal_eval(x)[1][0]
                parsed = replace_spans_with_bbox(parsed)
                return parsed

            except:
                return None

        responses = [prepare_response(c) for c in samples["captions"]]
        # prompts = [random.choice(GROUNDED_PROMPTS) for _ in range(len(responses))]
        prompts = [CAPTION_PROMPT] * len(responses)

        samples["prompt"] = prompts
        samples["response"] = responses
        return samples

    # preprocess
    data = data.map(preprocess, batched=True)  # remove_columns=data.column_names

    def filter_fn(x):
        if x["response"] is None:
            return False

        pattern = "\[\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*\]"
        matches = re.findall(pattern, x["response"])
        if len(set(matches)) < 3 or len(matches) > 8:
            return False
        return True

    data = data.filter(filter_fn)

    # data = apply_chat_template(
    #     data,
    #     tok,
    #     truncate=True,
    #     ctx_len=MAX_LEN,
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    # )

    # data = data.to_list()
    # return LazyCustomDatasetForImages(data)

    return data


# lazy
def load_blip3_ocr(
    n: int = 100,
    split="train",
):
    data = datasets.load_dataset(
        "Salesforce/blip3-ocr-200m", split=split, streaming=True
    )
    data = data.take(n)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        def replace_spans_with_bbox(text):
            # Regex pattern to capture the whole span (starts at [x1, y1] and extends upto [x2, y2])
            pattern = r"starts at \[([\d\.\-]+),\s*([\d\.\-]+)\] and extends upto \[([\d\.\-]+),\s*([\d\.\-]+)\]"

            # ad-hoc rule to remove multiple bounding boxes
            boxes = []

            # Function to process each match
            def replace_match(match):
                x1, y1 = float(match.group(1)), float(match.group(2))
                x2, y2 = float(match.group(3)), float(match.group(4))

                # some checks
                if x2 < x1 or y2 < y1:
                    return ""

                if (x2 - x1) * (y2 - y1) > 0.90:
                    return ""

                # low effort nms
                box = [x1, y1, x2, y2]
                if any((iou(box, b) > 0.1 for b in boxes)):
                    return ""

                boxes.append(box)

                # Otherwise, format it in <bbox> tag
                eps = 1e-04
                return f"<bbox>[{round(x1+eps,2)},{round(y1+eps,2)},{round(x2+eps,2)},{round(y2+eps,2)}]</bbox>"

            # Replace all occurrences of the pattern in the text
            new_text = re.sub(pattern, replace_match, text)
            # new_text = re.sub(r"-([.][0-9]+)", r"\1", new_text)
            new_text = re.sub(" +", " ", new_text)

            return new_text

        def prepare_response(x):
            try:
                parsed = json.loads(x)[-2]["text"]
                parsed = replace_spans_with_bbox(parsed)
                return parsed

            except:
                return None

        responses = [prepare_response(c) for c in samples["captions"]]
        prompts = [OCR_PROMPT] * len(responses)

        samples["prompt"] = prompts
        samples["response"] = responses
        return samples

    # preprocess
    data = data.map(preprocess, batched=True)  # remove_columns=data.column_names

    def filter_fn(x):
        if x["response"] is None:
            return False

        pattern = "\[\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*,\s*([\d\.\-]+)\s*\]"
        matches = re.findall(pattern, x["response"])
        if len(set(matches)) < 3 or len(matches) > 8:
            return False
        return True

    data = data.filter(filter_fn)

    return data
