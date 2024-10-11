import datasets
from ..prompts import OCR_PROMPT


def load_textocr(
    local_path="train_data",
    n=100,
    split="train",
):
    """requires images to be downloaded locally"""
    data = datasets.load_dataset("jimmycarter/textocr-gpt4v", split=f"{split}[:{n}]")

    # preprocess
    def preprocess(samples):
        filenames = [f"{local_path}/{f}" for f in samples["filename"]]
        samples["prompt"] = [OCR_PROMPT] * len(samples["filename"])
        samples["filename"] = filenames
        return samples

    data = data.map(preprocess, batched=True)
    data = data.rename_columns({"filename": "url", "caption_condensed": "response"})

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
