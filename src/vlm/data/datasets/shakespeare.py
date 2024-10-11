import datasets
import torch
from ..utils import CustomDataset


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
