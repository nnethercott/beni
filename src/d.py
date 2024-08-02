import datasets
import torch 
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def sft_collate_fn(inputs, tok):
    # LD -> DL
    inputs = {k: [i[k] for i in inputs] for k in inputs[0].keys()}
    input_ids = inputs["input_ids"]
    prompt_len = inputs["prompt_len"]

    # needed for masking PAD loss in train
    seq_len = [len(i) for i in input_ids]
    max_len = max(seq_len)

    # pad inputs and create attention mask
    input_ids_t = [
        torch.tensor(i + [tok.pad_token_id] * (max_len - len(i))).unsqueeze(0)
        for i in input_ids
    ]
    input_ids_t = torch.cat(input_ids_t, 0)

    pos = torch.arange(max_len).unsqueeze(0).repeat((input_ids_t.shape[0], 1))
    seq_end = (
        torch.tensor([len(i) for i in input_ids]).unsqueeze(1).repeat((1, max_len))
    )
    attn_mask = (pos < seq_end).to(dtype=torch.float32)

    # TODO: add mask for prompt tokens as in pico
    labels = input_ids_t.masked_fill(attn_mask == 0, -100)

    return {
        "input_ids": input_ids_t,
        "attention_mask": attn_mask,
        "labels": labels,
    }

def tiny_shakespeare(tok, slen = 512):
    data = datasets.load_dataset(
        "karpathy/tiny_shakespeare",
        split="train",
        trust_remote_code=True,
    )
    # one big string
    text = data['text'][0]
    tokens = tok.encode(text, add_special_tokens=False)
    tokens = torch.tensor(tokens[:(slen-2)*(len(tokens)//(slen-2))]).reshape((-1, slen-2))
    bos = tok.bos_token_id*torch.ones(len(tokens)).unsqueeze(1).to(torch.long)
    eos = tok.eos_token_id*torch.ones(len(tokens)).unsqueeze(1).to(torch.long)

    #concat 
    tokens = torch.cat((bos, tokens, eos), dim=1)
    tokens = tokens.tolist()[:128]
    
    data = [{"prompt_len": 0, "input_ids": t} for t in tokens]
    ds = CustomDataset(data)

    return ds
    

def load_data(tok):
    data = datasets.load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )
    data = data.take(50000)

    def dataset_generator(dataset):
        yield from dataset

    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))
    data = data.map(
        lambda x: {
            **x,
            "input_ids": [
                tok.encode(
                    y
                )
                + [tok.eos_token_id]
                for y in x["content"]
            ],
        },
        batched=True,
    )
    data = data.filter(lambda x: [len(y) <= 512 for y in x["input_ids"]], batched=True)

    data = [{"prompt_len": 0, "input_ids": i} for i in data["input_ids"]]
    ds = CustomDataset(data)

    return ds

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    d = tiny_shakespeare(tok)
