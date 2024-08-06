import datasets
import torch 
from torch.utils.data import Dataset, DataLoader
import os
from typing import Callable
import functools

# from concurrent.futures import ThreadPoolExecutor dataset.map uses concurrency already
import requests
from PIL import Image
import io

"""
NOTES:
    * should sort samples by length and guesstimate boundaries based on # tokens 
        * e.g. batches of varying sizes depending on number of total tokens 

IDEA:
    * stateful DataLoader subclass which modifies batch size based on last seen sequence length of an ordered dataset
        * can adjust batch size dynamically for us
"""

# torch.utils.data.Dataset subclass
class CustomDataset(Dataset):
    def __init__(self, data):
        #TODO: sorted(data, key=lambda x: len(x['input_ids'])) 
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


class CustomDatasetForImages(CustomDataset):
    def __init__(self, data):
            super().__init__(data)
            self.offset = 0
            #self.purge() # lets see what breaks first 

    #@override
    def __getitem__(self, i):
        """
        this is where the magic happens. if we can't load an image from its url
        we increment offset and pop the bad sample

        since we sort the dataset by seq length before train we'll load & clean images
        from self.data
        """
        img = None
        while img is None:
            idx = min(i + self.offset, self.__len__()-1) # prevent indexing errors
            url = self.data[idx]['url']
            try:
                img = Image.open(io.BytesIO(requests.get(url).content)).convert('RGB')
                break
            except: 
                self.offset+=1
        return {**self.data[idx], 'images':img}


    def purge(self):
        """
        method to remove bad images from self by iterating __getitem__
        """
        pass


def sft_collate_fn(inputs, tok):
    """
    dynamically pads input tensors and constructs attn mask
    """
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
    prompt_end = (
        torch.tensor(prompt_len).unsqueeze(1).repeat((1, max_len))
    )
    attn_mask = (pos < seq_end)


    # TODO: zero out loss on prompt tokens for labels
    labels_t = input_ids_t.clone()
    labels_t.masked_fill_(attn_mask == 0, -100)

    if prompt_len != [0]*len(prompt_len):
        labels_t.masked_fill_(pos<=prompt_end, -100) #-1 comes from \n between prompt and answer 

    return {
        "input_ids": input_ids_t,
        "attention_mask": attn_mask.to(torch.long),
        "labels": labels_t,
        "images": inputs.get("images", None),
        "url": inputs.get("url"), #debug
    }


# a test dataset for llms
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
    

def load_data_from_generator(data: datasets.Dataset, 
                             tok, 
                             n: int = 100, 
                             preprocess: Callable[[datasets.Dataset,],datasets.Dataset]=None):
    """
    Arguments:
        - data: lazy dataset loaded with `streaming=True`. dataset should have column 'response'. if no 'prompt' present we use ''
        - tok: tokenizer
        - template: optional syntaxing 
        - n: how many samples to load
        - preprocess: optional preprocessing mapped before the boilerplate
    
    features:
        - adds EOS token to the end of each text entry -> formats like SOS+encoded+EOS
            - ensures free lunch with pure-text datasets, image ones handled by vlm 
    """

    template = "{prompt}\n{response}"

    # load
    data = data.take(n)
    def dataset_generator(dataset):
        yield from dataset
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))
    #return data

    # pre-boilerplate processing
    if preprocess is not None:
        data = preprocess(data)

    def process(samples):
        if 'prompt' in samples.keys():
            prompts = samples['prompt']
            prompt_len = [len(tok.encode(p)) for p in prompts]
        else:
            prompts = ['']*len(samples['response'])
            prompt_len = [0 for _ in prompts]

        resp = samples['response']
        inputs = [template.format(prompt=p, response=r).strip() for p,r in zip(prompts,resp)]
        input_ids = [tok.encode(i) + [tok.eos_token_id] for i in inputs]

        # NOTE: this doesn't consider the additional tokens coming from the template - e.g. "USER: {prompt}\nASSISTANT: {response}"
        
        samples['input_ids'] = input_ids
        samples['prompt_len'] = prompt_len

        return samples
        
    # map 
    data = data.map(process, batched=True,)

    # filter
    #data = data.filter(lambda x: [len(y) <= 512 for y in x["input_ids"]], batched=True)

    # loadable in torch dataloader
    return data


def load_recap(tok, n):
    """
    some samples won't download so we'll have to delete those
    """
    data = datasets.load_dataset(
        "UCSC-VLAA/Recap-DataComp-1B",
        split="train",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    data = data.rename_columns({'re_caption': 'response'})
    
    # some pretty trash images - but their values are broken
    def preprocess(samples):
        samples = samples.filter(lambda x: x['re_clip_score']>0.85)
        return samples

    data = load_data_from_generator(data, tok, n=100, preprocess=None)

    #return CustomDatasetForImages(data)
    return data

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    #d = tiny_shakespeare(tok)

    ds = load_recap(tok, 100)
    dl = DataLoader(ds, batch_size = 4, collate_fn = functools.partial(sft_collate_fn, tok=tok))
    batch = next(iter(dl))
    #print(batch)
    
