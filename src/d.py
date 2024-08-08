import datasets
import torch 
from torch.utils.data import Dataset, DataLoader
import os
from typing import Callable
import functools
import random 
from tqdm import tqdm 

from concurrent.futures import ThreadPoolExecutor 
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


# TODO: use this to download images instead of dynamic get  
class CustomDatasetForImages(CustomDataset):
    def __init__(self, data, cache=True, request_timeout: float = 3.):
            super().__init__(data)
            self.offset = 0
            self.cache = cache
            self.timeout = request_timeout

            self.build() 
            self.trim()

    def get_image(self, url):
        try:
            return Image.open(io.BytesIO(requests.get(url, timeout=self.timeout).content)).convert('RGB')
        except:
            raise RuntimeError(f"image: {url} could not be downloaded !")

    #@override
    def __getitem__(self, i):
        if self.cache:
            return self.data[i] 
        
        img = self.get_image(i)
        return {**self.data[i], 'images': img}
        

    def build(self):
        urls_enumed = [(e,d['url']) for e,d in enumerate(self.data)]

        def download(idx, url):
            try:
                img = self.get_image(url)
                if not len(img.size)==2 or img.size[:2] == (1,1):  # edge cases
                    return (idx, None)

                if self.cache:
                    self.data[idx] = {**self.data[idx], 'images': img}
                return (idx, img)
            except:
                return (idx, None)

        with ThreadPoolExecutor(max_workers = int(os.environ.get("OMP_NUM_THREADS", 4))) as executor:
            res = list(tqdm(executor.map(lambda args: download(*args), urls_enumed), total=len(urls_enumed)))

        # remove bad ids from data 
        # nice leet code stuff 
        bad_ids = [p[0] for p in res if p[1] is None]        
        bad_ids = sorted(bad_ids)[::-1]
        for idx in bad_ids:
            self.data.pop(idx)

    def trim(self):
        if torch.distributed.is_initialized():
            # broadcast self.data lengths and choose min 
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
        torch.tensor(i + [tok.eos_token_id] * (max_len - len(i))).unsqueeze(0)
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
                             preprocess: Callable[[datasets.Dataset,],datasets.Dataset]=None,
                             template = "{prompt}\n{response}</s>"):
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
            prompt_len = [len(tok.encode("{p}\n".format(p=p))) for p in prompts] #TODO: make this better
        else:
            prompts = ['']*len(samples['response'])
            prompt_len = [0 for _ in prompts] # 1 for the <

        resp = samples['response']
        inputs = [template.format(prompt=p, response=r).strip() for p,r in zip(prompts,resp)]
        input_ids = [tok.encode(i) for i in inputs]

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


def load_recap(tok, n=100):
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

    data = load_data_from_generator(data, tok, n=n, preprocess=None)
    data = data.to_list()
    return CustomDatasetForImages(data)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    ds = load_recap(tok, 100)
    #dl = DataLoader(ds, batch_size = 4, collate_fn = functools.partial(sft_collate_fn, tok=tok))
    #batch = next(iter(dl))
    #print(batch)
    
