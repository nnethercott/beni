from .utils import *
import datasets
import re


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

    data = load_data_from_generator(data, tok, n=n, skip=100000, ctx_len = (196-91-2), preprocess=None)
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_allava_laion(tok, n=100):
    """
    provides text-only and image-text datasets
    """
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        "allava_laion",
        split="instruct",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    def preprocess(data):
        def batch_fn(samples):
            # build prompt and answer cols 
            convo = samples['conversations']
            human = [c[0]['value'] for c in convo]
            ai = [c[1]['value'] for c in convo]

            # remove llava image tag
            human = [re.sub('<image>', '', h).strip() for h in human]

            samples['prompt'] = human
            samples['response'] = ai
            return samples
        return data.map(batch_fn, batched=True)


    data = load_data_from_generator(data, tok, n=n, ctx_len = (256-93-2), preprocess=preprocess)
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_allava_text(tok, n=100):
    """
    provides text-only and image-text datasets
    """
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        "allava_text",
        split="train",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    def preprocess(data):
        def batch_fn(samples):
            # build prompt and answer cols 
            convo = samples['conversations']
            human = [c[0]['value'] for c in convo]
            ai = [c[1]['value'] for c in convo]
            samples['prompt'] = human
            samples['response'] = ai
            return samples
        return data.map(batch_fn, batched=True)


    data = load_data_from_generator(data, tok, n=n, ctx_len = 256, truncate=False, preprocess=preprocess)
    data = data.to_list()
    return CustomDataset(data)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    dst = load_allava_text(tok, n=100)
    dsv = load_allava_laion(tok, n=100)
    dlt = DataLoader(dst, batch_size = 2, collate_fn = functools.partial(sft_collate_fn, tok=tok))
    dlv = DataLoader(dsv, batch_size = 4, collate_fn = functools.partial(sft_collate_fn, tok=tok))
    
    ml = MultiDataLoader(dlv, dlt)
    batch = next(iter(ml))
    #print(batch)
    
