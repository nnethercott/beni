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
    

def load_recap(tok, n=100, skip=0, template = None):
    """
    our pretraining/alignment data for making llm understand text
    """
    data = datasets.load_dataset(
        "UCSC-VLAA/Recap-DataComp-1B",
        split="train",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    data = data.rename_columns({'re_caption': 'response'})
    
    # load 
    data = data.skip(skip).take(n)
    def dataset_generator(dataset):
        yield from dataset
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    # no prompt
    def preprocess(samples):
        samples['prompt'] = ['']*len(samples['response'])
        return samples
    data = data.map(preprocess, batched=True)

    data = apply_chat_template(data, tok, ctx_len = 320, template = template)
    data = data.to_list()
    return LazyCustomDatasetForImages(data)
    #return CustomDataset(data)


def load_allava_laion(tok, n=100, template = None):
    """
    image-text SFT dataset
    """
    data = datasets.load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        "allava_laion",
        split="instruct",
        streaming=True,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    # load n
    data = data.take(n)
    def dataset_generator(dataset):
       yield from dataset
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

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

    data = data.map(batch_fn, batched=True)

    data = apply_chat_template(data, tok, ctx_len = (320-93-2), template=template)
    data = data.to_list()
    return LazyCustomDatasetForImages(data)


def load_allava_text(tok, n=100, template = None):
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

    # load n
    data = data.take(n)
    def dataset_generator(dataset):
       yield from dataset
    data = datasets.Dataset.from_generator(functools.partial(dataset_generator, data))

    def preprocess(samples):
        # build prompt and answer cols 
        convo = samples['conversations']
        human = [c[0]['value'] for c in convo]
        ai = [c[1]['value'] for c in convo]
        samples['prompt'] = human
        samples['response'] = ai
        return samples


    # preprocess
    data = data.map(preprocess, batched=True)

    data = apply_chat_template(data, tok, ctx_len = 320, template=template)
    data = data.to_list()
    return CustomDataset(data)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")

    template = tok.bos_token + "user\n{prompt}" + tok.eos_token + "assistant\n"
    d = load_allava_text(tok, n=100, template=template)
    
    ml = MultiDataLoader(d)
    batch = next(iter(ml))
    #print(batch)
    
