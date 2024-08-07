from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Union
from PIL import Image
import functools
import os

import torch 
from torch import nn 
import torch.nn.functional as F
import torch.distributed as dist

from transformers import(
    CLIPImageProcessor, 
    CLIPVisionModel, 
    CLIPVisionConfig,
    PreTrainedModel, 
    AutoModelForCausalLM,
    AutoConfig,
    LlamaConfig,
    AutoTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


"""
NOTES: 
    * ideally: Beni(text_config, vision_config)
        * need some hf utils to find the right vision classes to load vision tower
"""

def rank_0_only(f):
    def wrap(*args, **kwargs):
        if not torch.distributed.is_initialized() or int(os.environ['LOCAL_RANK']) == 0:
            return f(*args, **kwargs)
    return wrap


class VisionTower(nn.Module):
    """
    todo:
        - add feature select index
        - trainable token pool ?
        - do we want the [cls] token as well?
    """
    def __init__(self, vision: PreTrainedModel, processor: Callable[[Image],torch.Tensor], r: int = 1):
        super().__init__()

        self.vision = vision 
        self.processor = processor
        self.r = r 
        self.feature_select_index = -1

        # update self.vision.config 
        setattr(self.vision.config, 'hidden_size', r*self.vision.config.hidden_size)
        
    @property
    def device(self):
        return self.vision.device

    @property
    def config(self):
        return self.vision.config


    def forward(self, x):
       x = self.processor(x, return_tensors = 'pt')['pixel_values']
       
       x = self.vision(x.to(self.device), output_hidden_states=True)
       x = x['hidden_states'][self.feature_select_index][:,1:,:]
       b,s,d = x.shape

       # concatenate adjacent tokens a la minigpt4-v2
       return x.reshape((b, s//self.r, -1))  


# revisit later
class Connector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)

        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )
    def forward(self, x):
        return self.proj(self.norm(x))


@dataclass 
class BeniConfig:
    vision_name_or_path: str = None
    llm_name_or_path: str = None
    r: int = 4
    freeze: bool = True # whether or not to freeze llm and vision
    attn_implementation: str = "sdpa" # should be able to switch this for flash attn i hope (same for clip?)


class Beni(nn.Module):
    """
    todo: 
        * allow for prompt templates
    """
    def __init__(self, config: BeniConfig):
        super().__init__()
        self.config = config

        # vision
        v = CLIPVisionModel.from_pretrained(config.vision_name_or_path)
        p = CLIPImageProcessor.from_pretrained(config.vision_name_or_path)
        self.vision = VisionTower(v, p, config.r)


        # text -- maybe turn into module like `vision`
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name_or_path, attn_implementation=config.attn_implementation,)
        self.tok = AutoTokenizer.from_pretrained(config.llm_name_or_path)
        self.tok.padding_side = 'right'
        self.tok.add_tokens(["<img>", "</img>"])
        
        self.llm.resize_token_embeddings(len(self.tok)) #https://huggingface.co/docs/transformers/en/main_classes/model

        # connector
        self.connector = Connector(self.vision_config.hidden_size, self.text_config.hidden_size)

        # freeze 
        if self.config.freeze:
            self.freeze()

    def freeze(self):
        for n,p in self.named_parameters():
            if 'connector' not in n:
                p.requires_grad = False


    @property
    def text_config(self):
        return AutoConfig.from_pretrained(self.config.llm_name_or_path)
    @property
    def vision_config(self):
        return self.vision.config
    @property
    def img_token(self):
        return self.tok.encode("<img>", add_special_tokens=False)[0]
    @property
    def end_img_token(self):
        return self.tok.encode("<\img>", add_special_tokens=False)[0]

    @property
    def device(self):
        return self.vision.device

    def pretty_print(self):
        params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() if p.requires_grad else 0 for p in self.parameters())

        if dist.is_initialized():
            params = torch.tensor(params, device="cuda") # dist group is nccl
            trainable = torch.tensor(trainable, device="cuda")

            # reduce on rank 0 and print
            dist.reduce(params, 0, op=dist.ReduceOp.SUM)
            dist.reduce(trainable, 0, op=dist.ReduceOp.SUM)

            if int(os.environ["LOCAL_RANK"]) == 0: #turn this into a `is_rank_0` function?
                print(f'VLM with: {params.item()/1e9:.1f}B params | {100*trainable.item()/params.item():.2f}% trainable\n')
                print(self)
        else:
            print(f'VLM with: {params/1e9:.1f}B params | {100*trainable/params:.2f}% trainable\nprint_trainable_parameters')
            print(self)


    def prepare_batch_if_images(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None, # in reality these are pil images
        ): 
        """
        NOTE: tokenizer right padding assumed for this to work
        * pad input_ids, attention_mask, and labels
        * need to decide on a prompt template 
            * <s><img>image_tokens_here</img>user_prompt_and_answer</s> ?
        """

        vision_embeds = self.vision(images)
        vision_embeds = self.connector(vision_embeds)
        
        if input_ids is not None:
            bsz, _ = input_ids.shape
            text_embeds = self.llm.model.embed_tokens(input_ids) 
            sos_embeds = text_embeds[:,0,:].unsqueeze(1)
            
            # embed <img> and </img>
            img_token = torch.tensor(self.img_token, device=self.device).unsqueeze(0)
            img_embeds = self.llm.model.embed_tokens(img_token).repeat((bsz,1,1))
            end_img_token = torch.tensor(self.end_img_token, device=self.device).unsqueeze(0)
            end_img_embeds = self.llm.model.embed_tokens(end_img_token).repeat((bsz,1,1))

            #<s><img>insert_image_featuers<img>prompt</s>
            inputs_embeds = torch.cat((sos_embeds, img_embeds, vision_embeds, end_img_embeds, text_embeds[:,1:,:]), dim=1)
            input_ids = None  # ensure input_ids is not passed to the model

            _, vis_len, _ = vision_embeds.shape

            # attention_mask
            additional_len = 1 + 1 #added <img> and </img>
            attention_mask = torch.cat((torch.ones((bsz, vis_len+additional_len), device=self.device), attention_mask), dim=1)

            
            # if we're computing loss
            if labels is not None:
                labels = labels[:,1:] #need to get rid of <s> since we're about to concatenate a prefix with img tokens 
                additional_len+=1 #<s> we just got rid of^
                labels_prefix = torch.tensor([-100]*(vis_len+additional_len), device = self.device)

                # IMPORTANT 
                # NOTE: this assumes all samples in batch have same number of image tokens -- in future generalize
                labels_prefix = labels_prefix.repeat((bsz, 1)) 

                labels = torch.cat((labels_prefix, labels), dim=1)

            #print(inputs_embeds.shape)
            #print(attention_mask.shape)
            #print(labels.shape)

        # TODO: consider case when input_embeds passed

        return input_ids, attention_mask, inputs_embeds, labels

        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None, # in reality these are pil images
        return_dict: Optional[bool] = None,
        **kwargs, #debug
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        """
        passes input embeddings directly to llm if images present. otherwise use input_ids
        """
        assert input_ids is not None or images is not None, "You can't forward without text and/or images!"

        # only if images do we modify & pad
        if images is not None:
                input_ids, attention_mask, inputs_embeds, labels = self.prepare_batch_if_images(input_ids, attention_mask, inputs_embeds, labels, images)
            
        #print(kwargs)
        #print(attention_mask)
        #print(self.tok.batch_decode(labels.masked_fill(labels==-100, self.tok.encode('X', add_special_tokens=False)[0])))

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                _,
                attention_mask,
                inputs_embeds,
                _
            ) = self.prepare_inputs(
                input_ids,
                attention_mask,
                None,
                None,
                images,
            )
            input_ids = None

        else:
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        return self.llm.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


if __name__ == "__main__":
    from datasets import load_dataset
    import sys
    sys.path.insert(1, "../")
    from d import *

    VISION_MODEL_ID = "openai/clip-vit-large-patch14"
    TEXT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    cfg = BeniConfig(
        vision_name_or_path = VISION_MODEL_ID,
        llm_name_or_path = TEXT_MODEL_ID,
        r = 1,
        freeze = True,
    )
        
    beni = Beni(cfg)
    beni.to("cuda")
    beni.pretty_print()

    optimizer = torch.optim.AdamW(
        beni.parameters(),
        lr = 1e-03,
    )

    ds = load_recap(beni.tok, 100)
    dl = torch.utils.data.DataLoader(ds, batch_size = 2, collate_fn = functools.partial(sft_collate_fn, tok=beni.tok))
    inputs = next(iter(dl))

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(beni.device)

    out = beni(**inputs)

    loss = out['loss']
    #loss.backward()

    #for p in beni.connector.parameters():
    #    print(p.grad)
    #
    #optimizer.step()
    #optimizer.zero_grad()


