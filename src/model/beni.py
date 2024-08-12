from dataclasses import dataclass, asdict
from typing import Callable, Optional, List, Tuple, Union
from PIL import Image
import functools
import os
import importlib 

import torch 
from torch import nn 
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import(
    AutoModel,
    AutoProcessor,
    PreTrainedModel, 
    AutoModelForCausalLM,
    AutoConfig,
    LlamaConfig,
    AutoTokenizer,
)
from peft import PeftModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .vision import VisionTower


def rank_0_only(f):
    def wrap(*args, **kwargs):
        if not torch.distributed.is_initialized() or int(os.environ['LOCAL_RANK']) == 0:
            return f(*args, **kwargs)
    return wrap


# transformer-like
# make sure to wrap in fsdp
#class Connector(nn.Module):
#    def __init__(self, d_in, d_out):
#        super().__init__()
#
#        self.norm_1 = nn.LayerNorm(d_in)
#        self.proj_1 = nn.Sequential(
#            nn.Linear(d_in, d_in),
#            nn.GELU(),
#        )
#        self.norm_2 = nn.LayerNorm(d_in)
#        self.proj_2 = nn.Linear(d_in, d_out)
#
#    def forward(self, x):
#        x = x + self.proj_1(self.norm_1(x))
#        return self.proj_2(self.norm_2(x))

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
    text_name_or_path: str = None
    vision_cls: str = 'AutoModel'
    vision_processor_cls: str = 'AutoProcessor'
    text_cls: str = 'AutoModelForCausalLM'
    r: int = 4
    freeze: bool = True # whether or not to freeze llm and vision
    attn_implementation: str = "sdpa" # should be able to switch this for flash attn i hope (same for clip?)
    img_size: dict = None


class Beni(nn.Module):
    """
    todo: 
        * allow for prompt templates
    """
    def __init__(self, config: BeniConfig):
        super().__init__()
        self.config = config
        self.build_vision_and_text(config)

        # connector
        self.connector = Connector(self.vision_config.hidden_size, self.text_config.hidden_size)

        # freeze 
        if self.config.freeze:
            self.freeze()

    def build_vision_and_text(self, config):
        vision_cls = getattr(transformers, config.vision_cls, 'AutoModel')
        vision_processor_cls = getattr(transformers, config.vision_processor_cls, 'AutoProcessor')
        text_cls = getattr(transformers, config.text_cls, 'AutoModelForCausalLM')

        v = vision_cls.from_pretrained(config.vision_name_or_path)
        p = vision_processor_cls.from_pretrained(config.vision_name_or_path)
        self.vision = VisionTower(v, p, **asdict(config))


        # text -- maybe turn into module like `vision`
        self.llm = text_cls.from_pretrained(config.text_name_or_path, attn_implementation=config.attn_implementation,)
        self.tok = AutoTokenizer.from_pretrained(config.text_name_or_path)
        self.tok.padding_side = 'right'
        self.tok.add_tokens(["<img>", "</img>"])
        #self.tok.add_tokens(["<s>", "</s>"])
        
        self.llm.resize_token_embeddings(len(self.tok)) #https://huggingface.co/docs/transformers/en/main_classes/model


    def freeze(self):
        for n,p in self.named_parameters():
            if 'connector' not in n:
                p.requires_grad = False


    @property
    def text_config(self):
        return AutoConfig.from_pretrained(self.config.text_name_or_path)
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

    @property
    def embed_tokens(self):
        if isinstance(self.llm, PeftModel):
            return self.llm.model.model.embed_tokens
        return self.llm.model.embed_tokens


    def prepare_inputs(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None, # in reality these are pil images
        ): 
        """
        TODO:
            * handle case where input_embeds passed 

        NOTE: tokenizer right padding assumed for this to work
        * pad input_ids, attention_mask, and labels
        * need to decide on a prompt template 
            * <s><img>image_tokens_here</img>user_prompt_and_answer</s> ?
        """

        if images is None:
            #print("text")
            return input_ids, attention_mask, inputs_embeds, labels

        #print("images")
        bsz = len(images)

        vision_embeds = self.vision(images)
        vision_embeds = self.connector(vision_embeds)

        # bos
        bos_token = torch.tensor(self.tok.bos_token_id, device=self.device).unsqueeze(0)
        bos_embeds = self.embed_tokens(bos_token).repeat((bsz,1,1))
        
        # embed <img> and </img>
        img_token = torch.tensor(self.img_token, device=self.device).unsqueeze(0)
        img_embeds = self.embed_tokens(img_token).repeat((bsz,1,1))
        end_img_token = torch.tensor(self.end_img_token, device=self.device).unsqueeze(0)
        end_img_embeds = self.embed_tokens(end_img_token).repeat((bsz,1,1))

        if input_ids is not None:
            text_embeds = self.embed_tokens(input_ids) 

            #<s><img>insert_image_featuers<img>prompt</s>
            inputs_embeds = torch.cat((bos_embeds, img_embeds, vision_embeds, end_img_embeds, text_embeds[:,1:,:]), dim=1)
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


        else:
            inputs_embeds = torch.cat((bos_embeds, img_embeds, vision_embeds, end_img_embeds), dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:-1], device=self.device)

        return None, attention_mask, inputs_embeds, labels

        
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

        #print(self.tok.batch_decode(input_ids))
        input_ids, attention_mask, inputs_embeds, labels = self.prepare_inputs(input_ids, attention_mask, inputs_embeds, labels, images)
            
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
            # NOTE: we'll never generate with a peft model -- always merge first so this should hold
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        #print(inputs_embeds.shape)
        #print(attention_mask.shape)

        return self.llm.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


if __name__ == "__main__":
    from datasets import load_dataset
    import sys
    sys.path.insert(1, "../")
    from data import *
    import io
    import requests
    from peft import PeftModel, PeftConfig

    cfg = BeniConfig(
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        r = 11,
        freeze = True,
        attn_implementation="sdpa",
        img_size = {'height': 448, 'width': 448},
    )
    beni = Beni(cfg)
    beni.connector.load_state_dict(torch.load("./model_checkpoints/finetune/tinyllama1b-siglip400m-ft-connector-step12000.pt")) 
    beni.llm = PeftModel.from_pretrained(beni.llm, "./model_checkpoints/finetune/tinyllama1b-siglip400m-ft-step12000")
    beni.llm = beni.llm.merge_and_unload()
    #beni.to("cuda")
    #beni.pretty_print()
    print(beni)
    

    optimizer = torch.optim.AdamW(
        beni.parameters(),
        lr = 1e-03,
    )

    #ds = load_recap(beni.tok, 10)
    #dl = torch.utils.data.DataLoader(ds, batch_size = 1, collate_fn = functools.partial(sft_collate_fn, tok=beni.tok))
    #inputs = next(iter(dl))

    #for k, v in inputs.items():
    #    if isinstance(v, torch.Tensor):
    #        inputs[k] = v.to(beni.device)

    #inputs.pop('input_ids')
    #inputs.pop('url')
    #inputs.pop('images')
    #print(inputs)

    inputs = {}
    img = Image.open(io.BytesIO(requests.get("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7PD6N_veaEeELjZn_7J4MNDBGEY1GC83VVQ&s").content)).convert("RGB")
    #sentence = "tell me a funny story about a russian."
    sentence = "describe this image please."
    template = "{prompt}</s>\n"
    inputs = beni.tok(template.format(prompt=sentence), return_tensors='pt')
    inputs['images'] = [img,]

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(beni.device)

    out = beni.generate(**inputs, max_new_tokens = 128, do_sample=False, num_beams=3, num_return_sequences=1)
    print(beni.tok.batch_decode(out))

    #out = beni(**inputs)
    #loss = out['loss']
    #loss.backward()

    #for p in beni.connector.parameters():
    #    print(p.grad)
    #
    #optimizer.step()
    #optimizer.zero_grad()


