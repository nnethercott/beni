from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Union
from PIL import Image

import torch 
from torch import nn 
import torch.nn.functional as F

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

from datasets import load_dataset

"""
NOTES: 
    * ideally: Beni(text_config, vision_config)
        * need some hf utils to find the right vision classes to load vision tower
"""

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
       x = x['hidden_states'][-1][:,1:,:]
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


# todo: make a real beniconfig
@dataclass 
class BeniConfig:
    vision_name_or_path: str = None
    llm_name_or_path: str = None
    r: int = 4

# language now?
class Beni(nn.Module):
    """
    todo: 
        - projector layer missing
    """
    def __init__(self, config: BeniConfig):
        super().__init__()
        self.config = config

        # vision
        v = CLIPVisionModel.from_pretrained(config.vision_name_or_path)
        p = CLIPImageProcessor.from_pretrained(config.vision_name_or_path)
        self.vision = VisionTower(v, p, config.r)

        # connector
        self.connector = Connector(self.vision_config.hidden_size, self.text_config.hidden_size)

        # text -- maybe turn into module like `vision`
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name_or_path)
        self.tok = AutoTokenizer.from_pretrained(config.llm_name_or_path)
        self.tok.padding_side = 'right'

    @property
    def text_config(self):
        return AutoConfig.from_pretrained(self.config.llm_name_or_path)
    @property
    def vision_config(self):
        return self.vision.config

    @property
    def device(self):
        return self.vision.device

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

        vision_embeddings = self.vision(images)
        vision_embeddings = self.connector(vision_embeddings)
        
        if input_ids is not None:
            bsz, _ = input_ids.shape
            text_embeddings = self.llm.model.embed_tokens(input_ids) 
            sos_embeddings = text_embeddings[:,1,:].unsqueeze(1)
            
            # embed <img> and </img>
            img_token = torch.tensor(self.tok.encode("<img>", add_special_tokens=False), device=self.device).unsqueeze(0)
            img_embeds = self.llm.model.embed_tokens(img_token).repeat((bsz,1,1))

            end_img_token = torch.tensor(self.tok.encode("</img>", add_special_tokens=False), device=self.device).unsqueeze(0)
            end_img_embeds = self.llm.model.embed_tokens(end_img_token).repeat((bsz,1,1))

            #<s><img>insert_image_featuers<img>prompt</s>
            inputs_embeds = torch.cat((sos_embeddings, img_embeds, vision_embeddings, end_img_embeds, text_embeddings[:,1:,:]), dim=1)
            input_ids = None  # ensure input_ids is not passed to the model


            _, seq_len, _ = vision_embeddings.shape

            # attention_mask
            additional_len = len(self.tok.encode("<img>", add_special_tokens=False))
            additional_len += len(self.tok.encode("</img>", add_special_tokens=False)) #TODO define these as property
            attention_mask = torch.cat((torch.ones((bsz, seq_len+additional_len), device=self.device), attention_mask), dim=1)

            
            # if we're computing loss
            if labels is not None:
                labels = labels[:,1:] #need to get rid of <s> since we don't wanna predict <s> after </img> 
                additional_len+=1 #<s> we just got rid of^
                labels_prefix = torch.tensor([-100]*(seq_len+additional_len), device = self.device)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        """
        Notes:
            * modify attention mask to be bidirectional over image tokens
            * add -100s for image tokens
        """

        assert input_ids is not None or images is not None, "You can't forward without text and/or images!"

        # only if images do we modify & pad
        if images is not None:
                input_ids, attention_mask, inputs_embeds, labels = self.prepare_batch_if_images(input_ids, attention_mask, inputs_embeds, labels, images)
            
        #print(attention_mask)
        #print(self.tok.batch_decode(labels.masked_fill(labels==-100, 100)))

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



if __name__ == "__main__":
    VISION_MODEL_ID = "openai/clip-vit-large-patch14"
    TEXT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    mnist = load_dataset('ylecun/mnist', split='train')
    samples = [i for i in mnist['image'][:2]]

    cfg = BeniConfig(
        vision_name_or_path = VISION_MODEL_ID,
        llm_name_or_path = TEXT_MODEL_ID,
        r = 16
    )
        
    beni = Beni(cfg)
    beni.to("cuda")

    tok = beni.tok

    print(f'VLM with: {sum(p.numel() for p in beni.parameters())/1e9:.1f}B params')

    sequences = ["once upon a time ", "long ago in a galaxy far far away"] 
    inputs = tok([s+tok.eos_token for s in sequences], return_tensors = 'pt', padding=True)
    inputs['labels'] = inputs['input_ids']

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(beni.device)

    inputs = {**inputs, 'images': samples}

    out = beni(**inputs)
