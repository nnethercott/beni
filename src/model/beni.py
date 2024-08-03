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


# adapt later
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

        # text
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name_or_path)

    @property
    def text_config(self):
        return AutoConfig.from_pretrained(self.config.llm_name_or_path)
    @property
    def vision_config(self):
        return self.vision.config

    @property
    def device(self):
        return self.vision.device

    # stolen 
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
       
        if images is not None:
            vision_embeddings = self.vision(images)
            vision_embeddings = self.connector(vision_embeddings)
            
            if input_ids is not None:
                text_embeddings = self.llm.model.embed_tokens(input_ids)
                inputs_embeds = torch.cat((vision_embeddings, text_embeddings), dim=1)
                input_ids = None  # ensure input_ids is not passed to the model
            else:
                inputs_embeds = vision_embeddings

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
    sample = mnist[0]['image']

    cfg = BeniConfig(
        vision_name_or_path = VISION_MODEL_ID,
        llm_name_or_path = TEXT_MODEL_ID,
        r = 4
    )
        
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    beni = Beni(cfg)
    beni.to("cuda")

    print(f'VLM with: {sum(p.numel() for p in beni.parameters())/1e9:.1f}B params')

    inputs = tok("this is a sentence", return_tensors = 'pt')
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(beni.device)

    inputs = {**inputs, 'images': sample}

    out = beni(**inputs)



    
