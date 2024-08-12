from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Union
from PIL import Image
import math

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
    SiglipVisionModel,
)


class VisionTower(nn.Module):
    """
    todo:
        - add feature select index
        - trainable token pool ?
        - do we want the [cls] token as well?
            - **only applies to some models!"
    """
    def __init__(self, vision: PreTrainedModel, processor: Callable[[Image],torch.Tensor], r: int = 1, **kwargs):
        super().__init__()
        self.vision = vision 
        self.feature_select_index = -1
        self.r = r 

        if hasattr(processor, "image_processor"):
            self.processor = processor.image_processor  # instantiated with AutoProcessor 
        else:
            self.processor = processor 

        # update self.vision.config 
        setattr(self.config, 'hidden_size', r*self.vision.config.hidden_size)

        img_size = kwargs.get("img_size", getattr(self.processor, 'size', None))
        self.is_high_res = img_size != self.processor.size
        self.processor.size = img_size #no-op if img_dims is None


    @property
    def device(self):
        return self.vision.device

    @property
    def config(self):
        # if model loaded with automodel 
        if hasattr(self.vision.config, "vision_config"):
            return self.vision.config.vision_config
        else:
            return self.vision.config


    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/vision.py#L94
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        # get rid of this in favor of dedicated visiontowers
        pass

    def forward(self, x):
       x = self.processor(x, return_tensors = 'pt')['pixel_values']

       fwd_kwargs = {}

       # check image size to determine pos interpolate 
       # TODO: support interpolation for clip models & check interpolate_pos_encoding in fn signature
       if self.is_high_res:
           if not isinstance(self.vision, (SiglipVisionModel,)):
               raise RuntimeError("Trying to upscale images to size {self.config.size} without known `interpolate_pos_encodings` impl")
           fwd_kwargs['interpolate_pos_encoding'] = True


       x = self.vision(x.to(self.device), output_hidden_states=True, **fwd_kwargs)
       x = x['hidden_states'][self.feature_select_index][:,1:,:]
       b,s,d = x.shape

       # concatenate adjacent tokens a la minigpt4-v2
       #print(x.shape)
       return x.reshape((b, s//self.r, -1))  
