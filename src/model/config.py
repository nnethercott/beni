from dataclasses import dataclass, field 
from typing import Any
import torch
from torch import nn 
import torch.nn.functional as F

@dataclass
class VLMConfig:
    vision_config: Any = None
    text_config: Any = None

    @property
    def num_hidden_layers(self):
        k = 'num_hidden_layers'
        return getattr(self.vision_config, k) + getattr(self.text_config, k)


class VLM(nn.Module):
    def __init__(self, vision, text)->nn.Module:
       self.vision = vision
       self.text = text
       super().__init__()
        
       # TODO: add code for hijacking llm to add cross attention layers 
