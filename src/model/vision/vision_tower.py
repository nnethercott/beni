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

from .perceiver import PerceiverResampler

class VisionTower(nn.Module):
    """
    todo:
        - add feature select index
        - trainable token pool ?
        - do we want the [cls] token as well?
            - **only applies to some models!"
    """
    def __init__(self, vision: PreTrainedModel, processor: Callable[[Image],torch.Tensor], config):
        # HERE WE ASSUME ONLY THE VISION MODEL IS PASSED. WE WILL SUBCLASS LATER WHEN USING TEXT
        super().__init__()
        self.feature_select_index = -1
        self.r = config.r 

        # for clip/siglip models
        self.vision = vision
        self.image_processor = processor

        # reshape sequence
        setattr(self.vision.config, 'hidden_size', self.r*self.vision.config.hidden_size)

        # image resolution
        img_size = config.img_size
        if isinstance(img_size, int) and isinstance(self.image_processor.size, dict):
            img_size = {'height': img_size, 'width': img_size}
        elif isinstance(img_size, dict) and isinstance(self.image_processor.size, int):
            img_size = image_size.get('height', None)
        else:
            pass

        self.is_high_res = img_size != self.image_processor.size
        self.image_processor.size = img_size #no-op if img_dims is None

        # perceiver resampler 
        if config.perceiver_config is not None:
            self.resampler = PerceiverResampler(config.perceiver_config)
        else:
            self.resampler = lambda x, _: nn.Identity()(x)


    @property
    def device(self):
        return self.vision.device

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/vision.py#L94
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        pass


    def vit_forward(self, x):
        x = self.image_processor(x, return_tensors = 'pt')['pixel_values']

        fwd_kwargs = {}
        if self.is_high_res:
            if not isinstance(self.vision, (SiglipVisionModel,)):
                raise RuntimeError("Trying to upscale images to size {self.vision.config.size} without known `interpolate_pos_encodings` impl in model forward")
            fwd_kwargs['interpolate_pos_encoding'] = True
        
        x = self.vision(x.to(self.device), output_hidden_states=True, **fwd_kwargs)
        x = x['hidden_states'][self.feature_select_index][:,1:,:]

        return x


    def forward(self, x, attention_mask = None):
        # vision-backbone
        x = self.vit_forward(x)

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=self.device)

        x = self.resampler(x, attention_mask)
        b,s,_ = x.shape

        # concatenate adjacent tokens a la minigpt4-v2
        return x.reshape((b, s//self.r, -1))  



if __name__ == "__main__":
    from ..beni import BeniConfig
    from .perceiver import PerceiverResamplerConfig
    from transformers import SiglipVisionModel, SiglipImageProcessor
    import io
    from PIL import Image
    import requests

    perceiver_config = PerceiverResamplerConfig(
            hidden_size = 1152, 
            depth = 3, 
            n_latents = 16,
            n_query_groups=4,
            n_heads = 32,
            head_dim = 96,
            concat_latents_kv = False,
            attention_dropout = 0.1,
    )

    cfg = BeniConfig(
        perceiver_config = perceiver_config,
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        r = 1,
        img_size = 224,
    )

    vision = SiglipVisionModel.from_pretrained(cfg.vision_name_or_path)
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_name_or_path)
    vt = VisionTower(vision, processor, cfg)

    img = Image.open(io.BytesIO(requests.get("https://c.files.bbci.co.uk/C8AF/production/_120357315_moshpit_crowdsurfer_gettyimages_976.jpg").content)).convert("RGB")   
    out = vt([img,img])


