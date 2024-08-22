from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Union, ClassVar, Type
from collections import OrderedDict
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
from . import sparsity

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
        self.feature_select_index = config.feature_select_index  #add to config
        self.cls_idx = 0 if config.use_cls else 1
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
            assert self.r == 1, 'Do not reshape image patches when using perceiver resampler!'
            self.resampler = PerceiverResampler(config.perceiver_config)
        else:
            self.resampler = lambda x, _: nn.Identity()(x) #no-op

        # sparsity
        if config.sparsity_plugins is not None:
            od = OrderedDict(((sp.name.lower(), getattr(sparsity, sp.name).build(sp)) for sp in config.sparsity_plugins))
            self.sparsity = nn.Sequential(od)
        else:
            self.sparsity = lambda x: (nn.Identity()(x), None)

        # self.p = 1
        self._losses = None


    @property
    def device(self):
        return self.vision.device

    #@property # not working??
    def get_losses(self):
        """
        if trainable feature selection/sparsity networks present
        recover their losses here
        """
        losses = torch.tensor(0., device = self.device)

        if isinstance(self.sparsity, nn.Sequential):
            for n, m in self.sparsity.named_children():
                if hasattr(m, 'losses'):
                    losses+=m.losses
        
        if losses.is_nonzero():
            self._losses = losses

        return self._losses


    def unfreeze(self):
        if isinstance(self.resampler, PerceiverResampler):
            for p in self.resampler.parameters():
                p.requires_grad = True
        if isinstance(self.sparsity, nn.Sequential):
            for p in self.sparsity.parameters():
                p.requires_grad = True


    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/vision.py#L94
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        raise NotImplementedError

    def vit_forward(self, x):
        x = self.image_processor(x, return_tensors = 'pt')['pixel_values']

        fwd_kwargs = {}
        if self.is_high_res:
            if not isinstance(self.vision, (SiglipVisionModel,)):
                raise RuntimeError("Trying to upscale images to size {self.vision.config.size} without known `interpolate_pos_encodings` impl in model forward")
            fwd_kwargs['interpolate_pos_encoding'] = True
        
        x = self.vision(x.to(self.device), output_hidden_states=True, **fwd_kwargs)
        x = x['hidden_states'][self.feature_select_index][:,self.cls_idx:,:] #note: siglip doesnt have [cls] -> add to config

        return x

    def forward(self, x, attention_mask = None):
        # vision-backbone
        x = self.vit_forward(x)
        x, attention_mask = self.sparsity(x)

        # DEBUG
        mask = x.mean(dim=-1).abs().detach().cpu()<1e-03
        self.p = mask.sum()/(mask.shape[0]*mask.shape[1])

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

    gumbel = sparsity.GumbelConfig(hidden_size=1152, temperature=0.1, p=0.3, lam=0.5)

    cfg = BeniConfig(
        perceiver_config = None,
        vision_name_or_path = "google/siglip-so400m-patch14-384",
        text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vision_cls = "SiglipVisionModel",
        vision_processor_cls = "SiglipImageProcessor",
        r = 1, 
        img_size = 384,
        sparsity_plugins = [gumbel],
    )

    vision = SiglipVisionModel.from_pretrained(cfg.vision_name_or_path)
    processor = SiglipImageProcessor.from_pretrained(cfg.vision_name_or_path)
    vt = VisionTower(vision, processor, cfg)

    # load gumbel params from disk
    path = "/home/nnethercott/beni/src/model_checkpoints/perceiver_test_p03/test-sparsitiy-step4389.pt"
    vt.sparsity.load_state_dict(torch.load(path))

    img = Image.open(io.BytesIO(requests.get("https://images.unsplash.com/photo-1486365227551-f3f90034a57c?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyZHxlbnwwfHwwfHx8MA%3D%3D").content)).convert("RGB")   
    out = vt([img])

    # get ids of zeroed tokens
    mask = out.mean(dim=-1).abs()<1e-03

    from PIL import ImageDraw 
    img = img.resize((384,384)) # match preprocessing
    draw = ImageDraw.Draw(img)
    h,w = img.size

    ids = torch.arange(mask.shape[1]).unsqueeze(0).repeat((len(mask), 1))
    ids = ids[mask].tolist()
    num_patches = 384//14

    # lower corners of the patch in question
    xx,yy = torch.meshgrid(14*torch.tensor(range(num_patches)), 14*torch.tensor(range(num_patches))) # might be an error here
    patches = list(zip(xx.flatten().tolist(), yy.flatten().tolist()))

    # todo: repeat patches so its the same len as bsz
    patches = [patches[i] for i in ids] # list of tensors
    
    for patch in patches:
        draw.rectangle((patch,(patch[0]+14, patch[1]+14)), 'black')

    img.save('./model/vision/test3.jpg')

    # could do a histogram over different images to see distrbution 
    
    











