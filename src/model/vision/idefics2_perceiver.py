import torch
from torch import nn 
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple, Optional
import math

from transformers.models.idefics2.modeling_idefics2 import Idefics2PerceiverResampler as PerceiverResampler
from transformers.models.idefics2.configuration_idefics2 import Idefics2PerceiverConfig, Idefics2Config 
from transformers import SiglipTextConfig

if __name__ == "__main__":
    perceiver_config = Idefics2PerceiverConfig(resampler_depth = 1)
    siglip_text_config = SiglipTextConfig.from_pretrained("google/siglip-so400m-patch14-384")
    c = Idefics2Config(text_config = siglip_text_config, perceiver_config=perceiver_config)
    p = PerceiverResampler(c)
