from dataclasses import dataclass
from typing import Callable, Optional, List, Union

from collections import OrderedDict
from PIL import Image

import torch
from torch import nn

from transformers import (
    PreTrainedModel,
    SiglipVisionModel,
)

from .perceiver import PerceiverResampler, PerceiverResamplerConfig
from . import sparsity


@dataclass
class VisionTowerConfig:
    r: int = 1
    img_size: int = 384
    # image_processors: Optional{AutoProcessor] for grid cropping etc
    use_cls: bool = False
    feature_select_index: int = -1
    perceiver_config: Optional[PerceiverResamplerConfig] = None
    sparsity_plugins: Optional[
        List[Union[sparsity.DragonFlyConfig, sparsity.GumbelConfig]]
    ] = None


class VisionTower(nn.Module):
    """
    todo:
        - add feature select index
        - trainable token pool ?
        - do we want the [cls] token as well?
            - **only applies to some models!"
    """

    def __init__(
        self,
        vision: PreTrainedModel,
        processor: Callable[[Image], torch.Tensor],
        config: VisionTowerConfig,
    ):
        # HERE WE ASSUME ONLY THE VISION MODEL IS PASSED. WE WILL SUBCLASS LATER WHEN USING TEXT
        super().__init__()
        self.feature_select_index = config.feature_select_index  # add to config
        self.cls_idx = 0 if config.use_cls else 1
        self.r = config.r

        # for clip/siglip models
        self.vision = vision

        # update hidden size in vision model config
        setattr(
            self.vision.config, "hidden_size", self.r * self.vision.config.hidden_size
        )

        # image processors
        self.image_processor = self.get_image_processors(processor, config.img_size)

        # perceiver resampler
        if config.perceiver_config is not None:
            assert (
                self.r == 1
            ), "Do not reshape image patches when using perceiver resampler!"
            self.resampler = PerceiverResampler(config.perceiver_config)
        else:
            self.resampler = lambda x, _: nn.Identity()(x)  # no-op

        # sparsity: Tensor -> Tensor signature
        if config.sparsity_plugins is not None:
            od = OrderedDict(
                (
                    (sp.name.lower(), getattr(sparsity, sp.name).build(sp))
                    for sp in config.sparsity_plugins
                )
            )
            self.sparsity = nn.Sequential(od)
        else:
            self.sparsity = nn.Identity()  # no-op

    @property
    def device(self):
        return self.vision.device

    @property
    def torch_dtype(self):
        dtypes = set((p.dtype for p in self.vision.parameters()))
        assert len(dtypes) == 1
        return list(dtypes)[0]

    def get_image_processors(self, processor, img_size: int):
        self.is_high_res = img_size != processor.size

        if isinstance(processor.size, dict):
            processor.size = {"height": img_size, "width": img_size}
        elif isinstance(processor.size, int):
            processor.size = img_size

        return processor

    def unfreeze(self):
        if isinstance(self.resampler, PerceiverResampler):
            for p in self.resampler.parameters():
                p.requires_grad = True
        if isinstance(self.sparsity, nn.Sequential):
            for p in self.sparsity.parameters():
                p.requires_grad = True

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/vision.py#L94
    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ):
        """
        TODO: for clip models
        """
        raise NotImplementedError

    def vit_forward(self, x):
        try:
            x = self.image_processor(x, return_tensors="pt")["pixel_values"]  # type: ignore
        except:
            # debug (sometimes we get single channel images)
            print(x)
            raise RuntimeError

        fwd_kwargs = {}
        if self.is_high_res:
            if not isinstance(self.vision, (SiglipVisionModel,)):
                raise RuntimeError(
                    "Trying to upscale images to size {self.vision.config.size} without known `interpolate_pos_encodings` impl in model forward"
                )
            fwd_kwargs["interpolate_pos_encoding"] = True

        x = self.vision(
            x.to(self.device, self.torch_dtype), output_hidden_states=True, **fwd_kwargs
        )
        x = x["hidden_states"][self.feature_select_index][:, self.cls_idx :, :]

        return x

    def forward(self, x, attention_mask=None):
        x = self.vit_forward(x)
        x = self.sparsity(x)

        if attention_mask is None:
            attention_mask = torch.ones(
                x.shape[:2], dtype=torch.long, device=self.device
            )

        x = self.resampler(x, attention_mask)
        b, s, _ = x.shape

        # concatenate adjacent tokens a la minigpt4-v2
        return x.reshape((b, s // self.r, -1))


if __name__ == "__main__":
    from .perceiver import PerceiverResamplerConfig
    from transformers import SiglipVisionModel, SiglipImageProcessor
    import io
    from PIL import Image
    import requests

    vision_name_or_path = "google/siglip-so400m-patch14-384"

    img_size = 384

    cfg = VisionTowerConfig(r=1, use_cls=True, img_size=img_size, sparsity_plugins=None)

    vision = SiglipVisionModel.from_pretrained(vision_name_or_path)
    processor = SiglipImageProcessor.from_pretrained(vision_name_or_path)
    vt = VisionTower(vision, processor, cfg)

    img = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.unsplash.com/photo-1486365227551-f3f90034a57c?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyZHxlbnwwfHwwfHx8MA%3D%3D"
            ).content
        )
    ).convert("RGB")

    out = vt([img])
