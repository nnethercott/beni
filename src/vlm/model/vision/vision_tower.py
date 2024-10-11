from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torch import nn
import inspect

from transformers import (
    SiglipVisionModel,
    BaseImageProcessor,
    CLIPVisionModel,
)

from .perceiver import PerceiverResampler, PerceiverResamplerConfig
from . import sparsity


def grid_based_crop(img: Image.Image, grid: Tuple[int, int]):
    w, h = img.size
    crops = []
    dw, dh = w / grid[0], h / grid[1]

    for i in range(grid[0]):
        for j in range(grid[1]):
            box = (dw * i, dh * j, dw * (i + 1), dh * (j + 1))
            crops.append(img.crop(box))
    return crops


# maybe faster
def grid_based_crop_numpy(
    img: Image.Image, grid: Tuple[int, int], use_global: bool = False
):
    img_np = np.array(img)
    h, w, _ = img_np.shape
    dw, dh = w // grid[0], h // grid[1]
    crops = [img] if use_global else []

    for i in range(grid[0]):
        for j in range(grid[1]):
            # numpy slicing
            crop = img_np[dh * j : dh * (j + 1), dw * i : dw * (i + 1), :]
            crops.append(Image.fromarray(crop))
    return crops


@dataclass
class VisionTowerConfig:
    n_concat_tokens: int = 1
    img_size: int = 384
    use_cls: bool = False
    feature_select_index: int = -1
    perceiver_config: Optional[PerceiverResamplerConfig] = None
    sparsity_plugins: Optional[List[sparsity.BilinearConfig]] = None

    # NOTE: this can't get parsed by cmdline, need to make this string or something in hf
    grid: List[int] = field(
        default_factory=lambda: [1, 1]
    )  # json stores tuples as list

    use_global_crop: bool = False


class VisionTower(nn.Module):
    def __init__(
        self,
        vision: Union[SiglipVisionModel, CLIPVisionModel],
        processor: BaseImageProcessor,
        config: VisionTowerConfig,
    ):
        super().__init__()
        self.feature_select_index = config.feature_select_index  # add to config
        self.cls_idx = 0 if config.use_cls else 1
        self.r = config.n_concat_tokens
        self.grid = tuple(config.grid)
        self.use_global_crop = config.use_global_crop

        # for clip/siglip models
        self.vision = vision

        # update hidden size in vision model config
        setattr(
            self.vision.config, "hidden_size", self.r * self.vision.config.hidden_size
        )

        # image processors
        self.image_processor = self.get_image_processor(processor, config.img_size)

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
        # assert len(dtypes) == 1
        return list(dtypes)[0]

    def get_image_processor(
        self, processor, img_size: int
    ):  # might need to rework later
        self.is_high_res = img_size != processor.size

        if isinstance(processor.size, dict):
            processor.size = {"height": img_size, "width": img_size}
            img_size = {"height": img_size, "width": img_size}  # type: ignore
        elif isinstance(processor.size, int):
            processor.size = img_size

        # print(self.is_high_res)

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
        TODO: implement this for clip-based
        """
        raise NotImplementedError

    def pre_forward(self, x: List[Image.Image]) -> List[Image.Image]:
        # TODO: aspect ratio-grid cropping
        if self.grid == (1, 1):
            return x

        crops = (
            grid_based_crop_numpy(img, self.grid, self.use_global_crop) for img in x
        )  # type: ignore

        return [img for c in crops for img in c]  # flatten

    @torch.no_grad()
    def vit_forward(self, x: List[Image.Image]) -> torch.FloatTensor:
        # img to list of images
        x = self.image_processor(x, return_tensors="pt")["pixel_values"]

        fwd_kwargs = {}
        if self.is_high_res:
            assert (
                "interpolate_pos_encoding"
                in inspect.signature(self.vision.forward).parameters.keys()
            ), "vision encoder does not support pos interpolation; please run using default image size"
            fwd_kwargs["interpolate_pos_encoding"] = True

        x = self.vision(
            x.to(self.device, self.torch_dtype),  # type: ignore
            output_hidden_states=True,
            **fwd_kwargs,
        )
        x = x["hidden_states"][self.feature_select_index][:, self.cls_idx :, :]  # type: ignore

        # reshape tensor using self.grid
        a, b = self.grid
        i = 1 if self.use_global_crop else 0
        x = x.view(-1, a * b + i, *x.shape[-2:])  # type: ignore

        return x  # type: ignore

    @torch.no_grad()
    def forward(self, x, attention_mask=None):
        x = self.pre_forward(x)
        x = self.vit_forward(x)
        x = self.sparsity(x)

        # flatten 4d->3d [bsz, num_crops, seq_len, d] -> [bsz, num_crops x seq_len, d]
        bsz, _, _, d = x.shape
        x = x.reshape(bsz, -1, d)

        # NOTE: if we use dynamic img sizes we have to modify this attn mask
        if attention_mask is None:
            attention_mask = torch.ones(
                x.shape[:2], dtype=torch.long, device=self.device
            )

        x = self.resampler(x, attention_mask)
        b, s = x.shape[:2]
        # print(x.shape)

        # concatenate adjacent tokens a la minigpt4-v2
        return x.reshape((b, s // self.r, -1))


if __name__ == "__main__":
    from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
    import io
    from PIL import Image
    import requests
    from .perceiver import PerceiverResamplerConfig

    vision_name_or_path = "google/siglip-so400m-patch14-384"
    vconfig = SiglipVisionConfig.from_pretrained(vision_name_or_path)

    img_size = 384

    cfg = VisionTowerConfig(
        r=1,
        use_cls=True,
        img_size=img_size,
        # sparsity_plugins=[sparsity.BilinearConfig(size=(27, 27))],
        grid=(2, 2),
        use_global_crop=True,
        perceiver_config=PerceiverResamplerConfig(
            hidden_size=vconfig.hidden_size,
            n_latents=128,
            depth=3,
            hidden_act="silu",
            n_query_groups=1,
            concat_latents_kv=True,
            attn_implementation="eager",
        ),
    )

    vision = SiglipVisionModel.from_pretrained(vision_name_or_path)

    processor = SiglipImageProcessor.from_pretrained(vision_name_or_path)
    vt = VisionTower(vision, processor, cfg).to("cuda")  # type: ignore

    print(vt)

    img = Image.open(
        io.BytesIO(
            requests.get(
                "https://images.unsplash.com/photo-1486365227551-f3f90034a57c?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyZHxlbnwwfHwwfHx8MA%3D%3D"
            ).content
        )
    ).convert("RGB")

    out = vt([img])

    # def timing(f):
    #     def wrap(*args, **kwargs):
    #         now = time.time()
    #         res = f(*args, **kwargs)
    #         later = time.time()
    #         print(f"{f.__name__}: {later - now:.4f}")
    #         return res

    #     return wrap

    # @timing
    # def dummy(x):
    #     out = vt([img] * x)
    #     out.cpu()
    #     del out
    #     torch.cuda.empty_cache()

    # for i in [1, 3, 5, 8, 10]:
    #     dummy(i)
