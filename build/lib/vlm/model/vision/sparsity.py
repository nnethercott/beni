from dataclasses import dataclass, field
from typing import Type, Tuple, Union, List
import math

from torch import nn
import torch.nn.functional as F


def sparsity_plugin(plugin_class: Type):
    def decorator(cls):
        cls = dataclass(cls)
        cls.name = plugin_class.__name__
        return cls

    return decorator


# not trainable
class BilinearInterpolationSparsityPlugin(nn.Module):
    """
    The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area']. For more details, please refer to the discussion in issue #104157.
    """

    def __init__(self, config):
        super().__init__()
        self.size = config.size

    def forward(self, x):
        assert len(x.shape) == 4
        bsz, num_crops, seq_len, d = x.shape

        patch_h_w = int(math.sqrt(seq_len))

        # flatten batch dim for interpolation
        x = x.reshape(-1, d, patch_h_w, patch_h_w)  # [bsz x num_crops, d, h, w]

        # mini-batch x channels x [optional depth] x [optional height] x width.
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)

        # reshape to 3d tensor
        x = x.view(bsz, num_crops, -1, d)

        return x

    @classmethod
    def build(cls, config, **kwargs):
        return cls(config)


@sparsity_plugin(BilinearInterpolationSparsityPlugin)
class BilinearConfig:
    size: Union[Tuple[int, int], List[int]] = field(default_factory=lambda: (27, 27))
    trainable: bool = False
