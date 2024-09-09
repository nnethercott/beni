from dataclasses import dataclass
from typing import Type

import torch
from torch import nn
import torch.nn.functional as F


def sparsity_plugin(plugin_class: Type):
    def decorator(cls):
        cls = dataclass(cls)
        cls.name = plugin_class.__name__
        return cls

    return decorator


class GumbelSoftmaxSparsityPlugin(nn.Module):
    def __init__(self, config):
        super().__init__()
        # maybe replace this with param so we can do more fine-grained stuff like normalized dot product

        # more complicated than basic linear
        self.h = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2, bias=False),
        )
        self.temperature = config.temperature
        self.prob = config.p
        self.lam = config.lam

        self.losses = None

    @classmethod
    def build(cls, config, **kwargs):
        return cls(config)

    @staticmethod
    def sample_gumbel(x):
        u = torch.empty_like(x).uniform_(0, 1)
        g = -torch.log(-torch.log(u))
        return g

    def register_loss(self, p, eps=1e-05):
        """
        batch mean over images
        """
        p = p.mean(dim=1)
        # self.losses = self.lam*(1/self.prob - 1/(p+ 1e-05)).pow(2).mean()
        x = p - self.prob
        self.losses = self.lam * (x * F.tanh(6 * x)).mean()  # smooth-ish approx to abs
        return

    def forward(self, x):
        """
        TODO: when !self.train remove patches and make an attention_mask
        """
        logits = self.h(x)
        g = GumbelSoftmaxSparsityPlugin.sample_gumbel(logits)
        logits += g

        # only keep the first storing P({event keep token x})
        p = F.softmax(logits / self.temperature, dim=-1)
        p = p[:, :, 0].unsqueeze(-1)

        self.register_loss(1 - p)
        return p * x, None


# not trainable
class BilinearInterpolationSparsityPlugin(nn.Module):
    # llava-next
    # F.interpolate(patches, size=size, mode="bilinear", align_corners=False)
    pass


@sparsity_plugin(GumbelSoftmaxSparsityPlugin)
class GumbelConfig:
    hidden_size: int
    temperature: float = 0.1
    p: float = 0.3
    requires_grad: bool = True
    lam: float = 1.0
    trainable: bool = True
