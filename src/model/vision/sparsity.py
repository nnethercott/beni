from dataclasses import dataclass, field
from typing import Type, List

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
class DragonFlySparsityPlugin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.num_patches = config.num_patches
        self.img_sizes = config.img_sizes
        self.vit_stride = config.vit_stride
        self.use_cls = config.use_cls

        assert (
            (self.img_sizes[-1] // self.vit_stride) ** 2
            >= self.num_patches * self.top_k
        ), f"trying to select {self.num_patches*self.top_k} patches from image with {(self.img_sizes[-1]//self.vit_stride)**2} total patches"

    @classmethod
    def build(cls, config, **kwargs):
        return cls(config)

    @torch.no_grad()
    def forward(self, x):
        """
        use self.img_sizes to split and patch select
        ensure num patches common divisor of img_sizes
        """
        i = 1 - int(self.use_cls)
        num_low_res_patches = (self.img_sizes[0] // self.vit_stride) ** 2 - i
        low_res = x[:, :num_low_res_patches, :]
        high_res = x[:, num_low_res_patches:, :]

        bsz, seq_len, d = low_res.shape

        low_res = low_res.view(bsz, self.num_patches, -1, d)
        low_res = low_res / low_res.norm(dim=-1, p=2, keepdim=True)

        high_res = high_res.view(bsz, self.num_patches, -1, d)
        high_res = high_res / high_res.norm(dim=-1, p=2, keepdim=True)

        # normalized inner product
        inner = low_res.mean(dim=2, keepdim=True) @ high_res.transpose(-1, -2)
        _, idx = inner.topk(self.top_k, dim=-1)
        idx = idx.transpose(-1, -2).expand(
            -1, -1, -1, d
        )  # [bsz, num_pathes, top_k, 1] -> [bsz, num_pathes, top_k, d]
        high_res = high_res.gather(2, idx)

        high_res = high_res.view(bsz, -1, d)
        low_res = low_res.view(bsz, -1, d)

        return torch.cat((low_res, high_res), dim=1)


# register all plugins
@sparsity_plugin(DragonFlySparsityPlugin)
class DragonFlyConfig:
    top_k: int = 64
    num_patches: int = 4
    img_sizes: List[int] = field(default_factory=lambda: [384])
    vit_stride: int = 14
    use_cls: bool = False
    trainable: bool = False


@sparsity_plugin(GumbelSoftmaxSparsityPlugin)
class GumbelConfig:
    hidden_size: int
    temperature: float = 0.1
    p: float = 0.3
    requires_grad: bool = True
    lam: float = 1.0
    trainable: bool = True
