# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Idefics2 model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers import AutoModel
from transformers.models.idefics2.configuration_idefics2 import Idefics2Config, Idefics2VisionConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


from transformers import PretrainedConfig
class PerceiverResamplerConfig(PretrainedConfig):

    def __init__(
        self, 
        hidden_size: int,
        rms_norm_eps: float = 1e-06,
        n_latents: int = 64,
        hidden_act: str = "silu",
        depth: int = 1,
        n_heads: int = 32, 
        head_dim: int = 96,
        n_query_groups: int = 1,
        concat_latents_kv: bool = False,
        attention_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.rms_norm_eps = rms_norm_eps 
        self.n_latents=n_latents
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.depth = depth
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_query_groups = n_query_groups
        self.concat_latents_kv = concat_latents_kv
        self.attention_dropout = attention_dropout


#@dataclass
#class PerceiverResamplerConfig:
#    hidden_size: int 
#    rms_norm_eps: float = 1e-06
#    n_latents: int = 64
#    hidden_act: str = "silu"
#    depth: int = 1
#    n_heads: int = 32
#    head_dim: int = 96
#    n_query_groups: int = 1
#    concat_latents_kv: bool = False
#    attention_dropout: float = 0.0


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))



# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics2
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()

        self.layer_idx = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        self.num_query_groups = config.n_query_groups
        self.attention_dropout = config.attention_dropout
        self.concat_latents_kv = config.concat_latents_kv

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_query_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_query_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = context.size()[1]

        if self.concat_latents_kv:
            kv_seq_len += q_len 
            hidden_states = torch.concat([context, latents], dim=-2)
        else:
            hidden_states= context

        q = self.q_proj(latents)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)


        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        # repeat k,v enough times so we can shove into F.scaled_dot_product_attention
        if q.shape != k.shape:
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PerceiverFlashAttention(PerceiverAttention):
    """
    Idefics2 flash attention module. This module inherits from `PerceiverAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Ignore copy
    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        # Query, Key, Value Projections --> Note that in Flamingo, latents are *concatenated* with context prior to attn!
        #   Note: This results in queries w/ `seq = n_latents`, and keys, values with `seq = len(context) + n_latents`
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, kv_seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            if hasattr(self.config, "sliding_window") and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window

                past_key = past_key_value[0]
                past_value = past_key_value[1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        "past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1,"
                        f" head_dim`), got {past_key.shape}"
                    )

                past_key_value = (past_key, past_value)

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_query_groups)
        v = repeat_kv(v, self.num_query_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = q.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = _flash_attention_forward(
            q,
            k,
            v,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=None,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


IDEFICS2_PERCEIVER_ATTENTION_CLASSES = {
    "eager": PerceiverAttention,
    "flash_attention_2": PerceiverFlashAttention,
}


class PerceiverLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_latents = config.n_latents
        self.depth = config.depth
        self.rms_norm_eps = config.rms_norm_eps

        self.input_latents_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PerceiverResampler(nn.Module):
    def __init__(self, config) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. The Resampler acts as a form of learned pooling and
        is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206).
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.n_latents = config.n_latents
        self.depth = config.depth
        self.rms_norm_eps = config.rms_norm_eps
        self.concat_latents_kv = config.concat_latents_kv 

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.ones(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = nn.ModuleList([PerceiverLayer(config, idx) for idx in range(self.depth)])
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        latents = self.latents.unsqueeze(0).expand((context.shape[0], *self.latents.size()))

        if self.concat_latents_kv:
            latent_attention_mask = torch.ones(
                (attention_mask.size(0), latents.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)

        attention_mask = (
            _prepare_4d_attention_mask(attention_mask, latents.dtype, tgt_len=self.n_latents)
            if not self._use_flash_attention_2
            else attention_mask
        )

        compressed_context = latents
        for perceiver_layer in self.layers:
            layer_outputs = perceiver_layer(
                compressed_context,
                context,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            compressed_context = layer_outputs[0]

        compressed_context = self.norm(compressed_context)

        return compressed_context

if __name__ == "__main__":
    from ..beni import BeniConfig

    c = PerceiverResamplerConfig(hidden_size = 512, depth = 3, n_latents = 32)
    p = PerceiverResampler(c)

    print(p)
    print(sum((i.numel() for i in p.parameters()))/1e9)

    samples = torch.rand(6, 384, 512)
    a = [[1]*384 for _ in range(6)]
    out = p(samples,a)
    print(out.shape)

