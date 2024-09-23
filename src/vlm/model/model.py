from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Union, Any
from PIL import Image
import os
import dacite
import copy

import torch
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from peft import PeftModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .vision import VisionTower, VisionTowerConfig


def rank_0_only(f):
    def wrap(*args, **kwargs):
        if not torch.distributed.is_initialized() or int(os.environ["LOCAL_RANK"]) == 0:
            return f(*args, **kwargs)

    return wrap


# transformer-like
# make sure to wrap in fsdp
# class Connector(nn.Module):
#    def __init__(self, d_in, d_out):
#        super().__init__()
#
#        self.norm_1 = nn.LayerNorm(d_in)
#        self.proj_1 = nn.Sequential(
#            nn.Linear(d_in, d_in),
#            nn.GELU(),
#        )
#        self.norm_2 = nn.LayerNorm(d_in)
#        self.proj_2 = nn.Linear(d_in, d_out)
#
#    def forward(self, x):
#        x = x + self.proj_1(self.norm_1(x))
#        return self.proj_2(self.norm_2(x))


class Connector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)

        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)
        )

    def forward(self, x):
        return self.proj(self.norm(x))


# @dataclass
# class VLMConfig(PretrainedConfig):
#    model_type = "beni"
#    is_composition = True
#
#    def __init__(
#            self,
#            perceiver_config: PretrainedConfig = None,
#            vision_name_or_path: str = None,
#            text_name_or_path: str = None,
#            vision_cls: str = None,
#            text_cls: str = 'AutoModelForCausalLM',
#            vision_processor_cls: str = None,
#            r: int = 4,
#            freeze: bool = True,
#            attn_implementation: str = "eager",
#            img_size: Optional[Union[dict, int]]=None,
#            **kwargs,
#    ):
#        super().__init__(**kwargs)
#
#        self.perceiver_config = perceiver_config
#        self.vision_name_or_path=vision_name_or_path
#        self.text_name_or_path=text_name_or_path
#        self.vision_cls=vision_cls
#        self.vision_processor_cls=vision_processor_cls
#        self.text_cls=text_cls
#        self.r=r
#        self.freeze=freeze
#        self.attn_implementation=attn_implementation
#        self.img_size=img_size
#
#        self._post_init()
#
#    def _post_init(self):
#        self.text_config = AutoConfig.from_pretrained(self.text_name_or_path)
#        self.vision_config = AutoConfig.from_pretrained(self.vision_name_or_path)
#
#        if hasattr(self.vision_config, 'vision_config'):
#            self.vision_config = self.vision_config.vision_config


@dataclass
class VLMConfig:
    text_name_or_path: str
    vision_name_or_path: str
    vision_tower_config: VisionTowerConfig
    vision_cls: Optional[str] = "AutoModel"
    text_cls: Optional[str] = "AutoModelForCausalLM"
    vision_processor_cls: Optional[str] = "AutoImageProcessor"
    freeze: bool = True
    attn_implementation: str = "eager"  # add flash_attention_2 when we get it
    text_config = None
    vision_config = None
    bos_token: Optional[Union[str, List[int]]] = "<s>user:"
    instruction_template: str = "<s>user:\n{instruction}\n</s>assistant:\n"
    response_template: str = "{response}</s>"
    llm_quantization_config: Optional[Any] = None

    def to_dict(self):
        # bit more work needs to go in here
        config_copy = copy.deepcopy(self)
        perceiver_config = config_copy.vision_tower_config.perceiver_config
        if perceiver_config is not None:
            perceiver_config = perceiver_config.to_dict()

        config_copy = asdict(config_copy)
        config_copy["vision_tower_config"]["perceiver_config"] = perceiver_config

        return config_copy

    @classmethod
    def from_dict(cls, config_dict):
        # FIXME later -> might not be able to load PretrainedConfig ?
        return dacite.from_dict(cls, config_dict)


class VLM(nn.Module):
    """
    Torch module representing a vision-language model.
    [Example goes here with some other comments]
    """

    def __init__(self, config: VLMConfig, hf_token=None):
        super().__init__()
        self.config = config
        self.token = hf_token

        self.build_vision_and_text(config)

        self.connector = Connector(
            self.vision_config.hidden_size, self.text_config.hidden_size
        )

        if self.config.freeze:
            self.freeze()

    def build_vision_and_text(self, config):
        vision_cls = getattr(transformers, config.vision_cls, "AutoModel")
        vision_processor_cls = getattr(
            transformers, config.vision_processor_cls, "AutoProcessor"
        )
        text_cls = getattr(transformers, config.text_cls, "AutoModelForCausalLM")

        # vision
        v = vision_cls.from_pretrained(  # type: ignore
            config.vision_name_or_path,
            attn_implementation=config.attn_implementation,
        )
        p = vision_processor_cls.from_pretrained(config.vision_name_or_path)  # type: ignore
        self.vision = VisionTower(v, p, config.vision_tower_config)

        # text
        self.llm = text_cls.from_pretrained(  # type: ignore
            config.text_name_or_path,
            attn_implementation=config.attn_implementation,
            quantization_config=self.config.llm_quantization_config,
            # torch_dtype = torch.float16,
            token=self.token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_name_or_path, token=self.token
        )
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_tokens(["<img>", "</img>"])

        # resize embeddings and lm_head https://huggingface.co/docs/transformers/en/main_classes/model
        self.llm.resize_token_embeddings(len(self.tokenizer))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False  # freeze all params

        for p in self.connector.parameters():
            p.requires_grad = True  # connector trainable

        self.vision.unfreeze()  # vision trainable

    @property
    def text_config(self):
        return AutoConfig.from_pretrained(
            self.config.text_name_or_path, token=self.token
        )

    @property
    def vision_config(self):
        return self.vision.vision.config

    @property
    def img_token(self):
        return self.tokenizer.encode("<img>", add_special_tokens=False)[0]

    @property
    def end_img_token(self):
        return self.tokenizer.encode("</img>", add_special_tokens=False)[0]

    @property
    def device(self):
        return self.vision.device

    @property
    def embed_tokens(self):
        if isinstance(self.llm, PeftModel):
            return self.llm.model.model.embed_tokens
        return self.llm.model.embed_tokens

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[Union[torch.FloatTensor, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,  # in reality these are pil images
    ):
        """
        TODO:
            * handle case where input_embeds passed

        NOTE: tokenizer right padding assumed for this to work
        * pad input_ids, attention_mask, and labels
        * need to decide on a prompt template
            * <s><img>image_tokens_here</img>user_prompt_and_answer</s> ?
        """

        if images is None:
            return input_ids, attention_mask, inputs_embeds, labels

        bsz = len(images)

        vision_embeds = self.vision(images)
        vision_embeds = self.connector(vision_embeds)

        # bos
        bos = self.config.bos_token
        if bos is not None:
            if isinstance(bos, list):
                tokens = bos
            else:
                tokens = self.tokenizer.encode(bos, add_special_tokens=False)
                if not isinstance(tokens, list):
                    tokens = [tokens]
            bos_token = torch.tensor(tokens, device=self.device)
            bos_len = len(tokens)
        else:
            bos_token = torch.tensor([self.tokenizer.bos_token_id], device=self.device)
            bos_len = 1

        bos_embeds = self.embed_tokens(bos_token).repeat((bsz, 1, 1))

        # embed <img> and </img>
        img_token = torch.tensor(self.img_token, device=self.device).unsqueeze(0)
        img_embeds = self.embed_tokens(img_token).repeat((bsz, 1, 1))
        end_img_token = torch.tensor(self.end_img_token, device=self.device).unsqueeze(
            0
        )
        end_img_embeds = self.embed_tokens(end_img_token).repeat((bsz, 1, 1))

        if input_ids is not None:
            _, vis_len, _ = vision_embeds.shape
            text_embeds = self.embed_tokens(input_ids)

            # <bos_seq><img>insert_image_featuers<img>prompt</s>
            inputs_embeds = torch.cat(
                (
                    bos_embeds,
                    img_embeds,
                    vision_embeds,
                    end_img_embeds,
                    text_embeds[:, bos_len:, :],
                ),
                dim=1,
            )

            # sanity again
            # full_seq_en = torch.cat(
            #     (
            #         bos_token.repeat((bsz, 1)),
            #         img_token.repeat((bsz, 1)),
            #         306
            #         * torch.ones(
            #             (bsz, vis_len), dtype=input_ids.dtype, device=self.device
            #         ),
            #         end_img_token.repeat((bsz, 1)),
            #         input_ids[:, bos_len:],
            #     ),
            #     dim=-1,
            # )
            # for seq in self.tokenizer.batch_decode(full_seq_en):
            #     print(seq)

            input_ids = None  # ensure input_ids is not passed to the model

            # attention_mask
            additional_len = 1 + 1  # added <img> and </img>
            attention_mask = torch.cat(  # type: ignore
                (  # type: ignore
                    torch.ones((bsz, vis_len + additional_len), device=self.device),
                    attention_mask,
                ),
                dim=1,
            )

            # if we're computing loss
            if labels is not None:
                labels = labels[:, 1:]
                additional_len += 1  # <s> we just got rid of^
                labels_prefix = torch.tensor(
                    [-100] * (vis_len + additional_len), device=self.device
                )

                # IMPORTANT
                # NOTE: this assumes all samples in batch have same number of image tokens
                labels_prefix = labels_prefix.repeat((bsz, 1))
                labels = torch.cat((labels_prefix, labels), dim=1)

        else:
            inputs_embeds = torch.cat(
                (bos_embeds, img_embeds, vision_embeds, end_img_embeds), dim=1
            )
            attention_mask = torch.ones(inputs_embeds.shape[:-1], device=self.device)

        return None, attention_mask, inputs_embeds, labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,  # in reality these are pil images
        return_dict: Optional[bool] = None,
        **kwargs,  # debug
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        passes input embeddings directly to llm if images present. otherwise use input_ids
        """
        assert (
            input_ids is not None or images is not None
        ), "You can't forward without text and/or images!"

        # print(self.tokenizer.batch_decode(input_ids))
        input_ids, attention_mask, inputs_embeds, labels = self.prepare_inputs(
            input_ids, attention_mask, inputs_embeds, labels, images
        )

        # print(attention_mask)
        # print(
        #     self.tokenizer.batch_decode(
        #         labels.masked_fill(
        #             labels == -100,
        #             self.tokenizer.encode("X", add_special_tokens=False)[0],
        #         )
        #     )
        # )

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (_, attention_mask, inputs_embeds, _) = self.prepare_inputs(
                input_ids,
                attention_mask,
                None,
                None,
                images,
            )
            input_ids = None

        else:
            # NOTE: we'll never generate with a peft model -- always merge first so this should hold
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        # print(inputs_embeds.shape)
        # print(attention_mask.shape)

        return self.llm.generate(
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(1, "../")
    from data import *
    import io
    import requests
    from peft import PeftModel
    from .vision import PerceiverResamplerConfig, BilinearConfig

    perceiver_config = PerceiverResamplerConfig(
        hidden_size=1152,  # from siglip.config
        depth=3,
        n_latents=64,
        n_query_groups=1,
        n_heads=32,
        head_dim=64,
        concat_latents_kv=False,
        attention_dropout=0.1,
    )

    vision_tower_config = VisionTowerConfig(
        r=1,
        feature_select_index=-1,
        use_cls=True,
        img_size=384,
        sparsity_plugins=[BilinearConfig(size=(6, 6))],
        perceiver_config=None,
        grid=(2, 2),
    )

    model_config = VLMConfig(
        vision_name_or_path="google/siglip-so400m-patch14-384",
        text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vision_tower_config=vision_tower_config,
        vision_cls="SiglipVisionModel",
        vision_processor_cls="SiglipImageProcessor",
        freeze=True,
        attn_implementation="eager",
        bos_token="<s>",
        instruction_template="<s>{instruction}</s>",
        response_template="{response}</s>",
    )

    # override
    import json
    from checkpointing import load_model

    with open("/mnt/nate/model_checkpoints/ref/model_config.json", "r") as f:
        config_dict = json.loads(f.read())
    model_config = VLMConfig.from_dict(config_dict)

    model = VLM(model_config)
    model = load_model(model, "/mnt/nate/model_checkpoints/ref/", trainable=False)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = {}
    img = Image.open(
        io.BytesIO(
            requests.get(
                "https://titanicfacts.net/wp-content/uploads/2018/06/titanic-deep-atlantic.jpg"
            ).content
        )
    ).convert("RGB")
    print(img)

    sentence = "provide a descriptive caption for this image"
    # sentence = ""

    template = model_config.instruction_template
    inputs = model.tokenizer(
        template.format(instruction=sentence),
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs["labels"] = inputs["input_ids"]
    inputs["images"] = [img]

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    # out = beni(**inputs)
    print("generating...")
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        # temperature=0.7,
        # top_p=0.95,
        num_beams=3,
        num_return_sequences=1,
        do_sample=False,
        eos_token_id=[
            model.tokenizer.eos_token_id,
            model.tokenizer.pad_token_id,
        ],
    )
    if "input_ids" in inputs.keys():
        print(model.tokenizer.batch_decode(inputs["input_ids"]))

    print(model.tokenizer.batch_decode(out))

    # from data import load_recap, sft_collate_fn
    # import functools
    # from torch.utils.data import DataLoader

    # data = load_allava_laion(
    #     beni.tokenizer,
    #     n=100,
    #     instruction_template=beni.config.instruction_template,
    #     response_template=beni.config.response_template,
    # )
    # inputs = next(
    #     iter(
    #         DataLoader(
    #             data,
    #             batch_size=4,
    #             shuffle=True,
    #             collate_fn=functools.partial(sft_collate_fn, tok=beni.tokenizer),
    #         )
    #     )
    # )
