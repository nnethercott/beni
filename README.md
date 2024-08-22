# beni 
## resources
### repos
* [llama-recipes](https://github.com/meta-llama/llama-recipes) (fsdp tools, activation checkpointing, mixed precision, wrapper for lora, overall holy grail)
* [TinyLlama](https://github.com/jzhang38/TinyLlama) (system monitor for tracking gpu usage)
* [llava](https://github.com/haotian-liu/LLaVA) 
* [cambrian](https://github.com/cambrian-mllm/cambrian) (vision projector design)
* [dragonfly](https://github.com/togethercomputer/Dragonfly) (multi crop vision)
* [minigptv](https://github.com/Vision-CAIR/MiniGPT-4) (concat of visual tokens, llava-style archi)

### papers and articles 
* [estimating transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)
* [making gpus go brrr](https://horace.io/brrr_intro.html)
* [idefics2](https://arxiv.org/pdf/2405.02246)

## feature roadmap:
* [x] convert to using llama-recipes training framework [25/07/24]
* [x] fsdp infra for multi-gpu training [29/07/24]
    * ran fsdp fine-tuning on [tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) for a 0.5B llama from scratch in 10 minutes
    * full state dict checkpointing + lora checkpoints 
* [x] lazy dataset for downloading images when collating [04/07/24]
* [x] llava-style vision-text connector [07/08/24]
    * supports any vision encoder and llm available on huggingface
* [x] v0 llava-style vlm trained [11/08/24]
* [x] multi dataloader and length-ordered datasets for optimized training [11/08/24]
* [ ] cross-attention vlm archi 
* [ ] support generic propmt templates in dataset module & model forward
* [ ] alt quantization schemes (e.g. HQQ) 
* [ ] DLoRA 
* [ ] play around with sequence packing in train
* [x] perceiver resampler [14/08/24]
    * either use the idefics2 one or we use our own implementation (test both) 
* [ ] interpolate positional embeddings for high res images in vision encoder
    * adding high res on clip will require us making our own clip fork
    * add better check that `interpolate_pos_encoding` in vision.forward fn signature
* [x] `enable_inputs_require_grads` for lora training & fsdp 
* [x] make `BeniConfig` subclass hf config so we can peft beni and `.save_pretrained`
* [x] add a demo subdir with docker compose for spinning up model worker, flask server, and gradio app [15/08/24]
    * [this commit](https://github.com/Deepomatic/vlm_dev/commit/ffc2e11e57aaac8ec63679978cbedef44bba3e41)
* [ ] something important i forgot
* [ ] check perf of normal computer vision models like resnet
* [ ] `torch.jit.trace` model forward pass and check idle vs nonidle time 

# Usage
Training is configured parametrically using various config objects which get passed to a launch script. These configs allow for defining optional model components (e.g. the `PerceiverResamplerConfig`) and control aspects like model architecture and training hyperparameters.

An example training configuration is shown below;

```python
perceiver_config = PerceiverResamplerConfig(
    hidden_size = 1152, # matches vision.hidden_size
    depth = 1, 
    n_latents = 64,
    n_query_groups=1,
    n_heads = 16,
    head_dim = 96,
    concat_latents_kv = True,
    attention_dropout = 0.0,
)

model_config = BeniConfig(
    perceiver_config = perceiver_config,
    vision_name_or_path = "google/siglip-so400m-patch14-384",
    text_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    vision_cls = "SiglipVisionModel",
    vision_processor_cls = "SiglipImageProcessor",
    freeze = True,
    attn_implementation = "eager",
    img_size = 384,
    r = 1
    sparsity_plugins = None,
)

train_config = TrainConfig(
    warmup_ratio = 0.03,
    batch_size = 8,
    gradient_accumulation_steps = 1,
    lr = 1e-04,
    weight_decay = 0.1,
    min_lr = 1e-05,
    grad_clip = 1.0,
    save_steps = 5,
    log_steps = 1,
    save_path = "./model_checkpoints/test",
    ckpt_path = None,
    betas = [0.9, 0.95],
    fsdp=True,
    enable_peft=True,
)

fsdp_config = FSDPConfig(
        transformer_cls=(LlamaDecoderLayer, SiglipEncoderLayer, PerceiverResampler), 
)

wandb_config = WandbConfig(
    enable = true,
    project = "your-project",
    entity = "your-username",
)

lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias = 'none')
```

This structure aims to let users quickly iterate with different architectural choices to find the best combo for the task at hand. 

We mainly use the hugging face API in this project which assumes models have specific properties and/or methods (e.g. embedding dimension usually indexed with `hidden_size`, llms have both a `generate` and `forward` defined). Of course this won't be the case for all models you want to train with, but it should cover a wide variety of models hosted on HF.

## ideas
Test these with same random seed over same datasets for fixed training steps
* img feature selection
    * dragonfly-like topk based on cosine similarity with embedding of user prompt 
        * initially we use 2 resolutions and know a-priori how many total patches we want (this or we start padding image token sequences -> bad for runtime since we already made the assumption when optimizing with dynamic collate; chill tho if perceiver resamplaer)
        * when no text present we'd need some trainable vector as the null, or uniform patching 
        * either toss direct into model or into resampler
    * region proposal network & crops
    * perceiver resampler lets us compress an arbitrary sequence into a fixed length 
        * can use arbitrary length concatenated sequences of different crop features, or threshold-based dragonfly 
        * grouped query attention perceiver resampler ??
        
* [gated attention](https://arxiv.org/pdf/1912.00349) might be cool as an alternative to cosing of prompt and image patch. 
    * could draw patches ~ p where p=sigmoid(cosine(prompt, patch))
    * might be more or less sparse than threshold based patch selection, still don't know
    * math idea: $p(z) \sim \text{Beta}(\alpha, \beta)$ w/ $\beta > \alpha$ so that $\mathbb{E}[z] = \frac{\alpha}{\alpha+\beta} <0.5$. Then we have $z = \phi(x)$ so that $p(z|x)\sim \text{Beta}(\alpha+k, \beta+n-k)$ (n=bsz) ??
        * first lets sample using the gumbel softmax trick and observe the sparsity. after that we can try to impose a prior on how many patches should be active using the beta prior/bernoulli formulation 
        * [blog post](https://www.johndcook.com/blog/2009/11/24/kumaraswamy-distribution/) and [cross validated post](https://stats.stackexchange.com/questions/51820/fast-approximation-to-inverse-beta-cdf) on psuedo-beta sampling
            * this is actually **hard** since the kumaraswamy dist requires larger and larger b to minimize moments between corresponding beta distribution -> gamma(b) blows up => we need to work in log space => we need some inequalities
            * possible solution is to minimize $(\log{f(a,b)}-\log{g(a,c)})^{2}$ but thats not the same as miniminizing $(f(a,b)-g(a,c))^{2}$
        * the auxiliary network would map the image patch to $[\alpha, \beta] \in \mathbb{R}^{2}$
        * i think we'd have to do a pretraining to determine the weights of the gating network and then when we finetune for real we can igore any token which is "turned off" by dropping it from the sequence. otherwise we can't jointly train the network on dynamically-determined variable-length sequences 
        * *need to override the positional embeddings of zeroed out patches during pretraining i think* maybe

        

        

        

## Notes
* when using llms with weight tying (lm_head weights = nn.Embedding) we **must** pass the torch.nn.Embedding in the wrapped transformer layers, otherwise fsdp blocks !
* not all llms have default sos and eos
* when training distributed, if processes see a batch of different modalities then our connector may not be active => gradients=None => all_reduce step blocking since one rank attemps to aggregate gradients which don't exist in other processes ?
    * could probably avoid this (statistically) with grad accumulation and hope each rank sees both images & text
    * **or** make sure modality per iter is the same for each rank
* on multimodal training data we should grad accumulate so that gradients contain info pertaining to all modalities before stepping (i think)
    * adamw maintains state so we have this mix-of-modalities step implicitly, but its best to be more clear
