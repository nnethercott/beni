# beni 
`beni` is a repo for mixing and matching vision backbones with LLMs to align text and vision embedding spaces. This gives the llm an "understanding" of images. For more technical details on this process check out [this blog post](https://medium.com/@natenethercott/vision-language-models-from-scratch-in-colab-cd073a753b8a). 

Here's some results you can get in an hour: 
![image (5)](https://github.com/user-attachments/assets/bc5b6ecd-07d4-47a4-a984-1ad34c7bf38d)

## resources
### repos
* [llama-recipes](https://github.com/meta-llama/llama-recipes) (fsdp tools, activation checkpointing, mixed precision, wrapper for lora, overall holy grail)
* [TinyLlama](https://github.com/jzhang38/TinyLlama) (system monitor for tracking gpu usage)
* [llava](https://github.com/haotian-liu/LLaVA) 
* [cambrian](https://github.com/cambrian-mllm/cambrian) (vision projector design)
* [dragonfly](https://github.com/togethercomputer/Dragonfly) (multi crop vision)
* [minigptv](https://github.com/Vision-CAIR/MiniGPT-4) (concat of visual tokens, llava-style archi)

### papers and articles 
* [fuzzy deduplication with minhash](https://blog.nelhage.com/post/fuzzy-dedup/)
* [estimating transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)
* [making gpus go brrr](https://horace.io/brrr_intro.html)
* [idefics2](https://arxiv.org/pdf/2405.02246)
* [llava-next ablations blog post](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/)
* [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
* [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training)


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
* [x] support generic propmt templates in dataset module & model forward [26/08/24]
* [x] alt quantization schemes (e.g. HQQ) 
    * natively supported now by huggingface
* [x] DLoRA 
* [x] perceiver resampler [14/08/24]
    * either use the idefics2 one or we use our own implementation (test both) 
* [ ] interpolate positional embeddings for high res images in vision encoder
    * native support exists for siglip 
    * adding high res on clip will require us making our own clip fork
    * add better check that `interpolate_pos_encoding` in vision.forward fn signature
* [x] `enable_inputs_require_grads` for lora training & fsdp 
* [x] make `BeniConfig` subclass hf config so we can peft beni and `.save_pretrained`
* [x] add a demo subdir with docker compose for spinning up model worker, flask server, and gradio app [15/08/24]
    * [this commit](https://github.com/Deepomatic/vlm_dev/commit/ffc2e11e57aaac8ec63679978cbedef44bba3e41)
* [ ] `torch.jit.trace` model forward pass and check idle vs nonidle time 
* [x] quantized/qlora training
    * need to make `quant_storage_type` same as `compute_dtype` so layers get flattened properly in fsdp
    * **need to add scaler and mixed precision decorators for llm forward call if quantization enabled**
* [ ] resolve the following; prompt token masking not yet configured for multi-turn conversations; image tokens inserted in a fixed location, no dynamic placement in user prompt 
* [ ] make connector architecture configureable from a string in the model config 
    * use transformers activation map 
* [ ] use meta's memory trace tool to get max cuda memory allocated and reallocations, etc
* [x] quantized generation [09/09/24] 
* [ ] post-training optimization 
    * torch jit compile, quantization, pruning, etc 
    * some starting points in this repo: [mervenoyan/smol-vision](https://github.com/merveenoyan/smol-vision), otherwise check out [torch docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
* [ ] enable eval on benchmarks like [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) using custom deepo vlm

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

vision_tower_config = VisionTowerConfig(
    r=8,
    feature_select_index=-1,  
    use_cls=False,
    img_size=384,
    sparsity_plugins=None,
    perceiver_config=perceiver_config,
)

model_config = BeniConfig(
    vision_name_or_path="google/siglip-so400m-patch14-384",
    text_name_or_path="stabilityai/stablelm-2-1_6b-chat",
    vision_tower_config=vision_tower_config,
    vision_cls="SiglipVisionModel",
    vision_processor_cls="SiglipImageProcessor",
    freeze=True,
    attn_implementation="eager",
    bos_token="<|user|>\n",  # offset needed for img token insert
    instruction_template="<|user|>\n{instruction}<|endoftext|>\n<|assistant|>\n",  # no loss part
    response_template="{response}<|endoftext|>\n",  # loss part
    llm_quantization_config = BitsAndBytesConfig(
       load_in_4bit = True,
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_quant_storage=torch.float16,
    ),
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


## fsdp gotchas
* transformers are made up of sequential blocks with residual connections. if we shard model parameters between gpus then a tensor op like $x+f(x)$ where `x.device=cuda:0` and `f(x).device=cuda:1` may break training. To resolve this we specify which layers to wrap in the fsdp config.  
    * if fine tuning an llm which has weight tying activated (`lm_head` = `embed_tokens`) then the embedding weights might simultaneously be needed to convert tokens to embeddings or hidden states to logits. this poses a race condition which breaks training. Solution in this case is to wrap the `nn.Embedding` layer in a fsdp module. 
        * in this case we need only wrap `nn.Embedding` and not the VisionTower
* a multimodal training dataset consisting of both image-text pairs and just text samples will activate different parts of the network. the `model.connector` has no gradients when pure text is used. thus the gradient all reduce in fsdp will break since we're trying to average out tensors between gpus which may not exist on others
    * the `data.utils.MultiLoader` was created to solve this (ensures each devices sees the same modality per iteration)
* original code for the training loop looked like:
```python
for e, batch in enumerate(dataloader):
    if batch is None: # batch can be none since we load images from web at runtime
        continue:
    ...
    if e%save_steps == 0:
        dist.barrier() # all ranks suspend execution until reaching this
        save_model(model)
```
the issue with this is that if we skip a batch one rank may encounter the `save_model` block before others and will not participate in fsdp model forward, thus leading to nccl watchdog timeout 
* fsdp shards must consist of uniform data types. techniques like llm quantization, where params are stored in low precision datatypes, do not play nice with fsdp. Thus QLoRA is made more difficult since we must store quantized weights in half precision (and convert model to half precision as well).
    * this acheives 2x memory reduction which is less than the llm_size/4 + vit_size/2 > 2x reduction we get from llm 4bit quantization normally 
    * this also means we need to scale and unscale half precision loss since `torch.float16` has a small dynamic range (otherwise nans). Ideally we'd use `torch.bfloat16` since it has 8bit exponents vs float16 5bit, but this only works on nvidia gpus with Ampere architecture or newer.
* [invalid device ordinal](https://stackoverflow.com/questions/64334033/how-to-solve-runtimeerror-cuda-error-invalid-device-ordinal) might occur if you hard set `CUDA_VISIBLE_DEVICES` manually ?
* `NCCL_SOCKET_IFNAME`: https://github.com/pytorch/pytorch/issues/29482
* if `dataloader_num_workers` too large, the asynchronous fetching of images may be faster than batches are consumed leading to an explosion in system RAM. Recommended values : {0,1}
    * conversely we spend a lot of time **waiting** for images to be read and `dataloader_num_workers` can reduce train time by 2-3x. If system has enough RAM and fast GPUs we can justify large values for this field


## deepspeed gotchas 
```python 
from deepspeed import get_accelerator
get_accelerator().is_fp16_supported()
```

## Best Practices
* if the model can fit on a single GPU use `ShardingStrategy.NO_SHARD` (equivalent to DDP). otherwise try `ShardingStrategy.SHARD_GRAD_OP` for small-ish models. By default we use `ShardingStrategy.FULL_SHARD`. 


## Training with vertex ai 
* either upload sdist of the project to a google bucket or use custom docker image 
* `AIP_CHECKPOINT_DIR` env variable defined if we specify [`baseOutputDirectory` API field](https://cloud.google.com/vertex-ai/docs/training/code-requirements)
* can run hyperparameter tuning job to find best meta architecture
