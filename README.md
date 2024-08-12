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
* [ ] perceiver resampler to replace minigpt token concat?
* [ ] interpolate positional embeddings for high res images in vision encoder
    * adding high res on clip will require us making our own clip fork
    * add better check that `interpolate_pos_encoding` in vision.forward fn signature
* [ ] `enable_inputs_require_grads` for lora training & fsdp 
* [ ] make `BeniConfig` subclass hf config so we can peft beni and `.save_pretrained`


## Notes
* when using llms with weight tying (lm_head weights = nn.Embedding) we **must** pass the torch.nn.Embedding in the wrapped transformer layers, otherwise fsdp blocks !
* not all llms have default sos and eos
* when training distributed, if processes see a batch of different modalities then our connector may not be active => gradients=None => all_reduce step blocking since one rank attemps to aggregate gradients which don't exist in other processes ?
    * could probably avoid this (statistically) with grad accumulation and hope each rank sees both images & text
    * **or** make sure modality per iter is the same for each rank
