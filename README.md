# beni 
## resources
### repos
* [llama-recipes](https://github.com/meta-llama/llama-recipes) (fsdp tools, activation checkpointing, mixed precision, wrapper for lora, overall holy grail)
* [TinyLlama](https://github.com/jzhang38/TinyLlama) (system monitor for tracking gpu usage)
* [llava](https://github.com/haotian-liu/LLaVA) 
* [cambrian](https://github.com/cambrian-mllm/cambrian) (vision projector design)
* [dragonfly](https://github.com/togethercomputer/Dragonfly) (multi crop vision)
* [minigptv](https://github.com/Vision-CAIR/MiniGPT-4) (concat of visual tokens, llava-style archi)

### other 
* [estimating transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)
* [making gpus go brrr](https://horace.io/brrr_intro.html)


## feature roadmap:
* [x] convert to using llama-recipes training framework [25/07/24]
* [x] fsdp infra for multi-gpu training [29/07/24]
    * ran fsdp fine-tuning on [tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) for a 0.5B llama from scratch in 10 minutes
    * full state dict checkpointing + lora checkpoints 
* [x] lazy dataset for downloading images when collating [04/07/24]
* [x] llava-style vision-text connector [07/08/24]
    * supports any vision encoder and llm available on huggingface
* [ ] cross-attention vlm archi 
* [ ] support generic propmt templates in dataset module & model forward
* [ ] alt quantization schemes (e.g. HQQ) 
* [ ] DLoRA 
* [ ] play around with sequence packing in train
