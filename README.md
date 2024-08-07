# beni 
## resources
### repos
* [llama-recipes](https://github.com/meta-llama/llama-recipes) (fsdp tools, activation checkpointing, mixed precision, wrapper for lora, overall holy grail)
* [TinyLlama](https://github.com/jzhang38/TinyLlama) (system monitor for tracking gpu usage)
* [llava](https://github.com/haotian-liu/LLaVA)
* [cambrian](https://github.com/cambrian-mllm/cambrian) (vision projector design)
* [dragonfly](https://github.com/togethercomputer/Dragonfly) (multi crop vision)

### other 
* [estimating transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)

### vlm design next steps
* read llama3 paper section on this
* copy CA block from idefics? or just use llama decoder layer ... 

## roadmap:
- [ ] get prompt templates working

## general todo:
- [x] run fsdp fine tuning on [tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) as sanity check 
    * full ft + lora, model checkpointing
- [x] setup gpu profiling 
    * extremely low mfu, might be since we're using huggingface models
- [x] convert to using llama-recipes training framework
- [ ] add alt quantization schemes (e.g. HQQ) to `src/configs/quantization.py`
- [ ] explore/add DLoRA 
- [ ] play around with sequence packing in train
- [ ] custom finetune script based on llama-recipes (for some reason their's is slower than ours so far, maybe reduction ops?)
