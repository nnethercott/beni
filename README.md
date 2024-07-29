# beni 
some useful repos:
* [llama-recipes](https://github.com/meta-llama/llama-recipes) (fsdp tools, activation checkpointing, mixed precision, wrapper for lora, overall holy grail)
* [TinyLlama](https://github.com/jzhang38/TinyLlama) (system monitor for tracking gpu usage)
* [llava](https://github.com/haotian-liu/LLaVA)
* [cambrian](https://github.com/cambrian-mllm/cambrian) (vision projector design)
* [dragonfly](https://github.com/togethercomputer/Dragonfly) (multi crop vision)

timeline:
- [ ] run *fsdp* fine tuning on [tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) to check it works, (change config.num_layers for small llama)


features to implement:
- [ ] [resource monitor](https://github.com/jzhang38/TinyLlama/blob/bf122247c486b6b897050e98cbb7bedae8eeba73/lit_gpt/speed_monitor.py#L15) to determine if code memory or compute bound 
- [x] checkpointing/loading for fsdp  
- [ ] mixed precision (newer architectures get huge speedups here too)
- [ ] **LoRA in fsdp**
- [ ] mixed precision & scaler 
- [ ] **activation checkpointing**
