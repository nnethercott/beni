## image patch selection tentatives
**Workflow:** implement new strategy in dedicated branch, test perf on limited compute buget for ocr dataset

Strategies:
* single high res **1 crop**
* low and high res patch features - **2 crops**
    * global image (maybe 224x224) and sparse selection of high res (672x672) ?
        i) crop into patches of large size before encoding
        ii) feed whole high res image
    * maintain a crop(s) with same aspect ratio?
    * techniques like dragonfly 
* **multi-crop**
    * cambrian spatial vision aggregator 
        * potentially requires higher compute budget to learn pooling 
        * originally recomputed in spaced transformer blocks - we'd do it once before passing to model
    * *train RPN* and work with these crops (complimentary to 2+ crops strategies)

![alt-text](https://cambrian-mllm.github.io/static/img/sva.png)

## notes 
* idefics2 conclusions
    * says we get boosted perf from gradual LoRA training of llm instead of full unfreeze
    * says decoder-only outperforms cross attention
    * better perf by using llms and vision models at top of respective leaderboards (imagenet, open llm leaderboard, etc)
    * has references to some ocr datasets

