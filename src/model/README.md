## todo
- [ ] get llava-style vlm working with small llm 
    * minigpt-type reshape of vision component ? 
    * large patch size for fewer v-tokens
- [ ] cross attention impl

## notes 
* idefics2 conclusions
    * says we get boosted perf from gradual LoRA training of llm instead of full unfreeze
    * says decoder-only outperforms cross attention
    * better perf by using llms and vision models at top of respective leaderboards (imagenet, open llm leaderboard, etc)
    * has references to some ocr datasets

