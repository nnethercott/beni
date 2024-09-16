## notes 
* idefics2 conclusions
    * says we get boosted perf from gradual LoRA training of llm instead of full unfreeze
    * says decoder-only outperforms cross attention
    * better perf by using llms and vision models at top of respective leaderboards (imagenet, open llm leaderboard, etc)
    * has references to some ocr datasets

* llava-next blog post 
    * data mix per stage mentioned [here](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts/train#about-the-llava-onevision-data)
    * "model size scaling of LLM is more effective than image encoder in yielding improved performance"
        * would be cool to try this one tho [OpenGVLab/InternViT-6B-448px-V1-2](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2)
    * [NOTE] get langage detector model to filter chinese texts
    * "We found that the vision encoder's learning rate should always be 10x or 5x smaller than the LM decoder's learning rate to stabilize training"
        * this is for *full training* though 
        * they use values in the range (2e-05, 2e-06) for (llm, vit)
    * for ViTs [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main) shows best performance 
        * outperforms other ViTs of size 8B 
    * batch sie of 64 or 128 ok 
    * #crops in mid range doesn't increase training time too badly (2x2 6h30 -> 4x4 7h30 -> 6x6 11h), on high end might double training time
        * high crops increases perf (obvi), but decent even in low range
        * less influential than llm size it seems
    * [NOTE] could use feature upscaler like [FeatUp](https://mhamilton.net/featup.html)  to uspcale images before crops ?
    * upscaling image resolution significantly increases training time without improving perf when using AnyRes
        * for single crop models inc. imagre res is beneficial though 
        * although higher res without modifying #tokens improves OCR
    * bilinear interpolation better than pooling 
        * could ablate which interplation strategy is best
    * [NOTE] llava-next uses dynamic grids depending on image aspect ratio; for a v1 we'll just stick with a 2x2 grid
