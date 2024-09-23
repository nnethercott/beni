# datasets
We use a mix of huggingface and torch dataset utils. First datasets for vision-language are loaded in from huggingface and preprocessed using the convenient `dataset.Dataset.map` method. 
Next, we use a `CustomDatasetForImages` class which overrides the `__getitem__` dunder to load the image from the url - if this fails it returns None for the image.
here we have multiple choices: 

Advantages of this setup:
* images are only loaded **once** during training instead of twice (once for filter, once during training)
    * saves **a lot** of time => saves money
* clearer API using a custom dataset class instead of obfuscated huggingface preprocessing to download images

Disadvantages:
* slower


# next steps:
* We'll also **need** to sort rows by seq length to get better value and speed out of our gpus


# idea:
- setup dataset for distributed context; each process tries to download images then communicates to other ones the length of their dataset. take the min 
    - add option for saving downloaded images or not (modify internal flag so behaviour on dataloader known at runtime)
