# datasets
We use a mix of huggingface and torch dataset utils. First datasets for vision-language are loaded in from huggingface and preprocessed using the convenient `dataset.Dataset.map` method. Next, we use a `CustomDatasetForImages` class which overrides the `__getitem__` dunder to load the image from the url - if this fails it increments `i+=1`. 

Advantages of this setup:
* images are only loaded **once** during training instead of twice (once for filter, once during training)
    * saves time => saves money
* clearer API using a custom dataset class instead of obfuscated huggingface preprocessing

Disadvantages:
* we increment the offset property of the dataset class which may modify the effect of `__len__` in the dataloader init
    * should be OK if big dataset
    * TODO: estimate % of unloadable mages


We'll also **need** to sort rows by seq length to get better value and speed out of our gpus

