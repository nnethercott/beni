from .datasets import *
from .d import prepare
from .utils import *


# def get_train_dataloader(tokenizer, model_config, train_config):
#     seed = 42 + dist.get_rank() if dist.is_initialized() else 42  # hope this helps

#     # synthdog
#     data_en = load_synthdog(
#         tokenizer,
#         split="train",
#         repo_name="podbilabs/synthdog-podbi-en",
#         instruction_template=model_config.instruction_template,
#         response_template=model_config.response_template,
#     )
#     data_en = data_en.shuffle(seed=seed, buffer_size=1000)
#     data_en = StreamingDataset(
#         data_en,
#         n=20,
#         batch_size=train_config.batch_size,
#         collate_fn=functools.partial(sft_collate_fn, tok=tokenizer),
#     )
#     dl = MultiDataLoader(data_en, seed=seed)

#     return dl
