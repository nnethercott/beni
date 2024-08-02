from transformers import HfArgumentParser
from configs import *

if __name__ == "__main__":
    parser = HfArgumentParser((fsdp_config, train_config))
    f,t = parser.parse_args_into_dataclasses()

    print(f)
    print(t)
