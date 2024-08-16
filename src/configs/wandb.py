# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from dataclasses import dataclass, field, asdict
import wandb 

@dataclass
class WandbConfig:
    project: str = 'beni' # wandb project name
    entity: Optional[str] = "nnethercott" # wandb entity name
    enable: bool = False
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None

    def build_run(self, configs={}, master_rank = False):
        self_dict = asdict(self)
        if not master_rank:
            return None

        if self_dict.pop('enable'):
            return wandb.init(**self_dict, config = configs)
        else:
            return None
