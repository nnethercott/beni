#!/bin/bash

set -e

exec torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes="$NNODES" \
    --node_rank="$RANK" \
    --master_addr="$ADDR" \
    --master_port="$PORT" \
    /app/src/vlm/hf_main.py "$@"
