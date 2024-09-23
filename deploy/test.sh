#!/bin/bash

docker run \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -e HF_TOKEN=${HF_TOKEN} \
    -e HF_HOME=${HF_HOME} \
    -v ${HF_HOME}:${HF_HOME} \
    --runtime=nvidia \
    vlm-test-img:latest


