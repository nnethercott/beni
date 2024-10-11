export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true 

NUM_GPUS=${NUM_GPUS:-4}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
ADDR=${ADDR:-'127.0.0.1'}
PORT=${PORT:-'29501'}

RUN_NAME=${RUN_NAME:-'vlm'}

AIP_MODEL_DIR=${AIP_MODEL_DIR:-'/mnt/nate/nate-again/some-uuid/model'}
AIP_MODEL_DIR_CLEAN=$(echo "$AIP_MODEL_DIR" | sed 's#gs://#/gcs/#g') # gs:// -> /gcs/
GCP_BUCKET_ROOT=$(echo "$AIP_MODEL_DIR_CLEAN" | sed -E 's#/[^/]+/?$##')

RUN_DIR="${GCP_BUCKET_ROOT}"
mkdir $RUN_DIR > /dev/null 2>&1

echo $AIP_MODEL_DIR_CLEAN
echo $RUN_DIR

ifconfig>"${RUN_DIR}/ifconfig.txt"

# --deepspeed scripts/zero2_fused_adamw.json\
# --use_liger_kernel false  # not in transformers==4.44.*
# --fsdp_activation_checkpointing false \
# --fsdp_cpu_offload false \
# --fsdp_sharding_strategy "full_shard" \

#accelerate launch ./src/vlm/hf_main.py \
torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" ./src/vlm/hf_main.py \
    --deepspeed "./scripts/zero2_offload.json" \
    --output_dir ${AIP_MODEL_DIR_CLEAN}\
    --text_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --vision_name_or_path="google/siglip-so400m-patch14-384" \
    --n_concat_tokens 9 --vision_cls="SiglipVisionModel" --freeze true --unfreeze_lm_head false --attn_implementation "eager"\
    --feature_select_index -1 --use_cls true --img_size 384 --use_global_crop false \
    --instruction_template "<s>user:\n{instruction}" \
    --lora_r 4 --lora_alpha 32 --lora_target_modules "q_proj,k_proj,v_proj,o_proj" --enable_peft true \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 4e-04 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 4e-05}' \
    --weight_decay 0.0 \
    --label_names "labels" \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --save_only_model true \
    --save_steps 10000 \
    --seed 42 \
    --optim "paged_adamw_8bit" \
    --report_to "none" \
    --run_name "${RUN_NAME}" \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --tf32 false \
    --group_by_length false \
    --ddp_find_unused_parameters false \
    --half_precision_backend "auto" \
    --fp16 true 2>&1 | tee "${RUN_DIR}/log.txt"
