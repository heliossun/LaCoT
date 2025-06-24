#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="ZachSun/Qwen2.5-gfn-sft-7b-250k"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=1

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
deepspeed src/training/train_gfn.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /mnt/disks/new-disk/data/gfn-3k.json\
    --image_folder /mnt/disks/new-disk/data/cot \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/Qwen2.5-gfn-7b-lora-v1.8-1epo \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 10 \
    --dataloader_num_workers 16\
    --skip_reward_step 8 \
    --explore_nums 6 \
    --explore_min_bs 6 \
    --rat_max_len 700 \
    --rat_min_len 64 \
    --reward_tolarent_start 1.5 \
    --reward_tolarent_end 1 \
    --reward_tolarent_horizon 50 \
    --reward_sched_horizon 50 \
    --reward_sched_start 1.0 \
    --reward_sched_end 0.7 \