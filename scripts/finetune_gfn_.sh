#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="ZachSun/Qwen2.5-gfn-sft-3b-250k"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=1

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
 /opt/conda/envs/qwen2/bin/deepspeed src/training/train_gfn.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/hhua/guohao/data/instruct/gfn-10k.json \
    --image_folder /home/hhua/guohao/data/cot \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/Qwen2.5-gfn-3b-lora-v1.7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 10 \
    --dataloader_num_workers 16\
    --skip_reward_step 16 \
    --explore_nums 6 \
    --explore_min_bs 1 \
    --rat_max_len 700 \
    --rat_min_len 64 \
    --reward_tolarent_start 1.2 \
    --reward_tolarent_end 1 \
    --reward_tolarent_horizon 50 \
    --reward_sched_horizon 50 \
    --reward_sched_start 1.0 \
    --reward_sched_end 0.8 \