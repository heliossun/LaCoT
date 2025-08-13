
export OPENAI_API_KEY="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"








  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_ftVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
  # --tasks mlvu \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mmmu_val \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-0.5b-lora-v1.5-wsft,conv_template=qwen_cot,model_name=sqllava_qwen-lora_0.5b-gfn \
#    --tasks mme \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

  #   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_cot \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava_qwen_0.5b \
  #  --tasks mmstar \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot

  #  CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_cot \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava_qwen_0.5b \
  #  --tasks scienceqa \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
  #   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_cot \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava_qwen_0.5b \
  #  --tasks olympiadbench_testmini \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_cot \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava_qwen_0.5b \
#   --tasks egothink \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/cot
#    CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_cot \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava_qwen_0.5b \
#   --tasks worldqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-1e5,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-0.5b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-0.5-gfn \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot


# accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathvision_reason_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot




   


#     accelerate launch -m lmms_eval \
#    --model qwen25_cot_bon \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn
#   accelerate launch -m lmms_eval \
#    --model qwen25_cot_bon \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

#   accelerate launch -m lmms_eval \
#    --model qwen25_cot_bon \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mmmu_val_thinking \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn



  #  accelerate launch -m lmms_eval \
  #  --model qwen25_cot_bon \
  #  --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
  #  --tasks mmmu_pro_vision_cot \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn

#   accelerate launch -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
#    --tasks mmmu_pro_vision_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/zeroshot

#   accelerate launch -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
#    --tasks mme \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/zeroshot

# accelerate launch -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
#    --tasks mmvet \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/zeroshot

   accelerate launch -m lmms_eval \
   --model qwen2_5_vl_gfn \
   --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-7b-v1.8-1epo-skip8-exp6-1024 \
   --tasks mathvision_reason_test \
   --batch_size 1 \
   --log_samples \
   --log_samples_suffix sqllava_ov \
   --output_path ./logs/gfn

   accelerate launch -m lmms_eval \
   --model qwen2_5_vl_gfn \
   --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-7b-v1.8-1epo-skip8-exp6-1024 \
   --tasks mmmu_pro_vision_cot \
   --batch_size 1 \
   --log_samples \
   --log_samples_suffix sqllava_ov \
   --output_path ./logs/gfn


     accelerate launch -m lmms_eval \
   --model qwen2_5_vl_gfn \
   --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-7b-v1.8-1epo-skip8-exp6-1024 \
   --tasks mathvista_testmini_cot \
   --batch_size 1 \
   --log_samples \
   --log_samples_suffix sqllava_ov \
   --output_path ./logs/gfn


     accelerate launch -m lmms_eval \
   --model qwen2_5_vl_gfn \
   --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-7b-v1.8-1epo-skip8-exp6-1024 \
   --tasks mmmu_val_thinking \
   --batch_size 1 \
   --log_samples \
   --log_samples_suffix sqllava_ov \
   --output_path ./logs/gfn


#  accelerate launch -m lmms_eval \
#    --model qwen25_cot_bon \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks llava_in_the_wild \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn


#   accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn


#   accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

#  accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks llava_in_the_wild \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn


# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
  

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=ZachSun/Qwen2.5-gfn-7B \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=2 python -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-3b-250k \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=3 python -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-7b-250k \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot


#   accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mmmu_val_thinking \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
  #  accelerate launch -m lmms_eval \
  #  --model qwen2_5_vl_gfn \
  #  --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-3b-250k \
  #  --tasks mathvision_reason_test \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot


  #    accelerate launch -m lmms_eval \
  #  --model qwen2_5_vl_gfn \
  #  --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-7b-250k \
  #  --tasks mathverse_testmini_vision_only \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot

   
  # accelerate launch -m lmms_eval \
  #  --model qwen2_5_vl_gfn \
  #  --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-7b-250k \
  #  --tasks mathvision_reason_test \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
  # accelerate launch -m lmms_eval \
  #  --model qwen2_5_vl_gfn \
  #  --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-3b-250k \
  #  --tasks mmmu_val_thinking \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
# accelerate launch -m lmms_eval \
#    --model qwen2_5_vl_gfn \
#    --model_args pretrained=/mnt/disks/new-disk/Qwen2-VL-gfn/output/Qwen2.5-gfn-3b-v1.8 \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot


# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/R1-OV-RL \
#    --tasks mmmu_val_thinking \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/R1-OV-RL \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

#    CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model qwen2_5_vl \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/R1-OV-RL \
#    --tasks mathvista_testmini_cot \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-7b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-7-gfn \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-0.5b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-0.5-gfn \
#    --tasks mathvision_reason_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

  #  CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-0.5b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-0.5-gfn \
  #  --tasks mmmu_val_thinking \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot

#    CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model qwen2_5_vl_cot \
#    --model_args pretrained=ZachSun/Qwen2.5-gfn-sft-7b-250k \
#    --tasks mmmu_val_thinking \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

   
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_cot,model_name=llava-qwen-7b \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-7b-1e5-v2,conv_template=qwen_cot,model_name=llava-qwen-7b \
#    --tasks mmmu_val \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llava-gfn-sft-7b-1e5-v2,conv_template=qwen_cot,model_name=llava-qwen-7b \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot


#CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-0.5b-lora-v1.5-wsft,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_0.5b-gfn \
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/gfn
#
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_gfn \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-0.5b-lora-v1.5-wsft,conv_template=qwen_cot,model_name=sqllava_qwen-lora_0.5b-gfn \
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/gfn
#
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_cot \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-0.5b-lora-v1.5-wsft,conv_template=qwen_cot,model_name=sqllava_qwen-lora_0.5b-gfn \
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/gfn
# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/ours-0.5b-qwen-lora-spo-fxVit-newloss,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_0.5b_spo\
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/



  



# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=ZachSun/sqllava-qwen-0.5b-interleave,conv_template=qwen_1_5,model_name=llava_qwen-lora_0.5b \
#   --tasks pope \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/0.5b

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=ZachSun/sqllava-qwen-0.5b-interleave,conv_template=qwen_1_5,model_name=llava_qwen-lora_0.5b \
#   --tasks llava_in_the_wild \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/0.5b

  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/ours-0.5b-qwen-lora-sdo-lmd50-b0.1-lr1e5-alpa1-2epo-new,conv_template=qwen_1_5,model_name=llava_qwen-lora_0.5b \
  # --tasks pope \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/0.5b
# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks nextqa_mc_test\
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/spobest

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks live_bench_2406\
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/spobest


#   CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks perceptiontest_val_mc\
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/spobest

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmstar \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmmu_val \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks llava_interleave_bench_out_domain \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks longvideobench_val_ \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks muirbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks llava_wilder_small \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
 

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks llava_interleave_bench_out_domain \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks longvideobench_val_v \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks muirbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/



# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks egoschema \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks mlvu \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mvbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks perceptiontest_val_mc \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 



# egoschema upload
#curl -X POST -H "Content-Type: application/json" -d @logs/submissions/inference_results_egoschema_MC_2024-09-19-11-13-12.json https://validation-server.onrender.com/api/upload