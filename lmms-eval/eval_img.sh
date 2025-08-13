
export OPENAI_API_KEY="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"

# mme, mathvista_testmini, realworldqa, egoschema,mmvet,mmmu_val





#CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#  --model llava_ovsq \
#  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llavaCoT-ov-lora-qwen-7b-3e6-fixViT/,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b_sft \
#  --tasks mathvista_testmini_cot \
#  --batch_size 1 \
#  --log_samples \
#  --log_samples_suffix sqllava_ov \
#  --output_path ./logs/cot
 


  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mmstar \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn

 

  #    CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks scienceqa \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn

  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_cot \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-sft-ov \
  #  --tasks mmstar \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot


# CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=llava_qwen-lora-7b-sft-ov \
#    --tasks olympiadbench_testmini \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mathverse_testmini_vision_only \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn
  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mmstar \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn
  #   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mathverse_testmini_vision_only \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-7b-lora-v1.7-3epo,conv_template=qwen_cot,model_name=llava-qwen-lora-7-gfn \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

#  CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-0.5b-lora-v1.6,conv_template=qwen_cot,model_name=llava-lora-qwen-0.5b-gfn \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-0.5b-lora-v1.6,conv_template=qwen_cot,model_name=llava-lora-qwen-0.5b-gfn \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
   --model qwen2_5_vl_gfn \
   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/Qwen2.5-gfn-3b-v1.7 \
   --tasks mathverse_testmini_vision_only \
   --batch_size 1 \
   --log_samples \
   --log_samples_suffix sqllava_ov \
   --output_path ./logs/gfn
  

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-0.5b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-0.5-gfn \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix llava-gfn-5 \
#    --output_path ./logs/cot
  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-qwen-gfn-0.5b-lora-v1.7,conv_template=qwen_cot,model_name=llava-qwen-lora-0.5-gfn \
  #  --tasks mmmu_val_thinking \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
#    CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_gfn \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-0.5b-lora-v1.6,conv_template=qwen_cot,model_name=llava-lora-qwen-0.5b-gfn \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/gfn

  
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-7b-250k,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-7b-250k,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmmu_val \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-7b-250k,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs

#      CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathverse_testmini_vision_only \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mathvision_test \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmmu_val \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#    --model llava_onevision \
#    --model_args pretrained=ZachSun/llava-qwen-gfn-sft-0.5b-250k-fx,conv_template=qwen_cot,model_name=llava-qwen-0.5b \
#    --tasks mmstar \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs
  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mathvision_test \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn
  #  CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  #  --model llava_gfn \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mmmu_val \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/gfn

  #    CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  #  --model llava_cot \
  #  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-gfn \
  #  --tasks mathverse_testmini_vision_only \
  #  --batch_size 1 \
  #  --log_samples \
  #  --log_samples_suffix sqllava_ov \
  #  --output_path ./logs/cot
# deepspeed -m lmms_eval \
#    --model llava_cot \
#    --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/gfn/llava-gfn-7b-lora-v1.6,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-sft-ov \
#    --tasks scienceqa \
#    --batch_size 1 \
#    --log_samples \
#    --log_samples_suffix sqllava_ov \
#    --output_path ./logs/cot

#CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#  --model llava_cot \
#  --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/deerllava-lora-qwen-7b-100k,conv_template=qwen_cot,model_name=sqllava_qwen-lora_7b-sft \
#  --tasks mmbench_en_dev \
#  --batch_size 1 \
#  --log_samples \
#  --log_samples_suffix sqllava_ov \
#  --output_path ./logs/gfn

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks realworldqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks longvideobench_val_v \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-si-lora-qwen-7b-interleave-5e6-0.3sq-ftViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
  # --tasks egoschema \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/
#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks egoschema \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen_7b \
#   --tasks llava_in_the_wild \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen_7b \
#   --tasks pope \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen_7b \
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/








  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
  # --tasks perceptiontest_val_mc \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/dpobest

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmbench_en_dev \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmstar \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmmu_val \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks llava_in_the_wild \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7/checkpoint-400,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/dpobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-800,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mmbench_en_dev\
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/spobest



  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-800,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
  # --tasks perceptiontest_val_mc\
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/spobest

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks mlvu \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
 

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/sqllava-ov-lora-qwen-7b-interleave-5e6-0.3sq-30frm-sLoRA-fixViT,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sft \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks llava_wilder_small \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mmmu_val \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks realworldqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mmstar \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/






  # CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-lora-qwen-0.5b-interleave-1e5-0.3sq-30frame,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_0.5b-sft-vit \
  # --tasks videomme \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/

#  CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/ours-0.5b-qwen-lora-sdo-lmd50-b0.1-lr1e5-alpa1-2epo-new,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_0.5b-sdo-vit \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/ours-0.5b-qwen-lora-dpo-g0-lr1e5-lmd50-1epo-newPrefv2,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_0.5b-sdo-vit \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks llava_interleave_bench_out_domain \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks longvideobench_val_v \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks egoschema \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks perceptiontest_val_mc \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 



# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mlvu \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-dpo-g0-lr5e6-lmd10-3epo-v4_sPat_fixVit_prj5e7,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks muirbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-dpo \
#   --tasks nextqa_mc_test \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/ 

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-apa1.2-lmd10-3epo-v4_sPat_fixVit_prj5e6/checkpoint-1200,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=ZachSun/sqllava-qwen-7b-interleave,conv_template=qwen_1_5,model_name=sqllava_qwen \
#   --tasks activitynetqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=ZachSun/sqllava-qwen-0.5b-interleave,conv_template=qwen_1_5,model_name=sqllava_qwen \
#   --tasks activitynetqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-lr5e6-b0.3-lmd10-2epo-QA,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks muirbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks muirbench \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


#   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks realworldqa \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks mmstar \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/

# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks videomme \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/
# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks llava_interleave_bench_out_domain \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/


# CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
#   --model llava_ovsq \
#   --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/qincan/sqllava-7b-qwen-lora-sdo-Q-b0.2-lmd10,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-sdo \
#   --tasks longvideobench_val_v \
#   --batch_size 1 \
#   --log_samples \
#   --log_samples_suffix sqllava_ov \
#   --output_path ./logs/




  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llavaCoT-si-lora-qwen-7b-3e6-fixViT-extraPropmt,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-si-sft \
  # --tasks mmvet\
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/cot7b
  # CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llavaCoT-si-lora-qwen-7b-3e6-fixViT-extraPropmt,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-si-sft \
  # --tasks mathvista_testmini_cot\
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/cot7b

  #   CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=/home/ztao/guohao/LLaVA-NeXT/checkpoints/cot/llavaCoT-si-lora-qwen-7b-3e6-fixViT-extraPropmt,conv_template=qwen_1_5,model_name=sqllava_qwen-lora_7b-si-sft \
  # --tasks mmbench_en_dev\
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/cot7b
  #   CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  # --model llava_ovsq \
  # --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-si,conv_template=qwen_1_5,model_name=llava_qwen_7b \
  # --tasks mathvista_testmini_cot \
  # --batch_size 1 \
  # --log_samples \
  # --log_samples_suffix sqllava_ov \
  # --output_path ./logs/
