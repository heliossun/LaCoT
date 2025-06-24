conda create -n qwen python=3.10 -y
conda activate qwen
pip install git+https://github.com/huggingface/transformers accelerate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install qwen-vl-utils[decord]
pip install -U flash-attn --no-build-isolation
pip install deepspeed
pip install peft
pip install ujson
pip install liger_kernel
pip install datasets
pip install torchvision
pip install wandb