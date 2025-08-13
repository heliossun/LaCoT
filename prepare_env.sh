conda create -n qwen python=3.10 -y
conda activate qwen

### if ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]
#pip install -U flash-attn --no-build-isolation
##
pip install deepspeed
pip install peft
pip install ujson
pip install liger_kernel
pip install datasets
pip install torchvision
pip install wandb

# transformers==4.51.3 for training