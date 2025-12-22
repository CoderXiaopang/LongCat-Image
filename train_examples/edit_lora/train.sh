#!/bin/bash

export TOKENIZERS_PARALLELISM=False
# export NCCL_DEBUG=INFO  # 注释掉，减少不必要的日志刷屏，除非报错再打开
export NCCL_TIMEOUT=12000

script_dir=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd -P)
echo "script_dir" ${script_dir}

# --- 修改说明 ---
# 1. 删除了 project_root 和 deepspeed_config_file 的定义
#    (因为我们不再使用官方那个为集群定制的 DeepSpeed 配置)
# 2. 删除了 --num_processes 8 
#    (让 accelerate 自动检测你的显卡数量)
# 3. 删除了 --config_file 
#    (使用 accelerate 的默认配置，通常自动适配单卡或多卡 DDP)
# --------------

accelerate launch --mixed_precision bf16 \
    ${script_dir}/train_edit_lora.py \
    --config ${script_dir}/train_config.yaml
