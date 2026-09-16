#!/bin/bash
# 一键启动 Qwen3-8B LoRA 训练（固定 DeepSpeed ZeRO-3 真分片，长序列用）
#   用法: bash train/run_train_sft.sh [额外参数...]
#   例:   bash train/run_train_sft.sh --epochs 3 --max_seq_length 8192
#         # --num_samples 不传 = 用全部数据（train_sft.py 默认 None）；传 N 则取前 N 条
#         # --dataset 默认 plan_sft（本项目只有它，一般不用传）
set -e
# 切到【仓库根】：下面的 configs/ 与 train/ 都是相对仓库根写的。
# 脚本自己住在 train/ 下，所以要加 /.. ——少了它就会在 train/ 里找
# train/train_sft.py 和 train/configs/，两个都不存在，脚本直接失败。
cd "$(dirname "$0")/.."

# ===== 按需修改 =====
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 4-7 暂被 wzy 占用，先切 0-3
# ⚠️ 两个名字都设：torch 2.9.1 的 CUDA caching allocator 编译代码只读旧名
# PYTORCH_CUDA_ALLOC_CONF（strings 验证：libc10_cuda.so 2× 旧名 / 0× 新名）。
# 只设新名 PYTORCH_ALLOC_CONF → expandable_segments 静默失效 → 长序列内存碎片 OOM。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 用哪个 accelerate。默认是本机装好 deepspeed/peft 的那个 conda 环境；
# 换机器时用 ACCELERATE=/path/to/accelerate 覆盖，或 conda activate 后改成裸 accelerate。
ACCELERATE="${ACCELERATE:-/DATASSD2/PycharmProjects_flh/anaconda3/envs/llm/bin/accelerate}"

"$ACCELERATE" launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes=4 \
  train/train_sft.py "$@"
