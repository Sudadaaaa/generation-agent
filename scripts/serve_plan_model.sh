#!/usr/bin/env bash
# 起本地 plan 模型的 vLLM 服务。
#
# 服务器与客户端是分开的两个 conda 环境：本脚本跑在 `vllm` 环境里，agent 仍跑在
# `llm` 环境，两边只通过 .env 的 PLAN_BASE_URL 相连。所以 vLLM 升降依赖不会碰到
# llm 环境里的 transformers / torch（SFT 与 diffusers 生图在用的那套）。
#
#   用法：
#     bash scripts/serve_plan_model.sh                    # 默认卡 0、端口 8000
#     GPU=1 PORT=8001 bash scripts/serve_plan_model.sh    # 换卡 / 换端口
#     LORA_PATH=/path/to/lora bash scripts/serve_plan_model.sh   # 微调完之后
#
# 起好之后配 .env：
#     PLAN_MODEL_ID="qwen3-8b"
#     PLAN_BASE_URL="http://localhost:8000/v1"
#     PLAN_API_KEY="EMPTY"     # vLLM 不校验，但字段不能空（AgentLLM 会拦空值）
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
PORT="${PORT:-8000}"
GPU="${GPU:-4}"

ARGS=(
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --gpu-memory-utilization 0.90
  --max-num-seqs 8
  --max-model-len 32768
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

# LoRA：训练完之后 export LORA_PATH=...，并把 .env 的 PLAN_MODEL_ID 改成 plan-lora。
# --enable-lora 时底座与每个 adapter 各是一个可选 model 名，所以不用改 SERVED_NAME。
if [[ -n "${LORA_PATH:-}" ]]; then
  ARGS+=(--enable-lora --lora-modules "plan-lora=${LORA_PATH}")
fi

export HF_HUB_OFFLINE=1

export CUDA_VISIBLE_DEVICES="$GPU"
echo "启动：$MODEL  →  http://localhost:${PORT}/v1  （model=${SERVED_NAME}，GPU=${GPU}）"
exec vllm serve "$MODEL" "${ARGS[@]}"
