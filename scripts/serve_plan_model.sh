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
GPU="${GPU:-0}"

ARGS=(
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --gpu-memory-utilization 0.90
  # 必须显式设。vLLM 默认 256，启动时会拿 256 个假请求预热采样器，在 24G 卡上
  # （权重 15.27 GiB + KV 5.28 GiB 之后只剩约 1 GiB）直接 OOM，报
  # "CUDA out of memory occurred when warming up sampler with 256 dummy requests"。
  # 而我们的负载是【单个 agent 顺序调用】，并发恒为 1，256 这个默认值毫无意义。
  # 8 是留给「以后多个子 agent 并行调 plan」的余量，不影响 KV 池大小。
  --max-num-seqs 8
  # 必须显式设，不能省。Qwen3-8B 的 config 写 max_position_embeddings=40960，但单卡
  # 24G 放不下：实测权重 15.27 GiB，扣掉激活与 CUDA graph 的 1.66 GiB，只剩 4.67 GiB
  # 给 KV，而 40960 要 5.62 GiB——不设的话 vLLM 直接拒绝启动。vLLM 自己算出的上限是
  # 34032，这里取 32768（2 的幂，留 3.9% 余量），KV 够不够由 gpu-memory-utilization
  # 决定，调大 max_model_len 不再多占内存。
  # 长度够不够，看实测（sft/out/ 全部 3088 条，用 Qwen3 自己的 tokenizer 数的）：
  #   prompt     中位 3174 / p99 3941 / 最大 4322 token（system 提示词本身占 2969）
  #   completion 中位  472 / p99 2104 / 最大 4294 token
  #   单轮合计   中位 3658 / 最大 7807 token  → 32768 够四五轮「在刚才基础上加 XX」
  # 【别被字符数骗】：最长那份计划 19765 字符只折 4294 token（0.217 token/字符），
  # 因为它主要是英文（"SR-71 Blackbird aircraft"…）。中文才接近 1 token/字。
  --max-model-len 32768
  # 必须带。vLLM 靠它把 Qwen3 的推理段与正文分开；不带的话，即使关了思考，
  # 正文也可能被判成 reasoning，而非流式请求下 content 会是 null。
  --reasoning-parser qwen3
  # 思考默认【关】。两个理由，第二个是硬的：
  #   1) sft/out/ 的语料里 0 条思考痕迹（assistant 内容是纯 JSON，无  thinking），
  #      训练目标没有推理段，推理时要它产出思考段等于让 LoRA 管不到的部分自由发挥；
  #   2) vLLM 在「思考开 + 非流式 + 结构化输出」下有已知陷阱：答案整个落进
  #      reasoning_content、content 变成 null——那正是 agent/planagent.py 的
  #      _complete() 当致命错误处理的情形。
  # 要改回开：把这个 flag 换掉，并加 --structured-outputs-config.enable_in_reasoning=True。
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

# LoRA：训练完之后 export LORA_PATH=...，并把 .env 的 PLAN_MODEL_ID 改成 plan-lora。
# --enable-lora 时底座与每个 adapter 各是一个可选 model 名，所以不用改 SERVED_NAME。
if [[ -n "${LORA_PATH:-}" ]]; then
  ARGS+=(--enable-lora --lora-modules "plan-lora=${LORA_PATH}")
fi

# 权重不用指定路径：~/.cache/huggingface 是个【软链接】→ /DATASSD1/flh_lm/huggingface，
# Qwen3-8B 就在里面。默认缓存查得到，所以这里【不需要】设 HF_HOME。
# （注意：那个软链接用 `find`/`du -x` 看会像空目录——find 默认不跟随软链接、
#   du -x 不跨文件系统。要确认得用 `ls -ld`。别被这两个工具骗了。）
#
# 只开离线模式：本机 huggingface.co 确实连不通（curl 12 秒超时返回 000，DNS 正常）。
# 缓存里万一缺东西时，offline 会立刻报「找不到」，而不是挂着等一轮网络超时——
# 后者看起来像「加载很慢」，很费排查时间。
export HF_HUB_OFFLINE=1

export CUDA_VISIBLE_DEVICES="$GPU"
echo "启动：$MODEL  →  http://localhost:${PORT}/v1  （model=${SERVED_NAME}，GPU=${GPU}）"
exec vllm serve "$MODEL" "${ARGS[@]}"
