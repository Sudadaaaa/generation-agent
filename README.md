# generation-agent

把自然语言的图片生成需求一步步变成一张图：先结构化出 **GenerationPlan**，经**人机评审会**定稿，
再渲染成自然语言最终提示词，最后交给本地生图模型出图（可选，环境配置决定）。

## 流程

```
用户需求（交互输入）
   ↓  大脑 = DeepSeek 云端（ReAct + 原生 function calling），自主编排：
make_plan → review（人机评审会 = 确认门）→ render → generate
   │        ↑                       │
   │      revise（问题+你的意见自动带回 make_plan 修订，直到你确认）
   ↓
GenerationPlan（结构化）→ LLM 渲染为最终提示词 → generation/ 本地出图 PNG
```

关键机制（详见 [docs/agent_design.md](docs/agent_design.md)）：

- **review = 确认门**：机器先按你的需求挑问题，你在同一交互里回车/p/q/补意见；
  只有 `confirmed` 定稿才允许 render/generate（门控由工具自报，可离线测试）。
- **谁干什么各归其层**：`domain/` 领域工序（提示词单一来源）· `llm/` 传输适配 ·
  `agentkit/` 通用 agent 机制（零图像知识）· `tools/` 本 agent 的工具实现 · `config.py` 环境配置。
- **qwen 只当 plan 工人**（本地 Qwen3，`AGENT_WORKER_PROVIDER=qwen`），不是大脑。
- **连续会话**：main 就是一个整场 while，user/工具结果/模型返回都进同一份 transcript——出图后说
  「在刚才基础上加 XX 再生成」会基于上一版已确认计划续版；重置对话 = 重跑 main（无步数/调用上限）。

## 安装

    pip install -r requirements.txt

所有模型从本地 HF 缓存加载（`local_files_only=True`），需预置 Qwen3、Z-Image、FLUX 缓存。

## 使用

    cp .env.example .env   # 填 DEEPSEEK_API_KEY，按需改 AGENT_* 键
    python main.py         # 无命令行参数：main 就是一个整场主循环（transcript 全程保留）；
                           # 「在刚才基础上加 XX 再生成」可直接续作；exit 退出；重跑 main = 新会话

干跑 / 只出提示词：`.env` **不配** `AGENT_IMAGE_MODEL`（注册的工具集里就没有 generate）。
想真出图：给 `AGENT_IMAGE_MODEL` 配 `zimage` / `flux` / 本地模型 id。

GPU：本地 LLM（qwen）固定逻辑 `cuda:0`、生图固定逻辑 `cuda:1`；映射到哪张物理卡由
`main.py` 顶部的 `CUDA_VISIBLE_DEVICES` 决定（换卡只改那一行）。

## 结构

```
main.py     会话 CLI：无参数，读配置 → 进主循环
config.py   .env → AgentConfig（大脑/工人/评审模型、生图可用性）
domain/     领域工序：schema / prompts / extraction / critique / rendering（无状态）
llm/        传输适配：ChatClient 协议 + DeepSeekChat(原生 function calling) + LocalQwenChat
agentkit/   通用 agent 机制：Tool/Registry(MCP 同构)、policy、runtime(agent_act 会话级单步)、io、usage
tools/      本 agent 工具：make_plan / review / render / generate（build_tools 注册）
generation/ 生图执行器（factory + backends: base / zimage / flux）
tests/      零依赖离线回归：python -m tests
docs/       agent_design.md（架构 + 接缝）
```
