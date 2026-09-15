# generation-agent

把自然语言的图片生成需求变成一张真实的图片：大脑（DeepSeek 云端）自己把需求写成
一段高质量的中文提示词，调用 `generate_image` 出图，再把保存路径汇报给你。

## 流程

```
用户需求（交互输入）
   ↓  大脑 = DeepSeek（Function Calling）
plan（PlanAgent 把需求拆成结构化 GenerationPlan，返回 JSON）
   ↓  大脑自己把计划渲染成一段自然语言提示词
generate_image（本地生图模型出图，PNG 落盘 outputs/）
   ↓
汇报：提示词 + 图片路径
```

整场对话一份 history（在 `MyAgent` 里），user / assistant / tool 消息按序追加。
「在刚才基础上加 XX 再生成」可直接续作；重置对话 = 重跑 main。
`PlanAgent` 也有自己的一份全量 history，所以它对「在刚才基础上加 XX」同样接得住。

> 评审（review）/ 渲染（render）两个子 Agent 是**后面**的事，本阶段不存在。
> 现在「把计划渲染成提示词」这一步暂时由大脑兼任——编排话术挂在
> `PlanTool.description` 上，等大脑从「生图助手」改成通用 agent 时再搬回系统提示词。

## 运行环境

本项目跑在 conda env **`llm`**：

    conda activate llm
    pip install -r requirements.txt

生图模型从**本地 HF 缓存**加载（`local_files_only=True`），需预置
`Tongyi-MAI/Z-Image-Turbo` 和/或 `black-forest-labs/FLUX.2-klein-9B`。

## 配置

`.env`（从 `.env.example` 拷，**不进 git**）：

| 键 | 说明 |
|---|---|
| `LLM_MODEL_ID` | 大脑模型 id |
| `LLM_API_KEY` | API 密钥 |
| `LLM_BASE_URL` | 服务地址；缺省 `https://api.deepseek.com` |
| `PLAN_MODEL_ID` | plan 子 agent 的模型 id；**留空 = 沿用 `LLM_MODEL_ID`** |
| `PLAN_API_KEY` | plan 子 agent 的密钥；**留空 = 沿用 `LLM_API_KEY`** |
| `PLAN_BASE_URL` | plan 子 agent 的服务地址；**留空 = 沿用 `LLM_BASE_URL`** |
| `PLAN_PROVIDER` | plan 子 agent 的服务商标识；**留空 = `deepseek`**。纯显示用（`str(agent)`），不参与任何分支判断 |
| `T2I_MODEL_ID` | 生图后端，取 [tools/t2i.py](tools/t2i.py) 里 `MODELS` 的键：`Z-Image-Turbo` / `FLUX.2-klein-9B`。**留空 = 无生图能力**，工具集里就没有 `generate_image` |
| `T2I_MODEL_DEVICE` | 生图卡；缺省 `cuda:0` |
| `T2I_OUTPUTS_DIR` | 图片输出目录；缺省 `outputs` |

键表的结构（类型 / 缺省 / 优先级）都在 [core/config.py](core/config.py)——那是全项目
**唯一**读环境的地方，业务代码里不该出现 `os.getenv`。

键名约定：**字段名大写 = 键名**。`LLM_*` 是所有 agent 共用的 LLM 凭证，
`T2I_*`（text-to-image）/ `I2I_*` 等按工具分。凭证**按服务商分、不按 agent 分**——
同一个 key 能调该服务商的多个模型。角色 → 模型的映射不进环境变量（env 是一维的，
那种二维结构该在代码表里），形状见 [.env.example](.env.example) 末尾的注释块。

`PLAN_*` 是这条规则的一个**例外**：它是「换模型」用的覆盖项，留空即沿用同名
`LLM_*`，所以不是一组新凭证、也不需要填就能跑。

步数、`guidance_scale`、分辨率、种子这些「模型怎么跑」的配方**不进 .env**，
在 [tools/t2i.py](tools/t2i.py) 的 `MODELS` 表里；plan 的最大修正次数
（`PlanAgent.max_repairs`）同理，在 [agent/planagent.py](agent/planagent.py) 里。

plan 对输出 JSON 的约束走 OpenAI 标准的 `response_format={"type": "json_object"}`
（见 [agent/planagent.py](agent/planagent.py) 的 `RESPONSE_FORMAT`）：它只保证
**语法合法的 JSON**，不保证合 Schema——合不合由 `parse_generation_plan` 判定，
不符就带着精确字段路径发一条修正指令重来。这个写法在 DeepSeek 与 vLLM 上**同一个
字段**，不用按 provider 翻译；换 vLLM 只改 `.env` 的 `PLAN_*`，代码一行不动。

## 用本地模型跑 plan（vLLM）

plan 可以换成**本地模型**跑。动机不只是「本地」——是**真正的 schema 级约束**：
DeepSeek 的 `response_format={"type":"json_object"}` 只保证语法合法的 JSON，
合不合 `GenerationPlan` 仍要靠 `parse_generation_plan` 的修正循环兜；vLLM 收到
**同一个字段**会转成约束解码（xgrammar 逐 token 掩码），结构上不可能输出非法 JSON。
所以**代码一行都不用改**，只改 `.env`。

服务器与客户端是**两个独立的 conda 环境**，只通过 HTTP 相连：

```bash
conda create -n vllm python=3.12 -y && conda activate vllm
pip install "vllm==0.19.1"                # ← 版本必须 pin，理由见下
bash scripts/serve_plan_model.sh          # 默认卡 0 端口 8000；GPU=/PORT= 可覆盖
curl -s http://localhost:8000/v1/models   # 应列出 qwen3-8b
```

**为什么 pin 0.19.1 而不是装最新版**：vLLM 从 0.20.2 起 pin 的 torch 是 2.11+，而
torch ≥2.11 的轮子带的是 **CUDA 13** 运行库（`nvidia-cudnn-cu13`、`nccl-cu13`…），
要驱动 ≥580。本机驱动是 535（CUDA 12.2），装最新版会在 CUDA 初始化时挂掉。
分界线在 torch 上：

| torch | 带的 nvidia 运行库 | 驱动 535（CUDA 12.2）上 |
|---|---|---|
| ≤ 2.10 | `*-cu12` | ✅ 可用 |
| ≥ 2.11 | `*-cu13` | ❌ 需驱动 ≥580 |

对应到 vLLM：`0.29 / 0.28 / 0.27` → torch 2.13，`0.26 … 0.20.2` → torch 2.11，
**`0.19.1` → torch 2.10**——它是最后一个落到 cu12 的版本。换机器先看 `nvidia-smi`
顶部的 CUDA 版本，再据此挑 vLLM，别照抄这一行。

**权重走默认 HF 缓存，不需要配任何路径。** Qwen3-8B 在 `/DATASSD1/flh_lm/huggingface`，
而 `~/.cache/huggingface` 正是指向它的**软链接**，所以 `vllm serve Qwen/Qwen3-8B`
直接就查得到——[scripts/serve_plan_model.sh](scripts/serve_plan_model.sh) 里**没有**
`HF_HOME`，也不要加。

> **排查提示**：那个软链接用 `find ~/.cache/huggingface` 或 `du -xsh` 看都**像空目录**
> ——`find` 默认不跟随软链接（起点是 symlink 时只打印一行就停，不下钻），
> `du -x` 不跨文件系统。要确认得用 `ls -ld`。同一个误会会让人以为缓存没配。

脚本里只开了 `HF_HUB_OFFLINE=1`：本机 `huggingface.co` 确实连不通（curl 12 秒超时、
DNS 正常），缓存缺东西时它会立刻报错，而不是挂着等一轮网络超时——**后者看起来像
「加载很慢」，很费排查时间**。

脚本里那三个参数是按**单卡 24G + 单 agent 顺序调用**调出来的，换卡要重算，理由都写在
脚本注释里：

| 参数 | 为什么是这个值 |
|---|---|
| `--max-model-len 32768` | config 写 40960，但放不下——权重 15.27 GiB 之后只剩 5.67 GiB 给 KV，40960 要 5.62 GiB 加激活就爆；vLLM 算出的上限是 34032，取 32768 |
| `--gpu-memory-utilization 0.90` | 留给同卡其他进程的余量 |
| `--max-num-seqs 8` | **不设会起不来**：vLLM 默认 256，启动时拿 256 个假请求预热采样器，24G 卡直接 OOM。单 agent 顺序调用并发恒为 1 |

然后 `.env` 里填（[.env.example](.env.example) 有现成的注释版）：

    PLAN_MODEL_ID="qwen3-8b"
    PLAN_BASE_URL="http://localhost:8000/v1"
    PLAN_API_KEY="EMPTY"      # vLLM 不校验 key，但字段不能空（AgentLLM 会拦空值）
    PLAN_PROVIDER="vllm"      # 纯显示用

**注释掉这四行即退回 DeepSeek**（回退链在 [core/config.py](core/config.py) 的
`plan_llm_kwargs`），agent 仍跑在 `llm` 环境里，不受影响。

**思考模式默认关**，两个理由，第二个是硬的：

1. [sft/out/](sft/out/) 的语料里 **0 条思考痕迹**——assistant 内容是纯 JSON，没有
   一段 ` thinking`。训练目标没有推理段，推理时要它产出思考段，等于让 LoRA 管不到的
   那部分自由发挥。
2. vLLM 在「思考开 + 非流式 + 结构化输出」下有已知陷阱：答案整个落进
   `reasoning_content`、`content` 变成 `null`——那正是
   [agent/planagent.py](agent/planagent.py) 的 `_complete()` 当致命错误处理的情形。

要改回开：换掉 `--default-chat-template-kwargs`，并加
`--structured-outputs-config.enable_in_reasoning=True`。

微调完成后加载 LoRA 只需多一个环境变量（`scripts/serve_plan_model.sh` 已预留）：

    LORA_PATH=/path/to/lora bash scripts/serve_plan_model.sh
    # 再把 .env 的 PLAN_MODEL_ID 改成 plan-lora

## 使用

    python main.py         # 无参数；exit / quit / 退出 结束；重跑 = 新会话

启动时会打印生图开关和工具清单——「为什么不出图」一眼可见。

各层还能脱离 `main.py` 单独自测（入口在各文件末尾的 `if __name__ == "__main__"`）：

    python core/llm.py        # 只验传输层 + 原生 function calling，不经过 MyAgent
    python agent/myagent.py   # 只验 MyAgent 的对话 + 流式重组
    python agent/planagent.py # 只验 plan 的产出与格式修正循环，不经过任何工具

都能用 `python -m core.llm` / `python -m agent.myagent` / `python -m agent.planagent`
（在仓库根跑）替代。

想真出图：`.env` 里给 `T2I_MODEL_ID` 配 `Z-Image-Turbo` 或 `FLUX.2-klein-9B`。

## 结构

```
main.py     入口：装配工具集 → 对话主循环
core/       与图像无关的通用机制
  message.py   Message（OpenAI 消息的本地类型 + 不变量校验）
  agent.py     BaseAgent（history / add_message / get_messages）
  llm.py       AgentLLM（OpenAI 兼容传输层，只收参数）
  config.py    AgentConfig（pydantic-settings，唯一读环境的地方）
  io.py        IO / ConsoleIO（工具的交互不直接碰 input/print）
  schema.py    GenerationPlan / Element（第二阶段的结构化计划契约）
  tool.py      Tool 基类（dataclass + run/parse_args/to_openai_schema）
  registry.py  ToolRegistry（注册 / 执行 / 导出 OpenAI schema）
  exceptions.py
agent/      子 agent
  myagent.py   MyAgent：run() 工具循环 + 流式重组（大脑）
  planagent.py PlanAgent：需求 → GenerationPlan，带格式修正循环
tools/      工具
  math.py      add（示例工具）
  t2i.py       generate_image + ImageGenerator（后端实现都在这里）
  plan.py      plan：包一个 PlanAgent，返回计划 JSON
  review.py / render.py   旧版，尚未重写（git show HEAD:tools/xxx.py 可看）
scripts/
  serve_plan_model.sh  起本地 plan 模型的 vLLM 服务（跑在独立的 vllm 环境里）
outputs/    出的图（img_<时间戳>_<tag>.png）
sft/         微调相关材料
```

分层（**import 只能朝下**）：

```
core  ←  tools  ←  agent          main.py / tools.build_tools（同时认识两边）
```

`core/` 是底层：不 import `tools/` 也不 import `agent/`——所以 `Tool` 与
`ToolRegistry`（纯机制，与图像/math/plan 无关）住在 `core/`，子 agent 才能直接持有
一个 `ToolRegistry` 而不成环。**唯一的例外**是 `tools/__init__.py` 的 `build_tools`：
它 import `agent.planagent` 来装配 plan 工具，是装配边而非协作边（`PlanTool` 自己
从不 import `agent`，只调注入进来的对象）。谁要往 `agent/` 里加东西，记住这条：
**别让 `agent/` 的模块 import `tools/` 下的具体模块**，要用机制就去 `core/`。
