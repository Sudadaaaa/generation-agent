# generation-agent

把自然语言的画面需求变成一张本地图片：主 agent 自己判断用哪些工具，调 `plan` 拆出结构化
计划、`critic` 评审，再把计划写成一段自然提示词交给 `generate_image` 出图。

## 流程

```
用户需求
   ↓  主 agent（模型由 LLM_* 决定，工具清单由 tools.build_tools 按配置装配）
plan     PlanAgent：需求 → GenerationPlan JSON（不合 schema 就带字段路径重发，至多 3 次）
   ↓     plan 工具的描述里提示「计划可能有问题、可以带修改意见再调一次」
critic   CriticAgent：需求 + 计划 → {评分, 意见, 问题数量}（可选，调不调由主 agent 判断）
   ↓     有意见就把「按意见改」再交给 plan
generate_image   ImageGenerator：一段自然提示词 → outputs/img_<时间戳>_raw.png
   ↓
主 agent 回报保存路径
```

- **计划 → 提示词没有独立工具，由主 agent 兼任**：`generate_image` 的参数是
  `prompt: 一段自然、连贯的提示词`，主系统提示词也写着「它要自然语言的描述，就不要把结构化的
  中间结果直接塞进去」。`GenerationPlan.to_prompt_text()` 是确定性兜底，但**当前没有调用方**。
- 整场对话一份 history（`MainAgent`）；`PlanAgent` 也保留全量 history（「在刚才基础上加 XX」
  靠它），`CriticAgent` **反过来每次重装**，只看「需求 + 这一版计划」——免得被计划的自辩与
  上一轮评语说服着放行。
- `MAIN_SYSTEM_PROMPT` 刻意**不点名任何工具、不规定语言风格**，编排提示挂在各工具自己的
  description 上，增删工具不用动它。

## 运行环境

conda env **`llm`**（[requirements.txt](requirements.txt) 同时含训练依赖）：
`conda activate llm && pip install -r requirements.txt`

vLLM 服务在**另一个独立环境**里，见下。生图模型从**本地 HF 缓存**加载
（`local_files_only=True`），已预置 `Tongyi-MAI/Z-Image-Turbo` 与
`black-forest-labs/FLUX.2-klein-9B`。

## 配置

`.env`（从 [.env.example](.env.example) 拷，**不进 git**；`.gitignore` 只忽略 `.env`、
`__pycache__`、`outputs/`）：

| 键 | 说明 |
|---|---|
| `LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` | 主 agent 的模型 / 密钥 / 服务地址（缺省 `deepseek-v4-flash` / `https://api.deepseek.com`） |
| `PLAN_MODEL_ID` / `PLAN_API_KEY` / `PLAN_BASE_URL` | plan 子 agent；**各自留空 = 沿用同名 `LLM_*`** |
| `CRITIC_MODEL_ID` / `CRITIC_API_KEY` / `CRITIC_BASE_URL` | critic 子 agent；同上。⚠️ [.env.example](.env.example) 里**还没有**这一段，要用得自己加 |
| `PLAN_PROVIDER` / `CRITIC_PROVIDER` | 服务商标识，**纯显示用**（`str(agent)`），不参与任何分支判断 |
| `T2I_MODEL_ID` | 生图后端，取 [tools/t2i.py](tools/t2i.py) `MODELS` 的键。**留空 = 无生图能力**——`build_tools` 压根不注册 `generate_image`，主模型在 tools schema 里看不见它 |
| `T2I_MODEL_DEVICE` / `T2I_OUTPUTS_DIR` | 生图卡（缺省 `cuda:0`）/ 输出目录（缺省 `outputs`） |

类型、缺省、优先级都在 [core/config.py](core/config.py)——全项目**唯一**读环境的地方，业务代码
里不该出现 `os.getenv`；键名约定是**字段名大写 = 键名**。**不进 .env 的**：生图的 `steps` /
`guidance` 在 `MODELS` 表里（Z-Image-Turbo 9 步 / guidance 0；FLUX.2-klein-9B 12 步 / 不传），
分辨率 1024 与种子 42 写死在 `ImageGenerator` 里，plan / critic 的 `max_repairs=2` 在各自 agent 里。

## 用本地模型跑 plan（vLLM）

换本地模型不只是「本地」——是**更强的输出约束**：DeepSeek 的
`response_format={"type":"json_object"}` 只保证语法合法，合不合 Schema 仍要靠修正循环兜；
vLLM 收到**同一个字段**会转成约束解码，结构上不可能输出非法 JSON。
**代码一行都不改，只改 `.env`。**

```bash
conda create -n vllm python=3.12 -y && conda activate vllm
pip install "vllm==0.19.1"                # ← 必须 pin，理由见下
bash scripts/serve_plan_model.sh          # 默认卡 4、端口 8000；GPU= / PORT= 可覆盖
curl -s http://localhost:8000/v1/models   # 应列出 qwen3-8b（带 LORA_PATH 时还有 plan-lora）
```

**pin 0.19.1 的理由**：vLLM ≥0.20.2 的 torch 是 2.11+，而 torch ≥2.11 的轮子带 **CUDA 13**
运行库（`nvidia-cudnn-cu13`、`nccl-cu13`…），要驱动 ≥580；本机驱动是 535（CUDA 12.2），装最新版
会在 CUDA 初始化时挂掉。分界线在 torch 上，`0.19.1` 是最后一个落到 torch 2.10（cu12）的版本
——换机器先看 `nvidia-smi` 顶部的 CUDA 版本再挑，别照抄这一行。

**权重走默认 HF 缓存，不用配任何路径**：`~/.cache/huggingface` 是指向
`/DATASSD1/flh_lm/huggingface` 的软链接，脚本里**没有** `HF_HOME`，也不要加。（它用 `find` /
`du -xsh` 看**像空目录**——前者默认不跟随软链接，后者不跨文件系统；要确认得用 `ls -ld`。）
脚本只开了 `HF_HUB_OFFLINE=1`：本机连不通 `huggingface.co`，缓存缺东西时它会立刻报错，而不是
挂着等一轮网络超时——**那看起来像「加载很慢」**。

三个参数是按**单卡 24G + 单 agent 顺序调用**定的：

| 参数 | 为什么 |
|---|---|
| `--max-model-len 32768` | Qwen3-8B 的 config 写 40960，但 bf16 权重约 15.3 GiB，0.9 × 24G 剩下的还要给 KV cache，40960 放不下 |
| `--gpu-memory-utilization 0.90` | 留给同卡其他进程的余量 |
| `--max-num-seqs 8` | vLLM 默认 256，启动时拿 256 个假请求预热采样器，24G 卡直接 OOM——**不设会起不来**（单 agent 顺序调用并发恒为 1） |

脚本另传了 `--reasoning-parser qwen3` 与 `--default-chat-template-kwargs '{"enable_thinking":
false}'`，**思考模式默认关**：① 语料 3088 条 assistant 正文**全是** GenerationPlan JSON（实测
3088/3088 可 `json.loads`、无一段 think），训练目标里没有思考段；② vLLM 在「思考开 + 非流式 +
结构化输出」下答案会整个落进 `reasoning_content`、`content` 变 `null`，那正是 `_complete()` 当
致命错误处理的情形。要改回开：换掉 chat-template-kwargs 并加
`--structured-outputs-config.enable_in_reasoning=True`。

`.env` 里填（[.env.example](.env.example) 有 PLAN_* 的注释版）：

    PLAN_MODEL_ID="qwen3-8b"
    PLAN_BASE_URL="http://localhost:8000/v1"
    PLAN_API_KEY="EMPTY"      # vLLM 不校验 key，但字段不能空（AgentLLM 会拦空值）

**注释掉这三行即退回 DeepSeek**（回退链在 [core/config.py](core/config.py) 的
`plan_llm_kwargs`）。微调后加载 LoRA 只需多一个环境变量：

    LORA_PATH=outputs/sft_lora bash scripts/serve_plan_model.sh   # 再把 PLAN_MODEL_ID 改成 plan-lora

## 训练与评估

- **数据**：[mydatasets/plan_sft.jsonl](mydatasets/plan_sft.jsonl)，3088 行 = 1544 en + 1544 zh，
  每行只有 `{id, messages}`（user + assistant），system 在分词时注入——它**不是副本**，
  [mydatasets/plan_sft.py](mydatasets/plan_sft.py) import 的是推理侧同一个函数（见下）。
- **切分**：`VAL_FRAC=0.02` 尾切，实测 train 3026 行 / val 62 行（语言各半）、**id 交集为 0**，
  不落盘副本；评估侧现场调 `plan_sft_rows("val")` 取，改比例时两边同时生效。
- **训练**：[train/train_sft.py](train/train_sft.py)（accelerate + DeepSpeed **ZeRO-3 真分片**，
  配置在 [configs/accelerate_zero3.yaml](configs/accelerate_zero3.yaml)），
  `bash train/run_train_sft.sh` 起；`--max_seq_length` 别低于 4096（短于 system prompt 会让样本
  截在 assistant 回复之前、labels 全 `-100` → `loss=nan`）。本机 4 卡实测约 7.2 s/it、757
  batch/epoch ≈ 1h28m。存 adapter 前会核对 LoRA 参数个数，对不上直接作废这次保存——ZeRO-3 下
  不 gather 会**静默写出空张量**。
- **评估**：[eval/run_eval.py](eval/run_eval.py) 跑「plan ↔ critic 改到 critic 满意」，上限
  `MAX_ROUNDS=3`；每条记 `score / issues_total / revisions / final_len`，汇总另给 `plan_ok` 与
  `scored`（均值的分母是 scored 不是 n——失败不记 0 分，否则「常常生不出来」的模型看起来只是
  「分低一点」）。结果写 `outputs/eval/<时间戳>.json`，途中逐条落 `.partial.jsonl` 防中断。
  一次只测 `.env` 当前那个模型；critic 由 `CRITIC_*` 决定、**不跟着 `PLAN_*` 动**，所以测本地臂
  时它就是那个更强的云端裁判。`--no-constraint` 可关掉 plan 侧的 `response_format`，用来量
  约束解码值多少。

## 单一真源（改提示词前必读）

plan 的 system 提示词有两个消费方：推理侧 [agent/plan_agent.py](agent/plan_agent.py) 与训练侧
[mydatasets/plan_sft.py](mydatasets/plan_sft.py)，两边必须逐字相同。从前各存一份、靠注释提醒
同步，**已经失效过一次**：`GenerationPlan` 的类 docstring 改了措辞，而 pydantic 会把它嵌成
schema 顶层的 description，训练语料（sha1 `510f9d91`）与推理提示词（`3b8c905a`）就此不一致，
且没有任何机制会报错。

现在两边都 import [core/prompt.py](core/prompt.py)，**副本在物理上不存在**；拆解规则全部写进
[core/schema.py](core/schema.py) 的字段说明，散文里只留身份与输出契约——critic 评审拼的是
**同一份** schema，规则留在散文它就得抄一份，两份副本必然漂。

## 使用

    python main.py         # 无参数；exit / quit / 退出 结束；重跑 = 新会话

启动时会打印生图开关和工具清单——「为什么不出图」一眼可见。各层还能脱离 `main.py` 单独自测
（入口在各文件末尾的 `if __name__ == "__main__"`）：

    python core/llm.py            # 只验传输层 + 原生 function calling，不经过 MainAgent
    python agent/main_agent.py    # 只验 MainAgent 的对话 + 流式重组（不传工具）
    python agent/plan_agent.py    # 连说两句可验 plan 的全量 history
    python agent/critic_agent.py  # 先让 plan 拆一版，再让 critic 评它

都能用 `python -m <模块路径>`（在仓库根跑）替代。想真出图：`.env` 里给 `T2I_MODEL_ID` 配
`Z-Image-Turbo` 或 `FLUX.2-klein-9B`。

## 结构

```
main.py     入口：装配工具集 → 对话主循环
core/       与图像无关的通用机制
  message.py   Message（OpenAI 消息的本地类型 + 不变量校验）
  agent.py     BaseAgent（history / add_message / get_messages / clear_history）
  llm.py       AgentLLM（OpenAI 兼容传输层，只收参数）
  config.py    AgentConfig（pydantic-settings，唯一读环境的地方）
  prompt.py    全项目提示词的唯一真源
  schema.py    GenerationPlan / Element（结构化计划契约，规则写在字段说明里）
  io.py        IO / ConsoleIO（工具的交互不直接碰 input/print）
  tool.py      Tool 基类（dataclass + parse_args / to_openai_schema）
  registry.py  ToolRegistry（注册 / 执行 / 导出 OpenAI schema）
  exceptions.py
agent/      子 agent
  main_agent.py   MainAgent：run() 工具循环 + 流式重组（主 agent）
  plan_agent.py   PlanAgent：需求 → GenerationPlan，带格式修正循环
  critic_agent.py CriticAgent：需求 + 计划 → 评分 / 意见 / 问题数量
tools/      工具
  __init__.py  build_tools(config)：按配置装配工具集（唯一的装配点）
  plan.py / critic.py  各包一个子 agent，只做参数与回执（工具不持有 LLM）
  t2i.py       generate_image + ImageGenerator（后端实现、MODELS 表都在这里）
  math.py      add（示例工具，演示 Tool 怎么写）
  review.py / render.py  旧版（import 已不存在的 agentkit.* / domain.*，现在 import 即崩），
                         也没有任何模块 import 它们——留着只为 `git show` 取回
train/      LoRA SFT（train_sft.py + run_train_sft.sh）
mydatasets/ plan_sft.jsonl 与切分——训练与评估共用的唯一入口
configs/    accelerate + DeepSpeed ZeRO-3 配置
eval/       run_eval.py：plan ↔ critic 循环的四项指标
scripts/    serve_plan_model.sh（跑在独立的 vllm 环境里）
outputs/    出的图与评估结果（不进 git）
```

分层（**import 只能朝下**）：

```
core  ←  tools  ←  agent          main.py / tools.build_tools（同时认识两边）
```

`Tool` 与 `ToolRegistry` 住在 `core/` 而不是 `tools/`，是因为 Python 里
`from tools.xxx import Y` 会先完整执行 `tools/__init__.py`，而后者要 import `agent` 来装配
——子 agent 一复用工具机制就成了环。**唯一的例外**是 `build_tools`：它 import
`agent.plan_agent`，是装配边而非协作边（`PlanTool` 自己从不 import `agent`，只调注入进来的
对象）。所以往 `agent/` 加东西时记住：**别 import `tools/` 下的具体模块**，要用机制就去 `core/`。

## 已知缺口

- [.env.example](.env.example) 里没有 `CRITIC_*` 段（config.py 支持，模板没跟上）。
- 全仓**没有测试**；`eval/run_eval.py --smoke`（= `--limit 2`）是唯一的低成本验证入口。
