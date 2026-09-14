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
| `T2I_MODEL_ID` | 生图后端，取 [tools/t2i.py](tools/t2i.py) 里 `MODELS` 的键：`Z-Image-Turbo` / `FLUX.2-klein-9B`。**留空 = 无生图能力**，工具集里就没有 `generate_image` |
| `T2I_MODEL_DEVICE` | 生图卡；缺省 `cuda:0` |
| `T2I_OUTPUTS_DIR` | 图片输出目录；缺省 `outputs` |

键表的结构（类型 / 缺省 / 优先级）都在 [core/config.py](core/config.py)——那是全项目
**唯一**读环境的地方，业务代码里不该出现 `os.getenv`。

键名约定：**字段名大写 = 键名**。`LLM_*` 是所有 agent 共用的 LLM 凭证，
`T2I_*`（text-to-image）/ `I2I_*` 等按工具分。凭证**按服务商分、不按 agent 分**——
同一个 key 能调该服务商的多个模型。角色 → 模型的映射不进环境变量（env 是一维的，
那种二维结构该在代码表里），形状见 [.env.example](.env.example) 末尾的注释块。

`PLAN_*` 是这条规则的一个**例外**：它是「换模型」用的覆盖项，三项留空即沿用同名
`LLM_*`，所以不是一组新凭证、也不需要填就能跑。

步数、`guidance_scale`、分辨率、种子这些「模型怎么跑」的配方**不进 .env**，
在 [tools/t2i.py](tools/t2i.py) 的 `MODELS` 表里；plan 的最大修正次数
（`PlanAgent.max_repairs`）同理，在 [agent/planagent.py](agent/planagent.py) 里。

plan 对输出 JSON 的约束走 OpenAI 标准的 `response_format={"type": "json_object"}`
（见 [agent/planagent.py](agent/planagent.py) 的 `RESPONSE_FORMAT`）：它只保证
**语法合法的 JSON**，不保证合 Schema——合不合由 `parse_generation_plan` 判定，
不符就带着精确字段路径发一条修正指令重来。这个写法在 DeepSeek 与 vLLM 上**同一个
字段**，不用按 provider 翻译；换 vLLM 只改 `.env` 的 `PLAN_*`，代码一行不动。

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
