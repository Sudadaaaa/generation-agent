# generation-agent 重构后架构与接缝

> 决策与施工详见会话 Plan（`agentkit/policy` 是对大脑说的「生图流程」，本文件是对人说的架构落盘）。
> 一句话：**先机械分层重构，再在分层上把基础版 agent（ReAct + 原生 function calling）做出来**。

## 1. TL;DR

- **大脑** = DeepSeek 云端（默认 `deepseek-chat`），只经 **OpenAI 兼容 `tools`/`tool_calls`**（原生 function calling）驱动。
  ReAct = 想（读 policy+历史+工具 schema，给 thought 文本）→ 调（`complete_tools` 返回 tool_calls）→ 看（工具结果回灌成 tool 消息）。
- **本地 qwen** 只当 **plan 工人**（`make_plan` 内部的规划/渲染调用方），不是大脑。config 一行可切 `AGENT_WORKER_PROVIDER=qwen`。
- **MVP 四个工具**：`make_plan / review / render / generate`。
- **确认门 = review 人机评审会**：机器先按需求+8 条规则挑问题，你在同一交互里回车/p/q/补意见；只有 `confirmed` 定稿才允许 render/generate。
  问答提示语与意见累计从重构前的 `agent_plan_review` **逐字冻结迁入**。
- **语义/环境分层**：要不要出图由大脑读需求决定；**能不能出图**由环境配置决定（未配 `AGENT_IMAGE_MODEL` → 注册集里根本没有 `generate`）。
- 自研主循环，不引 LangChain；通过规范接缝保留将来外接的能力（§6）。

## 2. 分层结构（谁管什么）

```
generation-agent/
├── main.py            # 会话 CLI：main 直接是一个 while 主循环（transcript 整场持续）；exit 结束
├── config.py          # .env → AgentConfig（provider/生图可用性），环境能力分层的唯一边界
├── domain/            # 领域工序（无状态、规则单一来源；llm 由调用方注入）
│   ├── schema.py      #   GenerationPlan / Element / to_prompt_text（结构 + 兜底渲染）
│   ├── prompts.py     #   PLAN/CRITIQUE/RENDER/REVISE 提示词唯一住处（勿在别处复制 8 条规则）
│   ├── extraction.py  #   create_generation_plan（生成 + 格式自修 + 修订上下文）
│   ├── critique.py    #   critique_plan → CritiqueVerdict（fail-open：故障按通过，由人把关兜底）
│   └── rendering.py   #   build_final_prompt（失败兜底 plan.to_prompt_text()）
├── llm/               # 传输适配层（只管传输，不知道「大脑/工人/评审」是什么角色）
│   ├── base.py        #   ChatClient 协议 + ToolCall + UsageStat
│   ├── deepseek.py    #   DeepSeekChat：chat + complete_tools（原生 function calling）
│   ├── local_qwen.py  #   LocalQwenChat：仅 chat（懒加载，首次调用才占显存）
│   └── factory.py     #   make_chat_client(provider, …)
├── agentkit/          # 通用 agent 机制（零图像知识，可复用到别的 agent）
│   ├── tool.py        #   Tool(name/desc/args_schema/run/gate) + ToolRegistry（MCP 同构）
│   ├── source.py      #   ToolSource 协议：discover(config, ctx) → list[Tool]
│   ├── runtime.py     #   RuntimeCtx（整场会话上下文）+ agent_act()：ReAct 单步 / 门控 / 异常回喂（无预算）
│   ├── policy.py      #   POLICY：大脑唯一需要的领域知识（生图流程编排提示词，单块）
│   ├── io.py          #   IO 抽象 + ConsoleIO
│   └── usage.py       #   Usage（整场用量记账；无步数/调用上限）
├── tools/             # 本 agent 的内置工具来源
│   ├── __init__.py    #   build_tools(config)：环境决定注册集
│   ├── plan.py        #   make_plan：生成/修订 GenerationPlan（工人 = ctx.worker）
│   ├── review.py      #   review：人机评审会（机器评审 + 你把关 = 确认门）
│   ├── render.py      #   render：定稿 → 最终提示词（gate：需 confirmed）
│   └── image.py       #   generate：真实出图（gate：需 confirmed 且 prompt 出自 render）
├── generation/        # 生图执行器（重型依赖，本轮不动结构）
├── sft/               # 离线数据管线（原地；import 已改为 domain/llm 新位置）
└── tests/             # 零依赖 runner（python -m tests）
```

**规则摆放原则**（避免副本漂移）：
- 大脑（ReAct 决策者）只读 `agentkit/policy` 一块编排提示词 —— plan 领域规则对它无用（它不生成 JSON Plan）。
- plan 领域规则（8 条 / identity / schema / 修正消息）只在 `domain/prompts.py`，worker 侧由工具代码固定拼进调用上下文。训练侧（`sft/` 建数据、qwen SFT）与推理侧必须同源（都引用这里）。
- 传输层不持有任何角色提示词；`domain` 不碰任何 SDK；`agentkit` 不认识「图片 / GenerationPlan / 生图」。

## 3. 主循环时序（main 单 while + agent_act 单步）

**main 直接就是一个 while**：整场会话一份 transcript（`history`），ctx 建一次、不重置。
每次用户输入、工具调用结果、模型返回都按序进同一份历史，模型基于完整上下文反复执行，
直到你 exit、整场对话超出上下文（提示重跑 main），或大脑连续传输失败交回你。

```
ctx = RuntimeCtx(整场唯一); history = []
while True:
    if human_turn:
        req = io.ask("> ");   exit/退出 → break
        ctx.user_input = req; history += user(req); human_turn = False
    else:
        final = agent_act(ctx, history)   # 一次大脑往返
        if final is None: continue        # 大脑还在调工具，不读输入
        io.out("[结果] " + final); human_turn = True
```

`agent_act` 一次往返：
```
(text, tool_calls) = brain.complete_tools(
    [system: POLICY] + history,
    [tool.openai_schema() for tool in registry.tools],
)
history += assistant(content=text, tool_calls=…wire…)   # 无 tool_calls 也是这条消息=模型收尾
for call in tool_calls:    # 原生 function calling 可并行多个
    observation = dispatch(ctx, call)      # 校验 → gate → run，全部异常转观察
    history += tool(role, tool_call_id=call.id, content=observation)
```

- **门控**：`Tool.gate(ctx, args)` 自报前置条件，返回字符串即拦截成观察（工具不执行）。render/generate 都靠它实现「确认门」。
- **无预算**：不加步数 / API 调用次数上限。结束 = 你 exit / `ContextLimit`（错误含上下文超限特征）→ 提示重跑 main / 大脑连续 3 次传输失败 → 交回你决定（防基础设施故障无限空转，非流程限制）。
- **异常回喂**：大脑调用单次 RuntimeError → 拼一条 user 消息让大脑重试/换策略；工具执行异常 → 观察文本。
- **usage 记账**：每个 ChatClient 自己累计 `UsageStat`；`agent_act` 每步把增量并入 `ctx.usage`（**会话级，跨消息不清零**），exit 打印各客户端整场用量。
- **连续会话状态**：`ReviewState`（版本/定稿/意见/渲染提示词）挂在 `ctx.state` 跨消息持续——上一版已确认计划仍在，「在刚才基础上加 XX」由 make_plan 续版（见 §4）。

## 4. 工具与确认门

| 工具 | 做什么 | gate（拦截条件） |
|---|---|---|
| `make_plan` | 生成 / 修订 / **续写** GenerationPlan（含格式自修）。修订自动带 review 的问题+意见；`base="prev"` 在上一版计划上按新需求续写（跨消息「加 XX」）；默认=新主题开新线程（清空旧状态） | 修订/续版没有上一版可改时给错误观察 |
| `review` | **人机评审会**：机器 `critique_plan` 挑问题 → 终端问答（回车/p/q/补意见）| 无（必须已 make_plan 才有内容可评，否则返回 error） |
| `render` | 定稿 → 最终提示词，记入会话状态 | 无 `review=confirmed` 定稿 |
| `generate` | 真实出图 | 无 confirmed 定稿 / 未先 render / 传入 prompt 与 render 结果不一致 |

会话状态挂在 `ctx.state["review"]`（`ReviewState`）：版本、最新一版、**定稿**、累计人意见、待修订问题/意见、渲染出的提示词。它**以会话为界**（不再按消息重置）：上一版定稿跨消息可见，供 `make_plan` 续版；只在用户开新主题时由 make_plan 清空（防旧主题意见串场）。大脑不需要复述计划 JSON —— 状态由工具自动追踪，参数大多留空即可。

## 5. review 人机评审会（确认门的语义）

```
review_meeting: 展示当前计划 → 机器评审（含你历轮的补充意见 user_additions，不误判）
  ├─ 机器发现问题 → ① 回车=按问题修正 | p=强制通过 | q=放弃 | 直接输入文字=一条意见（跳过②）
  │                 若①回车 → ② 你的修改意见？（回车=无）
  ├─ 机器通过     → 回车=确认通过（定稿）| 直接输入文字=意见（再修一轮）| q=放弃
  └─ 结果三种：confirmed / revise / aborted
revise → 问题+意见挂入会话状态（pending_*），下次 make_plan 自动带上（ReAct 修订环）
confirmed → 该稿成为定稿（render/generate 唯一可用稿）
```

机器评审 fail-open：评审 API 故障时按通过处理 —— 因为最终放行权始终在你这道 human gate，机器故障不会造成静默漏检，只会少一层「预检」。

## 6. 三条规范接缝（将来不洗内部）

1. **Tool ≈ MCP 工具**：`args_schema: BaseModel`，`args_schema.model_json_schema()` 即 MCP `inputSchema`；
   对 OpenAI 的 function schema 由 runtime 统一生成（`Tool.openai_schema()`），新增工具只需写契约三件套 + run/gate。
2. **ToolSource**：`discover(config, ctx) → list[Tool]`。内置来源 = `tools.build_tools`；将来接 MCP 就是再加一个实现（server → Tool 包装），主循环不感知。
3. **ChatClient**：传输只此一个门面。`chat`（工人）与可选 `complete_tools`（大脑）分能力实现；
   失败统一抛 RuntimeError；usage 自记。接新模型 = 新子类 + factory 一行，角色/流程不动。

## 7. 环境能力分层（config）

- `AGENT_IMAGE_MODEL` 非空 = 生图后端可用 → 注册集含 `generate`；留空 = 干跑（只出计划/提示词）。
  **能不能出图由环境决定，不由请求内容决定**（结构上防越权）。
- `AGENT_WORKER_PROVIDER=deepseek|qwen` 切 plan 工人；qwen 走 `LocalQwenChat(+LoRA)`，仅 chat。
- 大脑本轮固定 DeepSeek（原生 function calling 最省事）；将来本地 qwen 升级为大脑需能吐规范 tool_calls（见 §9 清单 #9）。
- 所有键可省略（见 `.env.example` 注释与 `config.py` 文档字符串）。

## 8. 计费与结束语义

- 单个需求通常 ≈ **4~10 次** API 调用（规划 1+ 评审 1+ 可能修订 + 渲染 1 + 大脑循环若干）。
- **结束 = 你 exit / 上下文超限 / 大脑连续失败交回**：不设步数或调用次数上限（会话长度 = 你自己跟模型聊多久）。
  上下文超限时打印「重跑 main 开新会话」；退出时打印各客户端整场 usage 汇总。
- **省钱纪律**：离线回归全部走 `tests/` 假客户端（整场驱动 main，不触网不占显存）；真机/GPU 冒烟先征得同意再跑。

## 9. 验证与回归

```bash
# 离线全绿（零依赖 runner，无需 pytest）
cd generation-agent
/path/to/anaconda3/envs/llm/bin/python -m tests     # 期望 61/61 通过

# 编译检查（含 sft/，只编译不 import）
/path/to/anaconda3/envs/llm/bin/python -m compileall -q .

# 真机冒烟（先问再跑 / 或由你亲自跑）
/path/to/anaconda3/envs/llm/bin/python main.py          # 大脑会话：喂「只要提示词」与「出一张…图」
```

## 10. 后续可加项

见会话 Plan §4 清单（ask_user 澄清 / raw+plan 双图 / seed 多稿 / VLM 观察闭环 / 作品集记忆 /
qwen 训练完成切工人 / MCP+skill 化 / qwen 升大脑）。实现方式要点都写在清单里，多数是「新 `Tool` + policy 一行 + config 键」的组合，不需动主循环。
