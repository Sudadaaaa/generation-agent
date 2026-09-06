# generation-agent

把自然语言的图片生成需求转换为严格的结构化计划（GenerationPlan），再由本地生图模型渲染成图。

## 流程

```
用户需求
   ↓
agent/ 编排层（计划→评审→修正→通过 循环）
   ├─ planning/  LLM 结构化提取（DeepSeek 云端 API 或本地 Qwen3）
   ├─ 评审：独立模型（缺省 DeepSeek 云端）对照需求检查遗漏/矛盾/偏离，
   │            不通过则带问题修正（避免同模型自我评审盖章式通过）
   ↓
GenerationPlan → LLM 渲染为自然语言提示词（语言随 plan，不写死中文）
   ↓
generation/  本地出图（Z-Image / FLUX）→ PNG
```

## 安装

    pip install -r requirements.txt

所有模型从本地 HF 缓存加载（`local_files_only=True`），需预置 Qwen3、Z-Image、FLUX 缓存。

## 使用

    cp .env.example .env    # 填入 DEEPSEEK_API_KEY，按需改 LLM_MODEL

    python main.py "一个女孩在夕阳下放风筝"
    python main.py "一只橘猫在窗台上打盹" --no-image
    python main.py "..." --llm qwen
    python main.py          # 交互模式（输入 exit 退出）

GPU 分配：固定逻辑卡号，物理卡在 `main.py` 顶部指定

本地 LLM（`--llm qwen`）固定用逻辑 `cuda:0`，生图固定用逻辑 `cuda:1`。
两者是**逻辑卡号**——映射到哪张物理卡由 `main.py` 顶部的
`CUDA_VISIBLE_DEVICES` 决定（换卡只改那一行）：

    os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
    # → 逻辑 0 = 物理卡 4（LLM），逻辑 1 = 物理卡 5（生图），两模型不抢卡

参数：

- `input`：图片生成需求；省略则进入交互模式
- `--llm {deepseek,qwen}`：生成计划的模型提供方，缺省 deepseek
- `--model {zimage,flux,…}`：生图模型，缺省 zimage
- `--output-dir`：图片输出目录，缺省 outputs
- `--no-image`：只生成计划，不出图

## 结构

```
agent/      编排层：持有「提取→评审→修正→通过」循环（真 agent 逻辑）
planning/   提取 + 渲染层：结构化提取（schema / prompts / llm / extract）
            与 Plan→最终提示词的 LLM 渲染（render）
generation/ 图片生成层（factory + backends: base / zimage / flux）
main.py     CLI 入口
```
