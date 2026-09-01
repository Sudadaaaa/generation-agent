import os
import re
from abc import ABC, abstractmethod

import torch
from openai import OpenAI


def create_client() -> OpenAI:
    """Create a DeepSeek API client."""

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


class PlanLLM(ABC):
    """把对话消息转换为一段文本回复的模型后端（Plan 生成提供方）。

    子类实现 complete()：输入对话消息，返回模型输出的文本内容，
    供 parse_generation_plan 解析为 GenerationPlan。
    新增模型 = 新增一个子类 + 在 create_plan_llm 里注册。
    """

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """调用模型，返回其文本回复。空内容/调用失败应抛 RuntimeError。"""


class DeepSeekPlanLLM(PlanLLM):
    """DeepSeek 云端 API（OpenAI 兼容接口）。"""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL") or "deepseek-v4-flash"
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = create_client()
        return self._client

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._ensure_client().chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=50000,
            stream=False,
        )

        choice = response.choices[0]
        message = choice.message
        content = message.content

        if not content:
            # deepseek-v4-flash 是推理模型：reasoning_content 与 content 都计入
            # max_tokens，推理过长或瞬时故障时 content 可能为空。带上现场信息方便排查。
            reasoning = getattr(message, "reasoning_content", None) or ""

            raise RuntimeError(
                "DeepSeek 返回了空内容（message.content 为空）。"
                f" finish_reason={choice.finish_reason!r}, "
                f"usage={response.usage!r}, "
                f"reasoning_content 长度={len(reasoning)}。"
                "可能是 max_tokens 被推理内容占满或瞬时故障。"
            )

        return content


# Qwen3 思考块：<think>...</think>，剥掉只留最终回答
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """剥掉 Qwen3 推理产出的思考块，返回纯回答文本。"""
    return _THINK_RE.sub("", text).strip()


class LocalQwen3PlanLLM(PlanLLM):
    """本地 Qwen3-8B：transformers 进程内加载，保留思考模式。

    local_files_only 强制只用本地缓存（本机网络不可达 HF hub）。
    模型惰性加载，首次 complete() 才占显存。
    可选 lora_path：在基座模型上叠加 PEFT LoRA adapter（train_sft.py
    保存的目录，含 adapter_config.json），用于加载微调后的权重。
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        lora_path: str | None = None,
        max_new_tokens: int = 4096,
        enable_thinking: bool = True,
    ) -> None:
        self.model_path = model_path or "Qwen/Qwen3-8B"
        # 固定逻辑卡号：本地 LLM 用 cuda:0。配合环境变量 CUDA_VISIBLE_DEVICES
        # 控制它映射到哪张物理卡，例如 CUDA_VISIBLE_DEVICES=4,5 时 cuda:0 = 物理卡 4。
        self.device = device or "cuda:0"
        self.lora_path = lora_path or ""
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        )

        if self.lora_path:
            if not os.path.isdir(self.lora_path):
                raise RuntimeError(
                    f"lora_path 目录不存在：{self.lora_path!r}。"
                    "应指向 train_sft.py 保存 LoRA adapter 的目录"
                    "（含 adapter_config.json）。"
                )
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "使用 LoRA（lora_path）需要安装 peft：pip install peft"
                ) from exc

            print(f"[planner] 加载 LoRA adapter：{self.lora_path}")
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            self._model.eval()

    def complete(self, messages: list[dict[str, str]]) -> str:
        self._load()
        assert self._tokenizer is not None and self._model is not None

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        new = out[0][inputs["input_ids"].shape[1]:]
        decoded = self._tokenizer.decode(new, skip_special_tokens=True)
        content = _strip_think(decoded)

        if not content:
            raise RuntimeError("本地 Qwen3 返回了空内容（可能只输出了思考块）。")

        return content


def create_plan_llm(provider: str | None = None) -> PlanLLM:
    """按 provider 创建 Plan 生成模型后端。

    provider 缺省用 deepseek；命令行入口用 main.py 的 --llm 切换。
    本地 LLM（qwen）固定用逻辑 cuda:0，物理卡由调用方在 torch 导入前
    用环境变量 CUDA_VISIBLE_DEVICES 指定（main.py 顶部设置，换卡只改那里）。
    支持：
      - deepseek：DeepSeek 云端 API（默认）
      - qwen：本地 Qwen3-8B（transformers 进程内加载）
    """

    provider = (provider or "deepseek").strip().lower()

    if provider == "deepseek":
        return DeepSeekPlanLLM()

    if provider == "qwen":
        return LocalQwen3PlanLLM(device="cuda:0")

    raise ValueError(f"未知 LLM 提供方：{provider!r}（可选 deepseek / qwen）")
