"""本地 Qwen3-8B 传输实现（plan 工人用）。

transformers 进程内加载，惰性（首次 chat 才占显存）。可选 LoRA adapter 叠加
（PEFT）。本类只做 plan 工人，**不做大脑**：不实现 complete_tools（原生
function calling 的本地化依赖 OpenAI 兼容服务/微调，见 §4 清单 #9）。

torch / transformers / peft 全部在 _load 内惰性导入——模块可被轻量测试导入，
不拉重型依赖。
"""

from __future__ import annotations

import os
import re
from typing import Any

from llm.base import ChatClient

# Qwen3 思考块：<think>...</think>，剥掉只留最终回答
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """剥掉 Qwen3 推理产出的思考块，返回纯回答文本。"""
    return _THINK_RE.sub("", text).strip()


class LocalQwenChat(ChatClient):
    """本地 Qwen3-8B：transformers 进程内加载（仅 chat，plan 工人用）。

    local_files_only 强制只用本地缓存（本机网络不可达 HF hub）。
    可选 lora_path：在基座模型上叠加 PEFT LoRA adapter（train_sft.py 保存的
    目录，含 adapter_config.json）。
    """

    provider = "qwen"

    def __init__(
        self,
        model: str | None = None,
        *,
        lora_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 40960,
        enable_thinking: bool = True,
    ) -> None:
        model = model or "Qwen/Qwen3-8B"
        super().__init__(model=model)
        # 固定逻辑卡号：本地 LLM 用 cuda:0。配合 CUDA_VISIBLE_DEVICES
        # 控制它映射到哪张物理卡（main.py 顶部设置）。
        self.device = device or "cuda:0"
        self.lora_path = lora_path or ""
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch  # 惰性：不占用时不必导入
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
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
            except ImportError as exc:  # pragma: no cover - 环境提示
                raise RuntimeError(
                    "使用 LoRA（lora_path）需要安装 peft：pip install peft"
                ) from exc

            print(f"[planner] 加载 LoRA adapter：{self.lora_path}")
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            self._model.eval()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = True,  # 本地模型本就不约束 JSON 输出，此参数忽略
    ) -> str:
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

        # 用量近似记账：以 token 序列长度计（本地不产生 API 费用，仅用于统计步数/规模）
        self._usage.add_values(
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(new.shape[0]),
        )

        if not content:
            raise RuntimeError("本地 Qwen3 返回了空内容（可能只输出了思考块）。")

        return content
