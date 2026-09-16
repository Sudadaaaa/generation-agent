import json
import os
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prompt import build_plan_system_prompt
from datasets import Dataset

# 本地 plan_sft 数据的固定路径（与本文件同目录）
PLAN_SFT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_sft.jsonl")
SYSTEM_PROMPT = build_plan_system_prompt()

# val 占比。jsonl 按 id 有序、每个 id 恰好两行(en/zh)。
# 2% ≈ 31 个 id / 62 行：这个量级同时作评估留出集（原计划按 64 条估的调用量）。
# 评估侧由 eval/run_eval.py 现场调 plan_sft_rows("val") 取这段，不落盘副本——
# 这里改一下，训练和评估同时生效。
VAL_FRAC = 0.02


def plan_sft_rows(split="train", num_samples=None):
    """读 plan_sft.jsonl 并按 VAL_FRAC 切片，返回原始行（含 messages）。

    切点只此一处，且不碰 tokenizer，所以训练与评估共用同一个函数——评估侧的留出集
    因此不需要另存一份副本，也就不存在「改了 VAL_FRAC 但文件没重导」这种不报错的漂移。

    split="train"/"val"：前段 train、后段 val。切点按行数算出并落在 id 对边界上，
    同一个 id 的中英两行不会被切到两侧。

    顺带补出 language 字段：文件按 (id, language) 排序，同一 id 恰好两行、en 在前
    zh 在后，所以「该 id 是否已出现过」就能定出语言（实测与元数据真值 3088/3088 一致），
    不必去回查 process_data 那份台账。
    """
    rows = []
    with open(PLAN_SFT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    index = round(len(rows) / 2 * (1 - VAL_FRAC)) * 2
    rows = rows[:index] if split == "train" else rows[index:]

    seen = set()
    for r in rows:
        r["language"] = "zh" if r["id"] in seen else "en"
        seen.add(r["id"])

    if num_samples is not None:
        rows = rows[: min(num_samples, len(rows))]
    return rows


def process_plan_sft(tokenizer, num_samples=None, max_seq_length=8192,
                     enable_thinking=True, split="train"):
    """
    读本地 plan_sft.jsonl（自研 GenerationPlan 规划数据），
    分词并做 assistant 段 loss 掩码，返回含 input_ids/labels 的 Dataset。
    mask 逻辑：只对 <|im_start|>assistant ... 区间计算 loss，其余位置填 -100。

    行怎么选见 plan_sft_rows()。
    """
    rows = plan_sft_rows(split, num_samples)

    # datasets 库的 Dataset 只当"容器"用，本文件只需要它的 messages 列。
    # jsonl 只存 (user, assistant)，system 在这里注入——唯一真源见文件头 SYSTEM_PROMPT。
    data = Dataset.from_list([
        {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + r["messages"]}
        for r in rows
    ])

    def tokenize_fn(batch):
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            for msgs in batch["messages"]
        ]

        # 扫描每个 <|im_start|>assistant ... 的 (起始, 结束) 字符区间 → 监督目标
        ASSISTANT_MARK = "<|im_start|>assistant"
        label_ranges = []
        for text in texts:
            ranges = []
            pos = 0
            while True:
                start = text.find(ASSISTANT_MARK, pos)
                if start == -1:
                    break
                end = text.find("<|im_start|>", start + len(ASSISTANT_MARK))
                if end == -1:
                    end = len(text)
                ranges.append((start, end))
                pos = end
            label_ranges.append(ranges)

        tok = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        input_ids = tok["input_ids"]
        offsets = tok["offset_mapping"]

        # labels：token 字符区间与 assistant 区间重叠 → 复制该 token id；否则 -100
        labels = []
        for ids, offs, ranges in zip(input_ids, offsets, label_ranges):
            lab = []
            for tid, (s, e) in zip(ids, offs):
                if any(s < r_end and e > r_start for r_start, r_end in ranges):
                    lab.append(tid)
                else:
                    lab.append(-100)
            labels.append(lab)

        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
            "labels": labels,
        }

    return data.map(tokenize_fn, batched=True, remove_columns=data.column_names)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ds = process_plan_sft(tok, num_samples=10, max_seq_length=8192)
    print("列:", ds.column_names)
    print("长度:", len(ds))
    for i in range(3):
        lab = ds[i]["labels"]
        n = sum(1 for x in lab if x != -100)
        print(f"样本{i}: seq_len={len(lab)}, 有效监督token={n}")
