from .plan_sft import process_plan_sft

def get_dataset(args, tokenizer, split="train"):
    name = args.dataset
    if name in ("plan_sft", "mydatasets/plan_sft.jsonl"):
        # 防呆：plan_sft 的系统提示词约 2962 token，若 max_seq_length 比它还短，
        # 整个对话被截断、assistant 回复进不了窗口 → labels 全 -100 → loss=nan
        if args.max_seq_length < 3600:
            import warnings
            warnings.warn(
                f"max_seq_length={args.max_seq_length} < plan_sft system prompt 长度(~2962)，"
                "样本会被截断到 assistant 回复之前，监督目标全 -100 导致 loss=nan，请用 ≥4096（建议 8192）"
            )
        return process_plan_sft(
            tokenizer,
            num_samples=args.num_samples,
            max_seq_length=args.max_seq_length,
            enable_thinking=args.enable_thinking,
            split=split,
        )
    raise ValueError(f"未知数据集: {name}")
