import os
import sys
import torch
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from accelerate import Accelerator
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup
)

# train/ 在仓库根下一层，把【仓库根】插进 sys.path，下面才能 `from mydatasets import ...`。
# 用 __file__ 反推而不是靠 cwd，所以从哪个目录调用都成立。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mydatasets import get_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def parse_args():
    p = ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace 模型 id 或本地路径")
    p.add_argument("--dataset", default="plan_sft", help="数据集名；目前只支持 plan_sft")
    p.add_argument("--num_samples", type=int, default=None, help="取数据集前 N 条；不传(默认)则用全部数据")
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--resume", default=None, help="从该 checkpoint 目录恢复，如 outputs/sft_lora/checkpoint-200")
    p.add_argument("--output_dir", default="outputs/sft_lora")
    p.add_argument("--enable_thinking", type=lambda x: x.lower() != "false", default=True,
                   help="Qwen3 thinking 模式（True=思考后回答）")
    p.add_argument("--grad_checkpoint", type=lambda x: x.lower() != "false", default=True,
                   help="gradient checkpointing（省显存）")
    return p.parse_args()

def save_lora(accelerator, model, tokenizer, output_dir, e):
    accelerator.wait_for_everyone()
    check_point_path = os.path.join(output_dir, f"checkpoint-{e}")
    os.makedirs(check_point_path, exist_ok=True)
    accelerator.save_state(check_point_path)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)   # ③ 剥掉 DDP 包装层
        unwrapped.save_pretrained(output_dir)         # ④ 存 LoRA adapter（几十MB）
        tokenizer.save_pretrained(output_dir)         # ⑤ 存 tokenizer 文件

        old_checkpoint_path = os.path.join(output_dir, f"checkpoint-{e - 1}")
        if os.path.exists(old_checkpoint_path):
            shutil.rmtree(old_checkpoint_path)

def main():
    args = parse_args()

    # ⚠️ 顺序敏感：ZeRO-3 下必须先创建 Accelerator，它内部会激活 transformers 的
    # HfDeepSpeedConfig(stage3)。之后 from_pretrained 才会被自动包进
    # deepspeed.zero.Init()，让每张卡从"出生"就只持有 1/4 的权重。
    # 若先 from_pretrained 再建 Accelerator，整份权重已落进内存，分片形同虚设，
    # 训练时每卡仍会扛着完整模型（实测 prepare 后看似 3.8G，一步就冲回 23G）。
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    )
    model.train()
    model.config.use_cache = False
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )   # 激活值不缓存、反向重算，省显存
        model.enable_input_require_grads()   # grad checkpoint 需要嵌入层也 require_grad 才能回传

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = get_dataset(args, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, 
        eps=1e-7, 
        weight_decay=0.01,
    )

    total_step = args.epochs * len(dataloader) // accelerator.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_step * 0.03),
        num_training_steps=total_step,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    e = 0
    if args.resume:
        accelerator.load_state(args.resume)
        e = int(args.resume.split("-")[-1]) + 1
        accelerator.print(f"从第{e}轮开始")

    while e < args.epochs:
        if accelerator.is_main_process:
            bar = tqdm(dataloader, disable= not accelerator.is_main_process)
            bar.set_description('Train Epoch({}/{})'.format(e, args.epochs))
        else:
            bar = dataloader
        for i, batch in enumerate(bar):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],         # 喂输入
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],               # 喂目标
                )

                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                learn_rate = optimizer.param_groups[0]['lr']
                if accelerator.is_main_process:
                    bar.set_postfix({'sum_loss':f'{loss:.4f}', 'lr':f'{learn_rate:.8f}'})

        save_lora(accelerator, model, tokenizer, args.output_dir, e)
        e += 1
    return

if __name__ == "__main__":
    main()