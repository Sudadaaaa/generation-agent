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

def save_lora(accelerator, model, tokenizer, output_dir, e, n_trainable):
    accelerator.wait_for_everyone()
    # save_state 是集合操作，每个 rank 都得进，别挪进下面的 is_main_process。
    check_point_path = os.path.join(output_dir, f"checkpoint-{e}")
    os.makedirs(check_point_path, exist_ok=True)
    accelerator.save_state(check_point_path)

    # ④ 取全量参数。⚠️ 这一行**四个 rank 必须一起执行**，不能挪进下面的 is_main_process：
    # DeepSpeed 的 _zero3_consolidated_16bit_state_dict 内部是 GatheredParameters(
    # modifier_rank=0) —— 集合通信，全量张量只落在 rank 0、其余 rank 拿到 None
    # （docstring: "must be called on all ranks and not just rank 0"）。
    # 只让 rank 0 进，它会卡在这里等一个永远不来的对端。实测踩过：另外三个 rank 训完
    # 就退出了，rank 0 独自占着 13.4G 显存空转十几分钟、一个字节没写出来。
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)   # ③ 剥掉 DDP 包装层
        # 为什么非得换这一份：ZeRO-3 下 unwrap 出来只是本卡那一份分片，没被 materialize
        # 的参数会**以空张量**写进 safetensors——不报错、文件大小也正常，要到起服务加载
        # 时才炸。（踩过：MLP 的 gate/up/down 各少一侧，43,646,976 个 LoRA 参数只存下
        # 22,413,312，51.4%，adapter 仍是完整的 87MB。）
        # get_state_dict 走的就是 accelerator.save_state 落那份 16bit 全量模型用的同一条路；
        # peft 的 save_pretrained 收到这份 dict 后只挑 "lora_" 键，底模权重不会进 adapter。
        n_saved = sum(v.numel() for k, v in state_dict.items() if "lora_" in k)
        if n_saved != n_trainable:
            # 宁可这轮训练白跑，也别再交出一个"能用但学残了"的 adapter。
            raise RuntimeError(
                f"adapter 不完整：gather 到 {n_saved:,} 个 LoRA 参数，训练时是 "
                f"{n_trainable:,} 个——差的那部分会以空张量写出去。本次保存作废。"
            )
        unwrapped.save_pretrained(output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(output_dir)         # ⑤ 存 tokenizer 文件
        accelerator.print(f"adapter 已存至 {output_dir}（LoRA 参数 {n_saved:,} 个）")

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
    # 存 adapter 时的核对基准（见 save_lora）。必须在 prepare **之前**取：ZeRO-3 分片
    # 之后每张卡只看得见自己那一份，这个数就再也算不出来了。
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

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

        save_lora(accelerator, model, tokenizer, args.output_dir, e, n_trainable)
        e += 1
    return

if __name__ == "__main__":
    main()