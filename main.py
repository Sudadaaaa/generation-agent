import os

# 物理卡选择：本地 LLM 固定逻辑 cuda:0、生图固定逻辑 cuda:1。
# 这里决定这两张逻辑卡映射到哪两张物理卡（换卡只改这一行）。
# 必须在任何 torch 导入之前设置，否则掩码不生效。
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import argparse

from dotenv import load_dotenv

from agent import agent_plan_review
from generation.backends import BaseImageBackend
from generation.factory import create_image_generator
from planning import PlanLLM, create_plan_llm
from planning.render import build_final_prompt


def run_once(
    user_input: str,
    generator: BaseImageBackend | None,
    llm: PlanLLM | None = None,
) -> None:
    llm = llm or create_plan_llm()

    # 人机评审循环：每版计划展示给用户，合并用户意见与评审问题回传修正，
    # 直到用户回车确认通过（或 p 强制通过）才继续；q/放弃则返回 None。
    # 计划 JSON 已在循环内逐版打印，这里不再重复。
    plan = agent_plan_review(user_input, llm=llm)

    if plan is None:
        print("\n本次已放弃，未生成图片。\n")
        return

    # 走到这里 = 计划已被用户确认 → 才渲染最终提示词、才出图（决策：确认后才出图）
    print("\n--- 渲染最终提示词 ---")
    prompt = build_final_prompt(plan, llm, user_input=user_input)
    print(prompt)

    if generator is not None:
        print("\n--- 生成图片 ---")
        print("正在加载模型并生成，请稍候……")

        # 1) 用原始用户提示词出图
        raw_path = generator.generate(user_input, tag="raw")
        print(f"[原始提示词] 图片已保存：{raw_path}")

        # 2) 用确认后计划渲染的结构化提示词出图
        plan_path = generator.generate(prompt, tag="plan")
        print(f"[gen_plan提示词] 图片已保存：{plan_path}")

    print("=" * 60 + "\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="把图片生成需求转换为结构化计划并出图"
    )
    parser.add_argument(
        "--model",
        help="生图模型：zimage / flux / 或完整本地模型 id；缺省 zimage",
    )
    parser.add_argument(
        "--output-dir",
        help="图片输出目录；缺省 outputs",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="只生成计划，不生成图片",
    )
    parser.add_argument(
        "--llm",
        choices=["deepseek", "qwen"],
        help=(
            "生成计划的模型提供方：deepseek（DeepSeek 云端 API）/ "
            "qwen（本地 Qwen3-8B）；缺省 deepseek"
        ),
    )
    args = parser.parse_args()

    # 惰性加载：即使创建对象也不占用显存，首次 generate 时才加载模型
    generator = None if args.no_image else create_image_generator(
        model=args.model,
        output_dir=args.output_dir,
    )

    # Plan 生成模型：也惰性加载（qwen 首次 complete 才占显存）
    llm = create_plan_llm(args.llm)

    print("=" * 60)
    print("Generative AI Generation Planner")
    print("=" * 60)

    print("\n请输入你的生成需求（exit 退出）。\n")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() == "exit":
            print("Bye!")
            break

        if not user_input:
            print("请输入一些内容。")
            continue

        try:
            run_once(user_input, generator, llm)
        except Exception as exc:
            print("\n发生错误：")
            print(exc)
            print()


if __name__ == "__main__":
    main()
