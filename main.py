import argparse

from dotenv import load_dotenv

from agent.backends import BaseImageBackend
from agent.image_gen import create_image_generator
from agent.planner import create_generation_plan


def run_once(
    user_input: str,
    generator: BaseImageBackend | None,
) -> None:
    plan = create_generation_plan(user_input)

    print("=" * 60)
    print("Generation Plan")
    print("=" * 60)

    print(plan.model_dump_json(indent=2))

    prompt = plan.to_prompt_text()

    print("\n--- 渲染提示词 ---")
    print(prompt)

    if generator is not None:
        print("\n--- 生成图片 ---")
        print("正在加载模型并生成，请稍候……")

        # 1) 用原始用户提示词出图
        raw_path = generator.generate(user_input, tag="raw")
        print(f"[原始提示词] 图片已保存：{raw_path}")

        # 2) 用 gen_plan 结构化提示词出图
        plan_path = generator.generate(prompt, tag="plan")
        print(f"[gen_plan提示词] 图片已保存：{plan_path}")

    print("=" * 60 + "\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="把图片生成需求转换为结构化计划并出图"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="图片生成需求；省略则进入交互模式",
    )
    parser.add_argument(
        "--model",
        help=(
            "生图模型：zimage / flux / 或完整本地模型 id；"
            "默认取 GEN_MODEL 环境变量（缺省 zimage）"
        ),
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="只生成计划，不生成图片",
    )
    args = parser.parse_args()

    # 惰性加载：即使创建对象也不占用显存，首次 generate 时才加载模型
    generator = None if args.no_image else create_image_generator(
        model=args.model,
    )

    print("=" * 60)
    print("Generative AI Generation Planner")
    print("=" * 60)

    if args.input:
        run_once(args.input, generator)
        return

    print("\n请输入你的生成需求。")
    print("输入 exit 退出。\n")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() == "exit":
            print("Bye!")
            break

        if not user_input:
            print("请输入一些内容。")
            continue

        try:
            run_once(user_input, generator)
        except Exception as exc:
            print("\n发生错误：")
            print(exc)
            print()


if __name__ == "__main__":
    main()
