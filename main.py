from dotenv import load_dotenv

from agent.planner import create_generation_plan


def main() -> None:
    load_dotenv()

    print("=" * 60)
    print("Generative AI Generation Planner")
    print("=" * 60)

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
            plan = create_generation_plan(user_input)

            print("\n" + "=" * 60)
            print("Generation Plan")
            print("=" * 60)

            print(plan.model_dump_json(indent=2))

            print("=" * 60 + "\n")

        except Exception as exc:
            print("\n发生错误：")
            print(exc)
            print()


if __name__ == "__main__":
    main()