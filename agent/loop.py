"""agent 编排层：把「提取→评审→修正→通过」的循环跑起来。

这是 agent 之所以是 agent 的地方：不是执行一次固定脚本，而是
持有目标（满足用户需求）+ 循环与反馈（评审结果驱动重做）+ 决策
（是否接受计划）。
"""

from agent.critique import critique_plan
from agent.prompts import build_revise_message
from planning.extract import create_generation_plan
from planning.llm import PlanLLM, create_plan_llm
from planning.schema import GenerationPlan


def agent_plan(
    user_input: str,
    llm: PlanLLM | None = None,
    critic_llm: PlanLLM | None = None,
    max_critique_rounds: int = 2,
) -> GenerationPlan:
    """生成一份通过评审的 GenerationPlan。

    流程：
      1. create_generation_plan 提取出格式合法的计划
      2. critique_plan 对照用户需求评审
      3. 通过 → 返回；不通过 → 带着评审问题让提取器修正，回到 2
      4. 达到 max_critique_rounds 仍未通过 → 打印剩余问题，接受当前计划

    评审模型与规划模型解耦：critic_llm 缺省固定用 DeepSeek 云端，
    即使规划用的是本地 Qwen3（llm），点评也交给独立的 DeepSeek——
    避免同一模型自我评审「盖章式通过」；critic_llm 仅供测试/高级用法覆盖。
    """

    llm = llm or create_plan_llm()
    critic = critic_llm or create_plan_llm("deepseek")

    plan = create_generation_plan(user_input, llm=llm)

    for round_no in range(max_critique_rounds + 1):
        verdict = critique_plan(user_input, plan, critic)

        if verdict.ok:
            print("[agent] 计划评审通过")
            return plan

        print(
            f"[agent] 计划评审发现问题（第 {round_no + 1} 轮，"
            f"共 {len(verdict.problems)} 条）："
        )

        for problem in verdict.problems:
            print(f"  - {problem}")

        if round_no >= max_critique_rounds:
            print("[agent] 已达最大修正轮数，接受当前计划")
            return plan

        print(f"[agent] 进入第 {round_no + 1} 轮修正…")

        plan = create_generation_plan(
            user_input,
            llm=llm,
            context_messages=[
                {
                    "role": "assistant",
                    "content": plan.model_dump_json(indent=2),
                },
                {
                    "role": "user",
                    "content": build_revise_message(verdict.problems),
                },
            ],
        )

    return plan  # 循环内已返回，此处仅为类型收敛
