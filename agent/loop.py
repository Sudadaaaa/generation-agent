"""agent 编排层：把「提取→评审→修正→通过」的循环跑起来。

这是 agent 之所以是 agent 的地方：不是执行一次固定脚本，而是
持有目标（满足用户需求）+ 循环与反馈（评审结果驱动重做）+ 决策
（是否接受计划）。

两个入口：
  - agent_plan          全自动循环，达到轮数即接受（headless/脚本用）。
  - agent_plan_review   人机循环：每轮把计划展示给用户，用户可补充意见、
                        强制通过或放弃，评审通过后仍须用户回车确认（main 用）。
"""

from agent.critique import critique_plan
from agent.prompts import build_revise_message
from planning.extract import create_generation_plan
from planning.llm import PlanLLM, create_plan_llm
from planning.schema import GenerationPlan


#: 用户在评审问答里输入「放弃/强制通过」时接受的词（宽容匹配，大小写不敏感）
_QUIT_WORDS = {"q", "quit", "exit", "放弃"}
_PASS_WORDS = {"p", "pass", "通过", "确认"}


def _is_quit(text: str) -> bool:
    """判断输入是否为「放弃本次」。"""
    return text.strip().lower() in _QUIT_WORDS


def _is_pass(text: str) -> bool:
    """判断输入是否为「强制通过」。"""
    return text.strip().lower() in _PASS_WORDS


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


def agent_plan_review(
    user_input: str,
    llm: PlanLLM | None = None,
    critic_llm: PlanLLM | None = None,
) -> GenerationPlan | None:
    """人机交互式评审：生成计划 → LLM 评审 → 展示给用户 → 回传修正 → 循环。

    与 agent_plan（全自动、达到轮数即接受）的区别：每一步决定权都交给用户。

    - 评审发现问题：问「① 回车 = 按评审问题修正 / p = 强制通过 / q = 放弃」，
      选修正后再问「② 你的修改意见」——用户的补充意见与评审问题**合并**
      一起回传给规划模型修正（用户意见单独成节、标为最高优先级）。
    - 评审通过：**仍需用户回车确认**才算通过；此时也可直接输入一条用户
      意见再修一轮。
    - 在①处直接输入其他文字 → 当作一条用户意见（跳过②）。
    - 用户历轮意见会被**累计**：既随修正消息回传给规划模型，也传给后续每轮
      评审模型（critique_plan 的 user_additions），避免评审把用户自己要求
      新增的内容误判成「凭空编造/与需求冲突」。

    返回通过确认的 GenerationPlan；用户放弃本次则返回 None。
    循环不设轮数上限——由用户用 p/q 主动收口。
    llm 缺省用 deepseek（规划），critic 缺省固定用 DeepSeek 评审（解耦，
    见 agent_plan 注释）。评审 infra 故障 fail-open 时落到「评审通过」
    分支，仍由用户回车把关，不硬放行。
    """

    llm = llm or create_plan_llm()
    critic = critic_llm or create_plan_llm("deepseek")

    plan = create_generation_plan(user_input, llm=llm)
    version = 1
    # 用户历轮补充的意见累计：规划模型每轮能看见，评审模型也必须看见，
    # 否则评审会把用户自己要求新增的内容误判成「凭空编造」。
    human_notes: list[str] = []

    while True:
        print(f"\n[agent] 当前 GenerationPlan（第 {version} 版）：")
        print(plan.model_dump_json(indent=2))

        verdict = critique_plan(
            user_input, plan, critic, user_additions=human_notes
        )

        if verdict.ok:
            # 情况 B：评审通过 → 仍须用户最终确认（或补意见再修一轮）
            print("[agent] 评审通过（未发现问题）。请最终确认：")
            reply = input(
                "  回车 = 确认通过；"
                "或直接输入你的修改意见（按意见再修正一轮）；"
                "q = 放弃本次\n> "
            ).strip()

            if _is_quit(reply):
                print("[agent] 本次已放弃。")
                return None

            if not reply or _is_pass(reply):
                print("[agent] 计划已由用户确认通过。")
                return plan

            problems: list[str] = []
            user_notes = [reply]

        else:
            # 情况 A：评审发现问题 → 两段式问答
            print(f"[agent] 评审发现 {len(verdict.problems)} 个问题：")
            for problem in verdict.problems:
                print(f"  - {problem}")

            act = input(
                "① 回车 = 按评审问题修正一轮 | p = 强制通过 | q = 放弃本次\n> "
            ).strip()

            if _is_quit(act):
                print("[agent] 本次已放弃。")
                return None

            if _is_pass(act):
                print("[agent] 用户强制通过，计划已确认。")
                return plan

            problems = verdict.problems
            if act:
                # ①处直接写了文字 → 当作唯一一条用户意见，跳过②
                user_notes = [act]
            else:
                note = input(
                    "② 你的修改意见？（回车=无；会与评审问题一起回传）\n> "
                ).strip()
                user_notes = [note] if note else []

        human_notes.extend(user_notes)  # 计入累计意见，供下一轮评审/回传使用

        parts = []
        if problems:
            parts.append(f"{len(problems)} 条评审问题")
        if user_notes:
            parts.append(f"{len(user_notes)} 条用户意见")
        print(f"[agent] 回传『{' + '.join(parts)}』修正第 {version} 版…")

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
                    "content": build_revise_message(problems, user_notes),
                },
            ],
        )
        version += 1
