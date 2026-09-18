"""critic 子 agent：一条需求 + 一份 GenerationPlan → 问题清单（JSON）。

由主 agent 经 tools/critic.py 的 CriticTool 调用，**不是 PlanAgent 的自检**：
要不要审、审完怎么办，都由主 agent 判断——critic 相当于「另一个人把计划看一遍」。

history 语义与 PlanAgent **刻意相反**：

    PlanAgent   全量保留——「在刚才基础上加 XX」要靠它
    CriticAgent 每次重装——评审只看「需求 + 这一份计划」

后者挡的是一个具体偏差：主 agent 的 history 里有整场对话，包括计划是怎么被要求改的、
上一轮评审说过什么。评审若看得见这些，就会被计划的自我辩解和它自己上次的结论说服
而放行。一轮对话里可能审多次（改完再审），实例是复用的，不重装还会把上一版的评语
带进来，越评越松——而这个偏差没有任何机制会报出来。

「调不通」类的失败不重试：空内容、被截断，一次调用后直接抛，不占任何预算——
重发同一份请求结果一样（见 _complete）。但**字段不合输出契约会回传修正**，
与 PlanAgent 同款、同样的 max_repairs：json_object 只保证「是合法 JSON」，
不保证任何一个字段存在，而「问题数量」正是调用方据以决定要不要再改一轮的信号，
缺了这次评审就白做。从前不校验，是因为没有消费方按字段算指标——评估就是那个消费方。

「评审挂了」和「评审说没问题」是两回事，而后者正是主 agent 会据以放行的东西。
旧 domain/critique.py 把 fail-open 写死在内部，那正是它没法复用到评估的原因——
评估里「裁判没跑成」会被记成「没问题」，直接污染指标。策略留给调用方：
主 agent 侧可以吞掉异常当通过（不能因为评审服务挂了就卡住用户），
评估侧则必须记 error 并排除该条。

输出契约（评分 / 意见 / 问题数量）写在 core/prompt.py 的 CRITIC_BASE_SYSTEM_PROMPT 里，
本模块的 parse_critique 是它唯一的实现——评估侧 import 它，不另抄一份。
"""

# 与 plan_agent 的同名常量字面相同，但**不 import 它**：两处关心的是各自的输出
# 契约（plan 是 GenerationPlan，critic 是评分/意见/问题数量），只是恰好都用
# json_object 保证「是合法 JSON」。让 critic 依赖 plan_agent 反而把两者拴在一起。
RESPONSE_FORMAT = {"type": "json_object"}

_REQUIRED = ("评分", "意见", "问题数量")


class CriticFormatError(RuntimeError):
    """critic 这次的输出不合输出契约。带原因，用来生成给它的修正指令。"""


def _num(v):
    """→ float。掰不动就 None——不猜、不默认成 0。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_critique(text: str) -> dict:
    """critic 正文 → {score, n_issues, opinions, n_opinions}。

    契约是 core/prompt.py 里 CRITIC_BASE_SYSTEM_PROMPT 的「## 输出」三个字段。
    **json_object 只保证「是合法 JSON」，不保证字段存在**，所以必须逐字段验——
    不合契约抛 CriticFormatError，由 run() 回传给模型重发。

    **不校验取值范围**：评分超出 0-10 照记，N 条意见只标 1 个问题也照记——那是
    critic 的行为，抹掉就看不出它跑偏了。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise CriticFormatError(
            f"你输出的不是合法 JSON（{e}）。只输出一个 JSON object，不要任何解释文字。"
        ) from e

    if not isinstance(data, dict):
        raise CriticFormatError(f"你输出的 JSON 不是 object（是 {type(data).__name__}）。")

    missing = [k for k in _REQUIRED if k not in data or data[k] is None]
    if missing:
        raise CriticFormatError(
            "缺少这些字段：" + "、".join(missing) + "。三个字段必须都给出来。"
        )

    score = _num(data["评分"])
    if score is None or not math.isfinite(score):
        raise CriticFormatError(f"「评分」要是 0 到 10 的数，你给的是 {data['评分']!r}。")

    n = _num(data["问题数量"])
    if n is None or not (math.isfinite(n) and n.is_integer()):
        raise CriticFormatError(f"「问题数量」要是整数，你给的是 {data['问题数量']!r}。")

    log = data["意见"]
    if not isinstance(log, list) or not all(isinstance(o, str) for o in log):
        raise CriticFormatError(f"「意见」要是字符串数组，你给的是 {log!r}。")

    return {"score": score, "n_issues": int(n), "opinions": log, "n_opinions": len(log)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math

from core.agent import BaseAgent
from core.config import AgentConfig
from core.exceptions import HelloAgentsException
from core.llm import AgentLLM
from core.message import Message
from core.prompt import (
    build_correction_message,
    build_critic_system_prompt,
    build_critique_request,
)


class CriticAgent(BaseAgent):
    """评一次计划，返回问题清单正文。上下文每次重装，理由见模块 docstring。"""

    def __init__(
        self,
        name: str,
        llm: AgentLLM,
        system_prompt: str | None = None,
        config: AgentConfig | None = None,
        max_repairs: int = 2,
        response_format: dict | None = RESPONSE_FORMAT,
    ) -> None:
        super().__init__(name, llm, system_prompt or build_critic_system_prompt(), config)
        self.max_repairs = max_repairs
        # 本次 run() 实际走了几轮修正（0 = 一次过）。只在末尾读，不参与控制流。
        self.repair_count = 0
        # 默认沿用 RESPONSE_FORMAT（vLLM 下会变成 xgrammar 逐 token 掩码）。
        # 显式传 None 可关掉——评估侧靠它量「约束解码值多少」，与 PlanAgent 同款开关。
        self.response_format = response_format

    def run(self, requirement: str, plan: str) -> str:
        """评一次。plan 传 GenerationPlan 的 JSON 文本（plan 工具的产出可直接传）。

        字段不合契约时把错误回传给模型重发，至多 max_repairs 次（理由见模块 docstring）。
        「调不通」类失败不重试，直接抛，由调用方决定当通过还是当失败。
        """
        self.repair_count = 0
        # 重装而非追加：clear_history 会把 system 重新压回首位。修正轮追加在本次
        # history 之后，跨 run 带不出去——下一次进来又重装。
        self.clear_history()
        self.add_message(Message("user", build_critique_request(requirement, plan)))

        last: Exception | None = None
        for _ in range(self.max_repairs + 1):
            content = self._complete()
            try:
                parse_critique(content)
            except CriticFormatError as e:
                last = e
                self.repair_count += 1
                self.add_message(Message("assistant", content))
                self.add_message(Message("user", build_correction_message(e)))
                continue

            return content

        raise HelloAgentsException(
            f"评审失败（{self.max_repairs + 1} 次尝试）：{last}\n"
            "可以让 plan 换一版措辞重出，或先跳过这次评审。"
        )

    def _complete(self) -> str:
        """一次调用，返回正文。

        两种「重试也没用」的失败在这里直接抛，不占修正预算——判断与 PlanAgent._complete
        同款，报错文案各自一份：两处说的是各自模型的失败，混成一句会看不出是谁挂的。
        """
        # 关约束时不能传 response_format=None——SDK 会把它序列化成 null 发出去，
        # 只能整个省掉这个键。（同 PlanAgent._complete）
        fmt = {"response_format": self.response_format} if self.response_format else {}
        response = self.llm.invoke(self.get_messages(), **fmt)
        choice = response.choices[0]
        content = (choice.message.content or "").strip()

        if not content:
            raise HelloAgentsException(
                f"评审模型返回了空内容（finish_reason={choice.finish_reason}，"
                f"usage={response.usage}）。"
                "推理模型的思考与正文共用一个输出预算，重发同一份请求没有意义。"
            )
        if choice.finish_reason == "length":
            raise HelloAgentsException(
                f"评审输出被 max_tokens 截断（usage={response.usage}）。"
                "截断的评语是一份**不完整**的问题清单，而调用方看不出它少了结尾，"
                "会照着残缺的清单去改。请调高 max_tokens。"
            )
        return content


if __name__ == "__main__":
    """本层自测入口：不经过 main.py / CriticTool，直接验 critic 的产出。

    用 `python -m agent.critic_agent`（在仓库根跑）或 `python agent/critic_agent.py`。
    输入一条需求，会先让 PlanAgent 拆一版，再让 critic 评它——
    重点看两件事：评语是否具体到能照着改；「无问题」是不是真的只在没问题时出现。
    """
    from core.io import ConsoleIO
    from agent.plan_agent import PlanAgent

    _EXIT_WORDS = {"exit", "quit", "退出"}

    io = ConsoleIO()
    cfg = AgentConfig()
    planner = PlanAgent(name="计划", llm=AgentLLM(**cfg.plan_llm_kwargs))
    critic = CriticAgent(name="评审", llm=AgentLLM(**cfg.critic_llm_kwargs))

    io.out("=" * 60)
    io.out(f"计划模型：{planner.llm.model}")
    io.out(f"评审模型：{critic.llm.model}")
    io.out("=" * 60)

    while True:
        request = io.ask("> ").strip()
        if not request:
            continue
        if request.lower() in _EXIT_WORDS:
            break

        try:
            plan = planner.run(request)
        except HelloAgentsException as e:
            io.out(f"\n⚠️ 计划生成失败：{e}")
            continue

        io.out("\n── 计划 ──")
        io.out(plan)
        io.out("\n── 评审 ──")
        try:
            io.out(critic.run(request, plan))
        except HelloAgentsException as e:
            io.out(f"\n⚠️ {e}")
