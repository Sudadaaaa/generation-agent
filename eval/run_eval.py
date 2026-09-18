"""plan 生成的评估：plan ↔ critic 来回改到 critic 满意，测**当前 .env 配置的 plan 模型**。

在仓库根跑，用 llm 环境：
    <env>/bin/python eval/run_eval.py --smoke     # 2 条，验通路
    <env>/bin/python eval/run_eval.py             # 全量 62 条

一条样例：需求 → PlanAgent → plan → CriticAgent 点评 → 有问题就把「意见」回给
PlanAgent 重出。**至多 3 份 plan、3 次评审**（= 最多改 2 轮），critic 说没问题就提前停。
两种停止不区分——最终得分和总问题数已经说明了改没改到位。

带意见回去重出靠 PlanAgent 保留全量 history（它记得原需求和上一版计划，所以传进去的
只是一句「按意见改」）；CriticAgent 反过来每轮重装，只看得到「需求 + 这一版计划」，
不会被自己上一轮的结论说服着放行。

四个指标，每条各一个值：

    score         最后一轮 critic 的评分      改到最后交出来的有多好（0-10）
    issues_total  各轮「问题数量」之和        一路上被指出多少问题（**不去重**，量的是折腾量）
    revisions     实际改了几轮（0-2）         0 = 初版就过
    final_len     最终那份 plan 的字符数      不用 token：思考 token 混在 completion_tokens 里，跨模型不可比

均值的分母是 **scored**（评出分来的条数），不是 n。失败不记 0 分也不记 0 轮——
那会让「计划常常生不出来」的模型看起来只是「分低一点、改得少一点」，等于奖励失败。
所以 `scored/n` 和四个均值打在同一屏上。

**这些数有 run-to-run 抖动**：plan 侧的思考是采样的，critic 自己也是推理模型，它的评分
本身在抖。实测同一条 `id=1869[en]` 两次跑评分 8.5 → 7.5、输出长度差 30%。n=62 取均值
能压掉一部分，但**臂间 0.3 分以内的差不要当差异读**。

换模型 = 改 `.env` 的 `PLAN_*`。critic 由 `CRITIC_*` 决定（留空回退 `LLM_*`），
**不跟着 `PLAN_*` 动**——测本地臂时它就是那个更强的云端裁判，三臂共用同一个 critic，
所以臂间差值有效；而测 DeepSeek 臂时 plan 与 critic 是同一个模型，那个臂是自己评自己。
结果里的 `model` 和 `critic_model` 都落了盘，比分数前先对这两项。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.critic_agent import CriticAgent, CriticFormatError, parse_critique
from agent.plan_agent import RESPONSE_FORMAT, PlanAgent
from core.config import AgentConfig
from core.llm import AgentLLM
from core.prompt import build_critic_system_prompt, build_plan_system_prompt
from mydatasets.plan_sft import VAL_FRAC, plan_sft_rows

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# outputs/ 已在 .gitignore；结果里含每条 plan 正文，是运行产物不是源码。
RESULTS_DIR = os.path.join(_REPO, "outputs", "eval")

# 一条样例最多出几份 plan / 评几次。有界是因为「直到 critic 满意」本身不保证收敛——
# critic 挑刺成性、或每轮都能挑出新毛病时，它会一直有话说。
MAX_ROUNDS = 3


def read_heldout(limit=None):
    """val 段 → [{id, language, user}]。

    只取 user，不带语料里那份参考 plan：它会把评审往参考答案上拽，而且它本身也是
    模型生成的，不是金标准。留出集现场从 plan_sft_rows() 算，不读副本文件——
    切点与训练共用一处，改了 VAL_FRAC 两边同时生效，没有「副本没重导」这种静默漂移。
    """
    items = [
        {
            "id": r["id"],
            "language": r["language"],
            "user": next(m["content"] for m in r["messages"] if m["role"] == "user"),
        }
        for r in plan_sft_rows(split="val")
    ]
    return items[:limit] if limit else items


def _revision_message(opinions):
    """评审意见 → 一条要求修改计划的消息。

    和 agent 的 build_correction_message 同类但不同事：那条管**格式**（上次输出解析不了），
    这条管**内容**（计划合 schema，只是拆得不对）。两者可能先后进同一段 history，
    措辞得区分开。**只放 eval 里**——主 agent 走工具调用，拿到的直接就是 critic 的返回
    值，它自己会读，不需要这个拼装函数。

    只喂意见列表，不喂 critic 的原始 JSON：评分（比如 7.5）混进去等于提前给「这次改得
    怎么样」定调，模型会照着分猜要改多少，而不是照着意见改。
    """
    body = "\n".join(f"- {o}" for o in opinions)
    return f"""评审者对当前计划提出了以下意见：

{body}

请按这些意见修改计划，重新输出【完整】的 GenerationPlan JSON。

要求：
- 只改意见指出的地方，没有被提到的内容逐字保持不变。
- 重新输出【完整】的 JSON object，不要只输出改动的那一小段。
- 只能返回 JSON，不要输出解释，不要 Markdown 代码围栏。
"""


def run_sample(planner, critic, requirement):
    """一条样例走完 plan ↔ critic 的循环，返回四个指标 + 逐轮明细。

    planner 是**这条样例专用的**（调用方每条新建）。PlanAgent 保留全量 history，
    复用会把上一条的需求和计划带进来——那是在测多轮对话，不是测单条。
    critic 可以全局复用：它每次重装 history，实例上没有跨条状态。
    """
    rec = {"revisions": 0, "issues_total": 0, "rounds": [], "plan": None,
           "plan_ok": False, "score": None, "final_len": None, "error": None}
    prompt = requirement

    for r in range(MAX_ROUNDS):
        try:
            rec["plan"] = planner.run(prompt)     # 失败时 rec["plan"] 保持上一份，不会被清掉
        except Exception as e:
            rec["error"] = f"第 {r + 1} 份 plan 失败：{type(e).__name__}: {e}"
            break
        rec["plan_ok"] = True
        rec["revisions"] = r                      # 出成了第 r+1 份 ⇒ 改过 r 轮

        t0 = time.time()
        try:
            raw = critic.run(requirement, rec["plan"])
        except Exception as e:
            rec["error"] = f"第 {r + 1} 次评审失败：{type(e).__name__}: {e}"
            break

        try:
            crit = parse_critique(raw)
        except CriticFormatError as e:
            # run() 内部已经回传修正过仍不合契约。评不成 ≠ 没问题，不能当通过。
            rec["error"] = f"第 {r + 1} 次评审字段不合契约：{e}"
            break
        crit["latency_s"] = round(time.time() - t0, 2)
        rec["rounds"].append(crit)
        rec["issues_total"] += crit["n_issues"]

        if crit["n_issues"] == 0:                 # critic 满意
            break
        if not crit["opinions"]:                  # 报了问题却给不出意见，没有可依据的要求
            rec["error"] = "critic 报了问题但「意见」为空，无法据此修改"
            break

        # 把意见回给 plan。用完 MAX_ROUNDS 次时这里照常赋值，只是循环也到头了。
        prompt = _revision_message(crit["opinions"])

    if rec["plan"] is not None:
        rec["final_len"] = len(rec["plan"])       # 字符数，不是 token，理由见模块 docstring
    if rec["rounds"]:
        rec["score"] = rec["rounds"][-1].get("score")
    return rec


def preflight(llm, model, base_url, role):
    """开跑前确认连得上，连不上直接退出。

    不探的话，连不上会在每条样例上重演（62 条 × 最多 6 次调用），最后拿到一份全是 error
    的结果，分不清是环境问题还是代码问题。模型名不在服务列表里**只警告**：云端 /models
    不一定列全，硬拦会挡住本来正常的运行。
    """
    try:
        available = sorted(m.id for m in llm.client.models.list().data)
    except Exception as e:
        hint = ("先起服务：bash scripts/serve_plan_model.sh" if role == "plan"
                else "检查 .env 的 CRITIC_* / LLM_* 与网络")
        raise SystemExit(f"✗ {role} 连不上 {base_url}\n  {type(e).__name__}: {e}\n  {hint}") from e

    if model not in available:
        print(f"  ⚠️ [{role}] {model!r} 不在 {base_url} 的服务列表里 {available}")
        print("     （云端 /models 可能不列全，先按能跑处理；若随后全是 404 就回来查这里）")


def summarize(records):
    """逐条 → 汇总。计数类以 n 为分母（不给失败开后门），均分类以 scored 为分母。"""
    n = len(records)
    if not n:
        return {}
    scored = [r for r in records if r.get("score") is not None]
    mean = lambda xs, nd=3: round(sum(xs) / len(xs), nd) if xs else None
    return {
        "n": n,
        "plan_ok": sum(1 for r in records if r.get("plan_ok")),
        "scored": len(scored),                    # 下面四个均值的分母，必须同屏
        "mean_score": mean([r["score"] for r in scored]),
        "mean_issues_total": mean([r["issues_total"] for r in scored], 2),
        "mean_revisions": mean([r["revisions"] for r in scored], 2),
        "mean_final_len": mean([r["final_len"] for r in scored], 0),
    }


def print_summary(s):
    n = s["n"]
    print(f"    n                 {n}")
    print(f"    plan_ok           {s['plan_ok']}/{n} ({s['plan_ok'] / n:.1%})    至少出过一份 plan")
    print(f"    scored            {s['scored']}/{n} ({s['scored'] / n:.1%})    ← 下面四个均值的分母")
    print(f"    mean_score        {s['mean_score']}     0-10，越高越好")
    print(f"    mean_issues_total {s['mean_issues_total']}     各轮问题数之和，越低越好")
    print(f"    mean_revisions    {s['mean_revisions']}     改了几轮，0 = 初版就过")
    print(f"    mean_final_len    {s['mean_final_len']}     最终 plan 的字符数")


def main():
    p = argparse.ArgumentParser(description="plan 生成评估（测当前 .env 配置的 plan 模型）")
    p.add_argument("--limit", type=int, default=None, help="只用前 N 条")
    p.add_argument("--smoke", action="store_true", help="等价于 --limit 2，验通路")
    p.add_argument("--no-constraint", action="store_true",
                   help="plan 侧不下发 response_format，关掉约束解码（critic 侧不受影响）")
    args = p.parse_args()
    constrained = not args.no_constraint

    cfg = AgentConfig()
    plan_kw, critic_kw = cfg.plan_llm_kwargs, cfg.critic_llm_kwargs
    critic = CriticAgent(name="评审", llm=AgentLLM(**critic_kw))
    items = read_heldout(2 if args.smoke else args.limit)
    plan_sha = hashlib.sha1(build_plan_system_prompt().encode()).hexdigest()[:8]
    critic_sha = hashlib.sha1(build_critic_system_prompt().encode()).hexdigest()[:8]

    print(f"被测 plan   {plan_kw['model']}  @ {plan_kw['base_url']}")
    print(f"评分 critic {critic_kw['model']}  @ {critic_kw['base_url']}")
    print(f"留出集 val（VAL_FRAC={VAL_FRAC}）{len(items)} 条，"
          f"每条约 3 份 plan + 3 次评审（上限，critic 满意即停）")
    print(f"提示词 sha1 plan={plan_sha}  critic={critic_sha}")
    print(f"plan 侧输出约束 {'开（response_format=json_object）' if constrained else '关（不下发 response_format）'}")
    preflight(AgentLLM(**plan_kw), plan_kw["model"], plan_kw["base_url"], "plan")
    preflight(critic.llm, critic_kw["model"], critic_kw["base_url"], "critic")
    print()

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 临时账本：逐条 append，跑成功就删。一轮跑几小时，中途断了不至于一分钟不剩。
    journal = os.path.join(RESULTS_DIR, f"{stamp}.partial.jsonl")
    print(f"中断保护：逐条写入 {os.path.basename(journal)}（跑完自动删除）\n")

    records = []
    for i, item in enumerate(items, 1):
        planner = PlanAgent(name="评估", llm=AgentLLM(**plan_kw),
                            response_format=RESPONSE_FORMAT if constrained else None)
        rec = {"id": item["id"], "language": item["language"]}
        rec.update(run_sample(planner, critic, item["user"]))
        records.append(rec)

        with open(journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  [{i}/{len(items)}] id={rec['id']:<5} [{rec['language']}] "
              f"改={rec['revisions']} 问题={rec['issues_total']:<3} "
              f"分={rec['score']} 长度={rec['final_len']}"
              + (f"  ⚠️ {rec['error']}" if rec["error"] else ""))

    s = summarize(records)
    print()
    print_summary(s)
    print()

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # model / critic_model 是这份结果的身份，不记就没人知道它测的是谁、跟谁比。
        "model": plan_kw["model"],
        "base_url": plan_kw["base_url"],
        "critic_model": critic_kw["model"],
        "critic_base_url": critic_kw["base_url"],
        "prompt_sha1": plan_sha,
        "critic_prompt_sha1": critic_sha,
        "constrained": constrained,    # 同模型同 prompt，约束开关的数字不可比
        "max_rounds": MAX_ROUNDS,
        "heldout": {"source": "mydatasets/plan_sft.jsonl", "split": "val",
                    "val_frac": VAL_FRAC, "n": len(items)},
        "summary": s,
        "records": records,
    }
    out = os.path.join(RESULTS_DIR, f"{stamp}.json")   # 文件名只放时间戳，身份在 json 里
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.remove(journal)             # 正式产物落了盘，临时账本没有留着的理由
    print(f"逐条结果已写入 {out}")


if __name__ == "__main__":
    main()
