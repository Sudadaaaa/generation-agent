"""plan 生成的评估：跑在 plan_sft.jsonl 的留出段上，测**当前 .env 配置的那个模型**。

在仓库根跑，用 llm 环境：
    <env>/bin/python eval/run_eval.py --smoke     # 2 条，验通路
    <env>/bin/python eval/run_eval.py             # 全量 62 条

## 换模型 = 改 .env

脚本没有「选模型」的参数，一次只测一个：`PLAN_*`（留空则回退 `LLM_*`，见 core/config.py）。
要测别的就先改 `.env` 的 `PLAN_MODEL_ID` / `PLAN_BASE_URL` / `PLAN_API_KEY`，再跑一次。
想比较两个模型就跑两次，结果各自落盘。

正因为模型由 `.env` 决定、而 `.env` 会变，**结果里必须记下本次实际连的 model 与 base_url**，
否则过两天打开一份 json 根本不知道它测的是谁。

## 留出集

= `mydatasets/plan_sft.jsonl` 的 val 段（VAL_FRAC 尾切，见 mydatasets/plan_sft.py），
**现场算，不读副本文件**。切点只有 `plan_sft_rows()` 一处、训练与评估共用，
所以改了 VAL_FRAC 两边同时生效，不存在「文件没重导」这种不报错的漂移。

## 两层，分开记

Layer 1 — 单次调用，**绕过** PlanAgent.run() 的修正循环。
  修正循环会掩盖模型的真实水平（一次没成、第二次成了，仍算成功），
  也会污染延迟与 token 统计——那是若干次调用的和。
Layer 2 — 走真实 PlanAgent，测端到端能否拿到合法计划、花了几轮修正。
  `repairs` 是「微调到底有没有用」的核心指标：0 = 一次过。

两层都跑，但各记各的：哪一层的结论都不靠另一层兜。
"""

import json
import os
import re
import sys
import time
import hashlib
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.planagent import PlanAgent, RESPONSE_FORMAT, parse_generation_plan
from core.config import AgentConfig
from core.llm import AgentLLM
from core.prompt import build_plan_system_prompt
from mydatasets.plan_sft import VAL_FRAC, plan_sft_rows

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 结果里含每条 plan 正文，是运行产物不是源码；outputs/ 已在 .gitignore，放这儿不必再动 .gitignore。
RESULTS_DIR = os.path.join(_BASE_DIR, "outputs", "eval")

# 解码参数：落盘，因为跨次运行必须一致才可比。temperature=0 是为了可复现——生产当前用的是
# 服务商默认值（PlanAgent._complete 没传 temperature），所以**绝对数字不代表线上**，
# 两次运行之间的**差值**才代表。
DECODE = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192}

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


# ────────────────────────── 留出集 ──────────────────────────

def read_heldout(limit=None):
    """val 段 → 每条 {id, language, user}。

    只取 user，**不带 assistant 的 plan**：参考 plan 会把后续 judge 往参考答案上拽，
    而且那份 plan 本身是模型生成的，不是金标准。
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


# ────────────────────────── 单条测量 ──────────────────────────

def _lang_of(text):
    """粗判一段文字是中文还是英文：按中日韩字符与拉丁字母的多少。"""
    cjk, latin = len(_CJK.findall(text)), len(_LATIN.findall(text))
    if cjk + latin == 0:
        return None
    return "zh" if cjk >= latin else "en"


def measure_layer1(llm, requirement, constrained=True):
    """单次调用 → 一条原始记录。**不做任何重试**，失败也如实记。

    constrained=False 时不下发 response_format。这是 json_ok 与 schema_ok 唯一能分开的
    配置：不约束时前者量「模型自己会不会老实输出纯 JSON」（裹围栏、加前言的都会挂），
    后者量「剥掉围栏前言后是否符合 schema」（parse_generation_plan 会剥围栏、会兜底取括号）。
    约束开着时 xgrammar 在解码期就把这两类失败掐掉了，两个指标一起恒真。
    """
    messages = [
        {"role": "system", "content": build_plan_system_prompt()},
        {"role": "user", "content": requirement},
    ]
    fmt = {"response_format": RESPONSE_FORMAT} if constrained else {}

    t0 = time.time()
    try:
        resp = llm.invoke(messages, **fmt, **DECODE)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}

    latency = round(time.time() - t0, 2)
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    usage = getattr(resp, "usage", None)

    rec = {
        "latency_s": latency,
        "finish_reason": choice.finish_reason,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "content_len": len(content),
    }

    # json_ok 用严格 json.loads：模型裹了 Markdown 围栏时它会挂，而 schema_ok 会过
    # （parse_generation_plan 会剥围栏）。这一层差正好量出「模型不老实输出纯 JSON」的频率。
    try:
        json.loads(content)
        rec["json_ok"] = True
    except json.JSONDecodeError:
        rec["json_ok"] = False

    try:
        plan = parse_generation_plan(content)
        rec["schema_ok"] = True
        rec["elements"] = len(plan.elements)
        text = "".join(
            (e.identity or "") + (e.appearance or "") + (e.layout or "") + (e.action or "")
            for e in plan.elements
        ) + (plan.overall or "")
        rec["plan_lang"] = _lang_of(text)
    except Exception as e:
        rec["schema_ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"

    return rec


def measure_layer2(llm, requirement, constrained=True):
    """走真实 PlanAgent（含修正循环）→ 最终成败 + 修正轮数。

    每条新建一个 agent：复用同一个会把上一轮的 history 带进来，那是在测多轮对话，
    不是测单条需求。
    """
    agent = PlanAgent(name="评估", llm=llm,
                      response_format=RESPONSE_FORMAT if constrained else None)
    t0 = time.time()
    try:
        agent.run(requirement)
        ok, err = True, None
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    return {
        "final_ok": ok,
        "repairs": agent.repair_count,
        "latency_s": round(time.time() - t0, 2),
        **({"error": err} if err else {}),
    }


# ────────────────────────── 探活 ──────────────────────────

def preflight(llm, model, base_url):
    """开跑前确认连得上。

    不做这一步的话，连不上会在每一条上重演（62 条 × 2 层 = 124 次同样的报错），
    最后拿到一份全是 error 的结果，分不清是环境问题还是代码问题。

    连不上 = 硬失败（本地臂最常见的原因就是 vLLM 没起）。模型名不在服务列表里只警告：
    云端 /models 不一定列全，硬拦会挡住本来正常的运行。
    """
    try:
        available = sorted(m.id for m in llm.client.models.list().data)
    except Exception as e:
        raise SystemExit(
            f"✗ 连不上 {base_url}\n"
            f"  {type(e).__name__}: {e}\n"
            "  若测的是本地模型，先起服务：bash scripts/serve_plan_model.sh"
        ) from e

    if model not in available:
        print(f"  ⚠️ {model!r} 不在 {base_url} 的服务列表里 {available}")
        print("     （云端 /models 可能不列全，先按能跑处理；若随后全是 404 就回来查这里）")
    return available


# ────────────────────────── 汇总 ──────────────────────────

_COUNT_KEYS = ("json_ok", "schema_ok", "language_match", "final_ok", "once_through")


def summarize(records):
    """逐条记录 → 汇总。

    分母一律是**尝试过的条数 n**，含报错的：不给失败开后门，否则失败越多分越高。
    """
    n = len(records)
    if not n:
        return {}
    l1 = [r["layer1"] for r in records]
    l2 = [r["layer2"] for r in records]

    counts = {
        "json_ok": sum(1 for x in l1 if x.get("json_ok")),
        "schema_ok": sum(1 for x in l1 if x.get("schema_ok")),
        "language_match": sum(1 for r in records
                              if r["layer1"].get("plan_lang") == r["language"]),
        "final_ok": sum(1 for x in l2 if x["final_ok"]),
        "once_through": sum(1 for x in l2 if x["repairs"] == 0),
    }

    lat = [x["latency_s"] for x in l1 if "latency_s" in x]
    tok = [x["completion_tokens"] for x in l1 if x.get("completion_tokens")]
    # 回答长度用字符数，不用 completion_tokens：推理模型的 completion_tokens 含看不见的
    # 思考 token（deepseek-v4-flash 实测 27 个里有 21 个是思考），跨模型不可比。
    cl = [x["content_len"] for x in l1 if x.get("content_len") is not None]
    return {
        "n": n,
        **counts,
        "mean_repairs": round(sum(x["repairs"] for x in l2) / n, 3),
        "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
        "mean_completion_tokens": round(sum(tok) / len(tok)) if tok else None,
        "mean_content_len": round(sum(cl) / len(cl)) if cl else None,
    }


def print_summary(s):
    n = s["n"]
    print(f"    n                        {n}")
    for k in _COUNT_KEYS:
        print(f"    {k:<24} {s[k]}/{n} ({s[k] / n:.1%})")
    for k in ("mean_repairs", "mean_latency_s", "mean_completion_tokens", "mean_content_len"):
        print(f"    {k:<24} {s[k]}")


# ────────────────────────── 主流程 ──────────────────────────

def main():
    p = argparse.ArgumentParser(description="plan 生成评估（测当前 .env 配置的模型）")
    p.add_argument("--limit", type=int, default=None, help="只用前 N 条")
    p.add_argument("--smoke", action="store_true", help="等价于 --limit 2，验通路")
    p.add_argument("--no-constraint", action="store_true",
                   help="不下发 response_format，关掉 vLLM 的 xgrammar 约束解码（两层都关）")
    args = p.parse_args()
    constrained = not args.no_constraint

    kw = AgentConfig().plan_llm_kwargs
    model, base_url = kw["model"], kw["base_url"]
    llm = AgentLLM(**kw)

    items = read_heldout(2 if args.smoke else args.limit)
    prompt_sha = hashlib.sha1(build_plan_system_prompt().encode()).hexdigest()[:8]

    print(f"模型   {model}  @ {base_url}")
    print(f"留出集 plan_sft.jsonl 的 val 段（VAL_FRAC={VAL_FRAC}）  {len(items)} 条")
    print(f"提示词 sha1 {prompt_sha}   decode {DECODE}")
    # 约束开没开必须亮相：同一个模型同一条需求，这两种配置的数字不可比。
    print(f"输出约束 {'开（response_format=json_object）' if constrained else '关（不下发 response_format）'}")
    preflight(llm, model, base_url)
    print()

    records = []
    for i, item in enumerate(items, 1):
        records.append({
            "id": item["id"],
            "language": item["language"],
            "layer1": measure_layer1(llm, item["user"], constrained),
            "layer2": measure_layer2(llm, item["user"], constrained),
        })
        r = records[-1]
        print(f"  [{i}/{len(items)}] id={r['id']:<5} json={r['layer1'].get('json_ok')} "
              f"schema={r['layer1'].get('schema_ok')} repairs={r['layer2']['repairs']} "
              f"{r['layer1'].get('latency_s')}s")

    s = summarize(records)
    print()
    print_summary(s)
    print()

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # model/base_url 是事实、是这份结果的全部身份；不记就没人知道它测的是谁。
        "model": model,
        "base_url": base_url,
        "prompt_sha1": prompt_sha,
        "decode": DECODE,
        # 同样必须落盘：同一模型、同一 prompt，约束开与关的数字不可比。
        "constrained": constrained,
        "heldout": {
            "source": "mydatasets/plan_sft.jsonl",
            "split": "val",
            "val_frac": VAL_FRAC,
            "n": len(items),
        },
        "summary": s,
        "records": records,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 文件名只放时间戳：model 名可能含 `/` 之类的字符，塞进文件名会出乱子。身份在 json 里。
    out = os.path.join(RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"逐条结果已写入 {out}")


if __name__ == "__main__":
    main()
