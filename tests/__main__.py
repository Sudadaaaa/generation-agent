"""零依赖测试运行器：python -m tests

环境里没有 pytest，就在 stdlib 上写个最小 runner：
- 收集本目录（tests 包）下 test_*.py；
- 逐个执行名为 test_* 的函数（模块/文件有 import 错误时按收集失败算）；
- 打印 PASS/FAIL（失败带 traceback），全部通过退出码 0，有失败退出码 1。

用法（generation-agent/ 下）：
    /DATASSD2/.../anaconda3/envs/llm/bin/python -m tests
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

_DIR = pathlib.Path(__file__).resolve().parent


def _run_module(path: pathlib.Path) -> list[tuple[str, str | None]]:
    """跑一个测试文件：返回 [(test 名, None=通过 | traceback 文本)]。"""
    module_name = f"tests.{path.stem}"
    results: list[tuple[str, str | None]] = []

    try:
        module = importlib.import_module(module_name)
    except Exception:
        results.append((path.name, traceback.format_exc()))
        return results

    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        func = getattr(module, name)
        if not callable(func):
            continue
        try:
            func()
            results.append((f"{path.stem}::{name}", None))
        except Exception:
            results.append((f"{path.stem}::{name}", traceback.format_exc()))

    return results


def main() -> int:
    files = sorted(_DIR.glob("test_*.py"))
    if not files:
        print("[tests] 没有找到 test_*.py")
        return 0

    failures: list[tuple[str, str]] = []
    passed = 0

    for path in files:
        for name, tb in _run_module(path):
            if tb is None:
                passed += 1
                print(f"PASS  {name}")
            else:
                failures.append((name, tb))
                print(f"FAIL  {name}")

    total = passed + len(failures)
    print(f"\n===== {passed}/{total} 通过 =====")
    if failures:
        for name, tb in failures:
            print(f"\n----- {name} -----")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
