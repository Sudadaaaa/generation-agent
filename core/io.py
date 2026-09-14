"""人机交互 IO 抽象：工具的交互不直接碰 input()/print()，经由 IO 接口。

默认实现 ConsoleIO（终端）；离线测试用脚本化 FakeIO 预置问答序列。
将来接 http/服务端场景时新增一个 IO 实现即可（预留实现点）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IO(ABC):
    """工具与用户交互的最小接口。"""

    @abstractmethod
    def ask(self, prompt: str) -> str:
        """向用户提一个问题/给一个输入提示，返回其原样输入（不含提示前缀）。"""

    @abstractmethod
    def out(self, text: str = "", end: str = "\n") -> None:
        """输出文本。end 默认换行；流式输出传 end=""。"""


class ConsoleIO(IO):
    """终端实现：ask→input()，out→print()。"""

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def out(self, text: str = "", end: str = "\n") -> None:
        print(text, end=end, flush=True)
