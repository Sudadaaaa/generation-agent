"""agentkit 底座离线回归：Tool 契约 / 注册表 / usage / 门控分发。"""

from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

from agentkit.tool import Tool, ToolRegistry
from agentkit.usage import Usage

from tests.helpers import FakeIO


class _AddArgs(BaseModel):
    x: int = 0
    y: int = 1


class _ReqArgs(BaseModel):
    a: int  # 必填


def _make_tool(gate=None):
    def _run(ctx, args: _AddArgs) -> str:
        return str(args.x + args.y)

    return Tool(
        name="add",
        description="把两个整数相加",
        args_schema=_AddArgs,
        run=_run,
        gate=gate,
    )


# ---- Tool 契约 -----------------------------------------------------------
def test_tool_openai_schema_is_mcp_isomorphic():
    tool = _make_tool()
    schema = tool.openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add" and fn["description"] == "把两个整数相加"
    # parameters 直接来自 args_schema.model_json_schema()（即 MCP inputSchema 形态）
    assert fn["parameters"]["properties"]["x"] == {"default": 0, "title": "X", "type": "integer"}


def test_tool_extra_args_ignored_missing_required_rejected():
    tool = _make_tool()
    assert tool.args_schema(x=2, y=3, surprise=99).x == 2  # 额外字段忽略
    try:
        _ReqArgs()
        raise AssertionError("缺必填字段应抛 ValidationError")
    except ValidationError:
        pass


def test_tool_gate_blocks_before_run():
    ctx = SimpleNamespace()

    def gate(c, args):
        return "被门控拦截"

    calls = {"n": 0}

    def run(c, args):
        calls["n"] += 1
        return "不该执行"

    tool = Tool(name="g", description="", args_schema=_AddArgs, run=run, gate=gate)
    obs = tool.gate_or_run(ctx, _AddArgs(x=1))
    assert obs == "被门控拦截" and calls["n"] == 0


def test_tool_gate_none_runs():
    tool = _make_tool()
    assert tool.gate_or_run(SimpleNamespace(), _AddArgs(x=2, y=3)) == "5"


# ---- ToolRegistry --------------------------------------------------------
def test_registry_order_and_lookup():
    reg = ToolRegistry([_make_tool()])
    assert reg.names() == ["add"]
    assert reg.get("add") is not None and reg.get("nope") is None


def test_registry_rejects_duplicate():
    reg = ToolRegistry()
    reg.add(_make_tool())
    try:
        reg.add(_make_tool())
        raise AssertionError("重名工具应抛 ValueError")
    except ValueError:
        pass


# ---- Usage ---------------------------------------------------------------
def test_usage_accumulates_and_describes():
    u = Usage()
    u.add_tokens(10, 5, 1)
    u.add_tokens(2, 1, 1)
    assert u.calls == 2 and u.total_tokens == 18
    assert u.describe().startswith("2 次调用")

    other = Usage()
    other.add_tokens(1, 1, 1)
    u.merge(other)
    assert u.calls == 3 and u.total_tokens == 20
