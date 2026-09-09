"""llm 传输层离线回归：DeepSeekChat function calling 解析 / 空内容 / 异常映射；
make_chat_client 构造；config 读取。全部脚本化，不触网。"""

from types import SimpleNamespace

from config import AgentConfig
from llm.base import UsageStat
from llm.deepseek import DeepSeekChat
from llm.factory import make_chat_client
from llm.local_qwen import LocalQwenChat


def _fake_tool_calls():
    return [
        SimpleNamespace(id="c1", function=SimpleNamespace(name="make_plan", arguments='{"requirement":"猫"}')),
        SimpleNamespace(id="c2", function=SimpleNamespace(name="render", arguments="{}")),
    ]


def _resp(*, content, usage=None, tool_calls=None, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish)],
        usage=usage or SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


class _FakeOpenAI:
    """记录请求、按脚本返回/抛错的假 OpenAI 客户端。

    deepseek.py 走 client.chat.completions.create(...)（OpenAI 兼容调用链），
    所以这里也要模拟出 .chat.completions.create 这一层。
    """

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.kwargs: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


def _deepseek(responses=None, error=None) -> tuple[DeepSeekChat, _FakeOpenAI]:
    client = DeepSeekChat(model="deepseek-chat", api_key="sk-test")
    fake = _FakeOpenAI(responses=responses, error=error)
    client._client = fake
    return client, fake


# ---- DeepSeekChat.complete_tools ----------------------------------------
def test_complete_tools_parses_calls():
    client, fake = _deepseek(responses=[_resp(content=None, tool_calls=_fake_tool_calls())])
    text, calls = client.complete_tools([{"role": "user", "content": "hi"}], tools=[{}])
    assert text is None
    assert [c.name for c in calls] == ["make_plan", "render"]
    assert calls[0].arguments == {"requirement": "猫"}
    assert calls[0].id == "c1"
    # tools 描述被透传给 API
    assert fake.kwargs[0]["tools"] == [{}]
    assert fake.kwargs[0]["tool_choice"] == "auto"
    assert client.usage.calls == 1 and client.usage.prompt_tokens == 11


def test_complete_tools_content_and_calls():
    client, _ = _deepseek(responses=[_resp(content="我先规划", tool_calls=_fake_tool_calls())])
    text, calls = client.complete_tools([], tools=[])
    assert text == "我先规划" and len(calls) == 2


def test_complete_tools_empty_both_raises():
    client, _ = _deepseek(responses=[_resp(content=None, tool_calls=None)])
    try:
        client.complete_tools([], tools=[])
        raise AssertionError("空内容且无 tool_calls 应抛 RuntimeError")
    except RuntimeError:
        pass


# ---- DeepSeekChat.chat ---------------------------------------------------
def test_chat_returns_content_and_records_usage():
    client, fake = _deepseek(responses=[_resp(content="你好")])
    out = client.chat([{"role": "user", "content": "hi"}])
    assert out == "你好"
    assert fake.kwargs[0]["response_format"] == {"type": "json_object"}
    assert client.usage.calls == 1


def test_chat_free_text_omits_json_mode():
    client, fake = _deepseek(responses=[_resp(content="自由文本")])
    client.chat([], json_mode=False)
    assert "response_format" not in fake.kwargs[0]


def test_chat_empty_content_raises():
    client, _ = _deepseek(responses=[_resp(content=None)])
    try:
        client.chat([], json_mode=True)
        raise AssertionError("空内容应抛 RuntimeError")
    except RuntimeError:
        pass


def test_sdk_error_mapped_to_runtime_error():
    import openai

    client, _ = _deepseek(error=openai.OpenAIError("boom"))
    for fn in (
        lambda: client.chat([]),
        lambda: client.complete_tools([], tools=[]),
    ):
        try:
            fn()
            raise AssertionError("OpenAIError 应被映射为 RuntimeError")
        except RuntimeError as exc:
            assert "boom" in str(exc) or "DeepSeek" in str(exc)


# ---- UsageStat（会话退出汇总用 describe）---------------------------------
def test_usage_stat_accumulates_and_describes():
    u = UsageStat()
    u.add_values(prompt_tokens=100, completion_tokens=50)
    u.add_raw(None)  # 缺 usage 也计入一次调用
    assert u.calls == 2 and u.total_tokens == 150
    assert u.describe() == "2 次调用 / 150 tokens（入 100 + 出 50）"


# ---- make_chat_client ----------------------------------------------------
def test_make_chat_client_deepseek():
    c = make_chat_client("deepseek", model="deepseek-chat")
    assert isinstance(c, DeepSeekChat) and c.model == "deepseek-chat"


def test_make_chat_client_qwen_lazy():
    c = make_chat_client("qwen", model="Qwen/Qwen3-8B")
    assert isinstance(c, LocalQwenChat)
    # 未触发 _load（不占显存）：私有状态仍为空
    assert c._model is None


def test_make_chat_client_unknown_raises():
    try:
        make_chat_client("nope")
        raise AssertionError("未知 provider 应抛 ValueError")
    except ValueError:
        pass


# ---- config --------------------------------------------------------------
def test_config_defaults():
    cfg = AgentConfig.from_env({})
    assert cfg.brain_model == "deepseek-chat"
    assert cfg.worker_provider == "deepseek"
    assert cfg.image_enabled is False


def test_config_reads_env_overrides():
    cfg = AgentConfig.from_env(
        {
            "LLM_MODEL": "deepseek-v4-flash",
            "AGENT_WORKER_PROVIDER": "qwen",
            "AGENT_WORKER_LORA": "/lora/adapter",
            "AGENT_IMAGE_MODEL": "zimage",
        }
    )
    assert cfg.brain_model == "deepseek-v4-flash"
    assert cfg.worker_provider == "qwen" and cfg.worker_lora == "/lora/adapter"
    assert cfg.image_enabled is True


def test_config_rejects_unknown_worker_provider():
    try:
        AgentConfig.from_env({"AGENT_WORKER_PROVIDER": "gpt"})
        raise AssertionError("未知 worker provider 应抛 ValueError")
    except ValueError:
        pass
