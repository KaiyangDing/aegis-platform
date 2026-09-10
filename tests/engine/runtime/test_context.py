"""上下文编译（M2.6 compile_prompt）：v1 test_context_layers / test_context_summary 平移到 v2 形态——固定层序、system fail-loud、
当前 user 恒保留、旧轮压成 user + 终答、历史层从最新往回装、摘要份额、工具结果层折叠可回溯、空 AIMessage 丢弃、确定性；
以及 ModelCall 的 input_tokens_est 按编译后 prompt 估算（链路）。零真实调用。"""

import inspect

import pytest

pytest.importorskip(
    "app.engine.runtime.context",
    reason="M2.6 未敲：app/engine/runtime/context.py 不存在",
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.core.tokens import estimate_messages_tokens, estimate_tokens
from app.engine.runtime import utterances as u
from app.engine.runtime.context import (
    SUMMARY_SOURCE,
    CompiledPrompt,
    clip_to_budget,
    compile_prompt,
)
from app.engine.runtime.spec import AgentSpec, ContextConfig
from app.engine.runtime.state import AEGIS_SOURCE, SOURCE_PROTOCOL_RETRY
from tests.engine.runtime.doubles import collect, make_runtime, text_turn


def _spec(**config) -> AgentSpec:
    return AgentSpec(system_prompt="规则", context_config=ContextConfig(**config))


def _summary(text: str) -> HumanMessage:
    return HumanMessage(text, additional_kwargs={"lc_source": SUMMARY_SOURCE})


def _retry() -> HumanMessage:
    return HumanMessage(
        u.PROMPT_PROTOCOL_RETRY, additional_kwargs={AEGIS_SOURCE: SOURCE_PROTOCOL_RETRY}
    )


def _tool_round(*results: tuple[str, str]) -> list[BaseMessage]:
    calls = [{"name": "demo_tool", "args": {}, "id": cid} for cid, _ in results]
    return [
        AIMessage(content="", tool_calls=calls),
        *[
            ToolMessage(text, tool_call_id=cid, name="demo_tool")
            for cid, text in results
        ],
    ]


def _u(i: int) -> str:
    """第 i 轮 user 原文：15 token（CJK 14 + 单位数字 ≈1）。"""
    return f"问{i}" + "长" * 13


def _a(i: int) -> str:
    return f"答{i}" + "长" * 13


def _turns(n: int) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for i in range(1, n + 1):
        out += [HumanMessage(_u(i)), AIMessage(_a(i))]
    return out


def _texts(compiled: CompiledPrompt) -> list[str]:
    return [m.text for m in compiled.messages]


# ---------------------------------------------------------------- 层序与 system 层


def test_layer_order_snapshot():
    """system → 摘要 → 旧轮（user + 终答）→ 当前 user → 本轮工作序列（AI(tool_calls) / Tool / 纠错提示 / AI）。"""
    messages = [
        _summary("摘要"),
        HumanMessage("旧问"),
        *_tool_round(("t-old", "旧结果")),
        AIMessage("旧答"),
        HumanMessage("当前问"),
        *_tool_round(("t-1", "结果")),
        _retry(),
        AIMessage("答"),
    ]
    compiled = compile_prompt(messages, _spec())
    assert isinstance(compiled.system, SystemMessage)
    assert compiled.system.content == "规则\n\n" + u.UNTRUSTED_NOTICE
    assert [type(m).__name__ for m in compiled.messages] == [
        "HumanMessage",  # 摘要
        "HumanMessage",  # 旧问
        "AIMessage",  # 旧答（旧轮的工具往返不进 prompt）
        "HumanMessage",  # 当前问
        "AIMessage",  # tool_calls
        "ToolMessage",
        "HumanMessage",  # 纠错提示
        "AIMessage",
    ]
    assert _texts(compiled)[:4] == ["摘要", "旧问", "旧答", "当前问"]
    assert compiled.all_messages[0] is compiled.system
    assert compiled.folded == () and compiled.dropped_turns == 0


def test_entry_notice_joins_system_layer_and_counts_against_its_budget():
    """M2.8：MEDIUM 打标提醒紧随不可信声明进 system 层（固定模板），与 system 同层受同一预算；缺省不加。"""
    if "notice" not in inspect.signature(compile_prompt).parameters:
        pytest.skip("M2.8 未敲：compile_prompt 尚无 notice 形参")
    base = "规则\n\n" + u.UNTRUSTED_NOTICE
    tagged = compile_prompt([HumanMessage("你好")], _spec(), notice=u.SUSPICION_NOTICE)
    assert tagged.system.content == base + "\n\n" + u.SUSPICION_NOTICE
    assert compile_prompt([HumanMessage("你好")], _spec()).system.content == base
    tight = _spec(system_budget=estimate_tokens(base))
    assert compile_prompt([HumanMessage("你好")], tight).system.content == base
    with pytest.raises(ValueError, match="system_budget"):
        compile_prompt([HumanMessage("你好")], tight, notice=u.SUSPICION_NOTICE)


def test_system_over_budget_is_loud():
    with pytest.raises(ValueError, match="system_budget"):
        compile_prompt(
            [HumanMessage("你好")],
            AgentSpec(
                system_prompt="长" * 100, context_config=ContextConfig(system_budget=50)
            ),
        )


# ---------------------------------------------------------------- 当前 user 与工作序列


def test_current_user_input_is_verbatim_and_never_evicted():
    """user 超长（200 > history_budget 101）⇒ 历史全丢、原文照放（挤掉 / 截断用户的话 = 答非所问 / 篡改输入）。"""
    big = "问" * 200
    messages = [*_turns(2), HumanMessage(big)]
    compiled = compile_prompt(messages, _spec(history_budget=101))
    assert _texts(compiled) == [big] and compiled.dropped_turns == 2


def test_empty_assistant_messages_are_dropped():
    """零话术终止 / 协议违规留下的空 AIMessage（M2.3 登记 L2）不进 prompt，无论在旧轮还是本轮。"""
    messages = [
        HumanMessage("旧问"),
        AIMessage(""),
        HumanMessage("当前问"),
        AIMessage(""),
        _retry(),
        AIMessage("答"),
    ]
    compiled = compile_prompt(messages, _spec())
    assert _texts(compiled) == ["旧问", "当前问", u.PROMPT_PROTOCOL_RETRY, "答"]


def test_working_within_budget_untouched_and_ai_tool_calls_never_folded():
    working = _tool_round(("t-1", "已发货"), ("t-2", "明天到"))
    compiled = compile_prompt([HumanMessage("问"), *working], _spec())
    assert compiled.messages[1:] == working
    big_args = {"ids": "x" * 400}
    calls = [{"name": "demo_tool", "args": big_args, "id": "t-1"}]
    messages = [
        HumanMessage("问"),
        AIMessage(content="", tool_calls=calls),
        ToolMessage("旧" * 50, tool_call_id="t-1", name="demo_tool"),
    ]
    compiled = compile_prompt(messages, _spec(tool_results_budget=5))
    assert compiled.messages[1] == messages[1]  # arguments 是协议字段，字节不变
    assert compiled.messages[2].content == u.FOLDED_TOOL_TEMPLATE.format(
        tool_call_id="t-1"
    )
    assert compiled.folded == ("t-1",)


def test_folding_starts_from_oldest_tool_message():
    working = _tool_round(("tc-old", "旧" * 100), ("tc-new", "新" * 10))
    compiled = compile_prompt(
        [HumanMessage("问"), *working], _spec(tool_results_budget=60)
    )
    tools = {m.tool_call_id: m for m in compiled.messages if isinstance(m, ToolMessage)}
    assert tools["tc-old"].content == u.FOLDED_TOOL_TEMPLATE.format(
        tool_call_id="tc-old"
    )
    assert tools["tc-new"].content == "新" * 10
    assert compiled.folded == ("tc-old",)


# ---------------------------------------------------------------- 历史层：旧轮、摘要、份额


def test_older_turns_reduced_to_user_and_final_answer():
    """旧轮 = user 原话 + 最后一条有文字的 assistant；同轮多条 assistant 取终态；孤儿 user（上次崩溃）只产 user 条。"""
    messages = [
        HumanMessage("问1"),
        *_tool_round(("t-1", "结果")),
        AIMessage("中间答"),
        AIMessage("终答1"),
        HumanMessage("问2"),  # 孤儿轮
        HumanMessage("问3"),
        AIMessage("答3"),
        HumanMessage("当前"),
    ]
    compiled = compile_prompt(messages, _spec())
    assert _texts(compiled) == ["问1", "终答1", "问2", "问3", "答3", "当前"]


def test_leading_orphan_assistant_after_summary_cut_is_kept():
    """摘要切点可能落在 AI/Human 之间（框架 keep 语义）：前导的 assistant 自成一轮，按终答保留。"""
    messages = [_summary("摘要"), AIMessage("第一答"), HumanMessage("第二问")]
    compiled = compile_prompt(messages, _spec())
    assert _texts(compiled) == ["摘要", "第一答", "第二问"]


def test_history_fills_from_newest_and_drops_oldest():
    """history_budget=101、当前 user 1 token ⇒ 可用 100；四轮各 30 token ⇒ 保住最新三轮，最老一轮整轮丢弃。"""
    messages = [*_turns(4), HumanMessage("问")]
    compiled = compile_prompt(messages, _spec(history_budget=101))
    texts = _texts(compiled)
    assert _u(1) not in texts and _a(1) not in texts
    assert texts[:2] == [_u(2), _a(2)] and texts[-1] == "问"
    assert compiled.dropped_turns == 1
    assert sum(estimate_tokens(t) for t in texts[:-1]) <= 100


def test_fat_summary_is_capped_to_half_when_turns_queue():
    """v1 复盘补丁三：有旧轮排队时摘要至多占版面一半——肥摘要不再独占历史层，最新轮进场。"""
    messages = [_summary("超" * 500), *_turns(2), HumanMessage("问")]
    compiled = compile_prompt(messages, _spec(history_budget=101))
    summary = compiled.messages[0]
    assert summary.additional_kwargs["lc_source"] == SUMMARY_SOURCE
    assert summary.text.endswith(u.CLIP_SUFFIX) and compiled.summary_clipped
    assert estimate_tokens(summary.text) <= 50
    texts = _texts(compiled)
    assert _u(2) in texts and _a(2) in texts  # 最新轮进场——修复的靶心
    assert sum(estimate_tokens(t) for t in texts[:-1]) <= 100


def test_summary_share_not_applied_without_queue():
    messages = [_summary("长" * 70), HumanMessage("问")]
    compiled = compile_prompt(messages, _spec(history_budget=101))
    assert compiled.messages[0].text == "长" * 70 and not compiled.summary_clipped


def test_history_budget_zero_closes_the_layer():
    messages = [_summary("摘要"), *_turns(2), HumanMessage("问")]
    compiled = compile_prompt(messages, _spec(history_budget=0))
    assert _texts(compiled) == ["问"] and compiled.dropped_turns == 2


def test_compile_is_deterministic():
    messages = [
        _summary("摘要" * 30),
        *_turns(3),
        HumanMessage("问"),
        *_tool_round(("t", "结" * 50)),
    ]
    a = compile_prompt(messages, _spec(history_budget=101, tool_results_budget=20))
    b = compile_prompt(messages, _spec(history_budget=101, tool_results_budget=20))
    assert _texts(a) == _texts(b) and a.folded == b.folded


def test_clip_to_budget_marks_and_fits():
    assert clip_to_budget("短", 10) == "短"
    clipped = clip_to_budget("长" * 100, 20)
    assert clipped.endswith(u.CLIP_SUFFIX) and estimate_tokens(
        clipped
    ) <= 20 + estimate_tokens(u.CLIP_SUFFIX)


# ---------------------------------------------------------------- 链路：input_tokens_est 按编译后 prompt


async def test_input_estimate_uses_compiled_prompt():
    spec = AgentSpec(system_prompt="你是演示客服。", model_tier="fast")
    rt, _cand, _, sessions = make_runtime(text_turn("好"))
    await sessions.create("s-1", tenant_id="t-a", user_id="u-1")
    got = await collect(
        rt, tenant_id="t-a", session_id="s-1", user_input="退款申请", spec=spec
    )
    expected = estimate_messages_tokens(
        [
            SystemMessage("你是演示客服。\n\n" + u.UNTRUSTED_NOTICE),
            HumanMessage("退款申请"),
        ]
    )
    assert got[1].payload["input_tokens_est"] == expected
