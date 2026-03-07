"""Tests for ChatInterface <-> provider format converters."""

import json
from agent.llm.interface import (
    ChatInterface,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ThinkingBlock,
    ImageBlock,
)


class TestAnthropicConverter:
    def test_user_text(self):
        from agent.llm.interface_converters import to_anthropic
        iface = ChatInterface()
        iface.add_user_message("hello")
        msgs = to_anthropic(iface)
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_assistant_with_tool_call(self):
        from agent.llm.interface_converters import to_anthropic
        iface = ChatInterface()
        iface.add_assistant_message([
            TextBlock(text="I'll fetch"),
            ToolCallBlock(id="toolu_123", name="fetch", args={"ds": "ACE"}),
        ])
        msgs = to_anthropic(iface)
        content = msgs[0]["content"]
        assert content[0] == {"type": "text", "text": "I'll fetch"}
        assert content[1] == {"type": "tool_use", "id": "toolu_123", "name": "fetch", "input": {"ds": "ACE"}}

    def test_tool_results(self):
        from agent.llm.interface_converters import to_anthropic
        iface = ChatInterface()
        iface.add_tool_results([
            ToolResultBlock(id="toolu_123", name="fetch", content='{"ok":true}'),
        ])
        msgs = to_anthropic(iface)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "toolu_123"

    def test_thinking_with_signature(self):
        from agent.llm.interface_converters import to_anthropic
        iface = ChatInterface()
        iface.add_assistant_message([
            ThinkingBlock(text="reasoning", provider_data={"anthropic": {"signature": "sig"}}),
            TextBlock(text="answer"),
        ])
        content = to_anthropic(iface)[0]["content"]
        assert content[0] == {"type": "thinking", "thinking": "reasoning", "signature": "sig"}

    def test_system_excluded(self):
        from agent.llm.interface_converters import to_anthropic
        iface = ChatInterface()
        iface.add_system("sys")
        iface.add_user_message("hi")
        msgs = to_anthropic(iface)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_from_anthropic(self):
        from agent.llm.interface_converters import from_anthropic
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "fn", "input": {"x": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": '{"ok":true}'},
            ]},
        ]
        iface = from_anthropic(messages, system_prompt="sys")
        assert iface.current_system_prompt == "sys"
        assert len(iface.entries) == 4  # system + 3 messages
        assert isinstance(iface.entries[2].content[1], ToolCallBlock)
        assert isinstance(iface.entries[3].content[0], ToolResultBlock)


class TestOpenAIConverter:
    def test_system_included(self):
        from agent.llm.interface_converters import to_openai
        iface = ChatInterface()
        iface.add_system("sys")
        iface.add_user_message("hi")
        msgs = to_openai(iface)
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_tool_call(self):
        from agent.llm.interface_converters import to_openai
        iface = ChatInterface()
        iface.add_assistant_message([
            TextBlock(text="calling"),
            ToolCallBlock(id="call_abc", name="fn", args={"x": 1}),
        ])
        msg = to_openai(iface)[0]
        assert msg["content"] == "calling"
        assert msg["tool_calls"][0]["id"] == "call_abc"
        assert msg["tool_calls"][0]["function"]["name"] == "fn"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}

    def test_tool_results_become_separate_messages(self):
        from agent.llm.interface_converters import to_openai
        iface = ChatInterface()
        iface.add_tool_results([
            ToolResultBlock(id="call_abc", name="fn", content={"ok": True}),
            ToolResultBlock(id="call_def", name="fn2", content="done"),
        ])
        msgs = to_openai(iface)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_abc"
        assert msgs[1]["tool_call_id"] == "call_def"

    def test_from_openai(self):
        from agent.llm.interface_converters import from_openai
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "fn", "arguments": '{"x":1}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'},
        ]
        iface = from_openai(messages)
        assert iface.current_system_prompt == "sys"
        assert len(iface.entries) == 4
        assert isinstance(iface.entries[2].content[1], ToolCallBlock)
        assert isinstance(iface.entries[3].content[0], ToolResultBlock)


class TestGeminiConverter:
    def test_user_text(self):
        from agent.llm.interface_converters import to_gemini
        iface = ChatInterface()
        iface.add_user_message("hello")
        turns = to_gemini(iface)
        assert turns == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_assistant_role_becomes_model(self):
        from agent.llm.interface_converters import to_gemini
        iface = ChatInterface()
        iface.add_assistant_message([TextBlock(text="hi")])
        turns = to_gemini(iface)
        assert turns[0]["role"] == "model"

    def test_tool_call(self):
        from agent.llm.interface_converters import to_gemini
        iface = ChatInterface()
        iface.add_assistant_message([
            ToolCallBlock(id="fc_1", name="fetch", args={"ds": "ACE"}),
        ])
        c = to_gemini(iface)[0]["content"][0]
        assert c == {"type": "function_call", "id": "fc_1", "name": "fetch", "arguments": {"ds": "ACE"}}

    def test_tool_results(self):
        from agent.llm.interface_converters import to_gemini
        iface = ChatInterface()
        iface.add_tool_results([
            ToolResultBlock(id="fc_1", name="fetch", content={"status": "ok"}),
        ])
        c = to_gemini(iface)[0]["content"][0]
        assert c["type"] == "function_result"
        assert c["call_id"] == "fc_1"

    def test_system_excluded(self):
        from agent.llm.interface_converters import to_gemini
        iface = ChatInterface()
        iface.add_system("sys")
        iface.add_user_message("hi")
        turns = to_gemini(iface)
        assert len(turns) == 1

    def test_from_gemini(self):
        from agent.llm.interface_converters import from_gemini
        turns = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "model", "content": [
                {"type": "function_call", "id": "fc1", "name": "fn", "arguments": {"x": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "function_result", "call_id": "fc1", "name": "fn", "result": '{"ok":true}'},
            ]},
        ]
        iface = from_gemini(turns, system_prompt="sys")
        assert iface.current_system_prompt == "sys"
        assert len(iface.entries) == 4
        assert iface.entries[2].role == "assistant"
        assert isinstance(iface.entries[2].content[0], ToolCallBlock)


class TestCrossProviderRoundtrip:
    def test_canonical_to_all_providers(self):
        from agent.llm.interface_converters import to_anthropic, to_openai, to_gemini
        iface = ChatInterface()
        iface.add_system("sys")
        iface.add_user_message("fetch ACE data")
        iface.add_assistant_message([
            TextBlock(text="fetching"),
            ToolCallBlock(id="c1", name="fetch", args={"ds": "ACE"}),
        ])
        iface.add_tool_results([
            ToolResultBlock(id="c1", name="fetch", content={"rows": 100}),
        ])
        iface.add_assistant_message([TextBlock(text="done")])

        anthropic = to_anthropic(iface)
        openai = to_openai(iface)
        gemini = to_gemini(iface)

        assert len(anthropic) == 4  # no system
        assert len(openai) == 5    # system + user + asst + tool + asst
        assert openai[0]["role"] == "system"
        assert len(gemini) == 4    # no system

    def test_provider_data_survives_roundtrip(self):
        iface = ChatInterface()
        iface.add_assistant_message([
            ThinkingBlock(text="reasoning", provider_data={"anthropic": {"signature": "sig_xyz"}}),
            TextBlock(text="answer"),
        ])
        saved = iface.to_dict()
        restored = ChatInterface.from_dict(saved)
        thinking = restored.entries[0].content[0]
        assert isinstance(thinking, ThinkingBlock)
        assert thinking.provider_data["anthropic"]["signature"] == "sig_xyz"
