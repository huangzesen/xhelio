"""Tests for InteractionsChatSession client-side history tracking."""

import pytest
from unittest.mock import MagicMock, patch
from agent.llm.gemini_adapter import InteractionsChatSession


class FakeInteraction:
    """Minimal mock of a Gemini Interactions API response."""

    def __init__(self, interaction_id="int_001", text="Hello"):
        self.id = interaction_id
        self.outputs = [FakeTextOutput(text)]
        self.usage = FakeUsage()


class FakeTextOutput:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeUsage:
    total_input_tokens = 10
    total_output_tokens = 5
    total_thought_tokens = 0
    total_cached_tokens = 0


class TestClientSideHistory:
    def test_history_empty_initially(self):
        client = MagicMock()
        session = InteractionsChatSession(
            client=client, model="gemini-test", config_kwargs={}
        )
        assert session.get_client_history() == []

    def test_history_records_user_and_model_turns(self):
        client = MagicMock()
        client.interactions.create.return_value = FakeInteraction(
            interaction_id="int_001", text="Hi there!"
        )
        session = InteractionsChatSession(
            client=client, model="gemini-test", config_kwargs={}
        )

        session.send("Hello")

        history = session.get_client_history()
        assert len(history) == 2  # user turn + model turn
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "model"
        # User turn contains the text input
        assert any(
            block.get("type") == "text" and block.get("text") == "Hello"
            for block in history[0]["content"]
        )
        # Model turn contains the response text
        assert any(
            block.get("type") == "text" and block.get("text") == "Hi there!"
            for block in history[1]["content"]
        )

    def test_history_records_multiple_turns(self):
        client = MagicMock()
        client.interactions.create.side_effect = [
            FakeInteraction("int_001", "Reply 1"),
            FakeInteraction("int_002", "Reply 2"),
        ]
        session = InteractionsChatSession(
            client=client, model="gemini-test", config_kwargs={}
        )

        session.send("First")
        session.send("Second")

        history = session.get_client_history()
        assert len(history) == 4  # 2 user + 2 model turns

    def test_get_history_includes_client_history(self):
        """get_history() should return both interaction_id and client history."""
        client = MagicMock()
        client.interactions.create.return_value = FakeInteraction("int_001", "Hi")
        session = InteractionsChatSession(
            client=client, model="gemini-test", config_kwargs={}
        )
        session.send("Hello")

        result = session.get_history()
        assert any("_interaction_id" in entry for entry in result)
        assert any("_client_history" in entry for entry in result)


class FakeToolCallOutput:
    def __init__(self, name, args):
        self.type = "function_call"
        self.name = name
        self.arguments = args
        self.id = "call_001"


class FakeInteractionWithToolCall:
    def __init__(self, interaction_id, tool_name, tool_args):
        self.id = interaction_id
        self.outputs = [FakeToolCallOutput(tool_name, tool_args)]
        self.usage = FakeUsage()


class TestClientSideHistoryToolResults:
    def test_tool_results_recorded_as_user_turn(self):
        client = MagicMock()
        client.interactions.create.side_effect = [
            FakeInteractionWithToolCall("int_001", "get_data", {"key": "val"}),
            FakeInteraction("int_002", "Here is the data"),
        ]
        session = InteractionsChatSession(
            client=client, model="gemini-test", config_kwargs={}
        )

        # First call: user sends text, model responds with tool call
        session.send("Get my data")

        # Second call: user sends tool results
        tool_results = [
            {
                "type": "function_result",
                "call_id": "call_001",
                "result": '{"data": "value"}',
                "name": "get_data",
            }
        ]
        session.send(tool_results)

        history = session.get_client_history()
        # Should have 4 entries: user, model(tool_call), user(tool_result), model(text)
        assert len(history) == 4
        assert history[2]["role"] == "user"
        assert any(
            block.get("type") == "function_result" for block in history[2]["content"]
        )


class TestSeedTurnsFromClientHistory:
    def test_interactions_session_receives_history_as_seed_turns(self):
        """InteractionsChatSession should receive history and convert to seed turns."""
        from agent.llm.gemini_adapter import _convert_history_to_turns

        # Simulate what happens in create_chat when history is provided
        client_history = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "model", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        # This is what create_chat does - converts history to seed turns
        seed_turns = _convert_history_to_turns(client_history)

        # The history should be converted to the expected format
        assert seed_turns is not None
        assert len(seed_turns) == 2
        assert seed_turns[0]["role"] == "user"
        assert seed_turns[1]["role"] == "model"
