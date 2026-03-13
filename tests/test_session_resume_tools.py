"""Tests for session resume tool restoration."""
import pytest
from unittest.mock import MagicMock
from agent.llm.base import FunctionSchema
from agent.llm.service import LLMService
from agent.llm.interface import ChatInterface


class TestResumeSessionTools:
    def test_resume_passes_tools_to_create_chat(self):
        """resume_session() should extract tools from interface and pass to create_chat()."""
        iface = ChatInterface()
        tool_dicts = [
            {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
            {"name": "list_data", "description": "List data", "parameters": {"type": "object", "properties": {}}},
        ]
        iface.add_system("You are a test assistant.", tools=tool_dicts)
        saved_state = {
            "session_id": "test_session",
            "messages": iface.to_dict(),
            "metadata": {},
        }

        mock_adapter = MagicMock()
        mock_chat = MagicMock()
        mock_chat.interface = MagicMock()
        mock_chat.interface.entries = []
        mock_adapter.create_chat.return_value = mock_chat

        service = LLMService.__new__(LLMService)
        service._adapter = mock_adapter
        service._model = "test-model"
        service._sessions = {}

        service.resume_session(saved_state)

        _, kwargs = mock_adapter.create_chat.call_args
        tools = kwargs.get("tools")
        assert tools is not None, "tools must not be None — they should be restored from the interface"
        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "list_data"

    def test_resume_no_tools_passes_none(self):
        """resume_session() with no tools in interface should pass tools=None."""
        iface = ChatInterface()
        iface.add_system("You are a test assistant.", tools=None)
        saved_state = {
            "session_id": "test_session",
            "messages": iface.to_dict(),
            "metadata": {},
        }

        mock_adapter = MagicMock()
        mock_chat = MagicMock()
        mock_chat.interface = MagicMock()
        mock_chat.interface.entries = []
        mock_adapter.create_chat.return_value = mock_chat

        service = LLMService.__new__(LLMService)
        service._adapter = mock_adapter
        service._model = "test-model"
        service._sessions = {}

        service.resume_session(saved_state)

        _, kwargs = mock_adapter.create_chat.call_args
        assert kwargs.get("tools") is None


class TestFunctionSchemaFromDicts:
    def test_round_trip(self):
        """FunctionSchema -> dict -> FunctionSchema preserves all fields."""
        original = FunctionSchema(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        dicts = [{"name": original.name, "description": original.description, "parameters": original.parameters}]
        restored = FunctionSchema.from_dicts(dicts)
        assert len(restored) == 1
        assert restored[0].name == "get_weather"
        assert restored[0].description == "Get weather for a city"
        assert restored[0].parameters["required"] == ["city"]

    def test_none_returns_none(self):
        assert FunctionSchema.from_dicts(None) is None

    def test_empty_returns_none(self):
        assert FunctionSchema.from_dicts([]) is None


class TestToolsRoundTrip:
    """Verify tools survive: create_chat → save interface → from_dict → resume_session."""

    def test_tools_survive_serialize_deserialize(self):
        """Tools survive the full round-trip through ChatInterface serialization."""
        tool_dicts = [
            {"name": "browse_datasets", "description": "Browse datasets", "parameters": {"type": "object", "properties": {"mission": {"type": "string"}}, "required": ["mission"]}},
            {"name": "assets", "description": "List data", "parameters": {"type": "object", "properties": {}}},
        ]

        # 1. Create interface with tools
        iface = ChatInterface()
        iface.add_system("You are a test assistant.", tools=tool_dicts)

        # 2. Serialize (simulates session save)
        saved = iface.to_dict()

        # 3. Restore (simulates session load)
        restored_iface = ChatInterface.from_dict(saved)

        # 4. Verify tools survived
        assert restored_iface.current_tools is not None
        assert len(restored_iface.current_tools) == 2
        assert restored_iface.current_tools[0]["name"] == "browse_datasets"

        # 5. Convert back to FunctionSchema (simulates resume_session)
        schemas = FunctionSchema.from_dicts(restored_iface.current_tools)
        assert schemas is not None
        assert len(schemas) == 2
        assert schemas[0].name == "browse_datasets"
        assert schemas[1].name == "assets"
        assert schemas[0].parameters["required"] == ["mission"]
