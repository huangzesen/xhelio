"""Tests for shared LLM session behavior."""
from agent.llm_session_mixin import COMPACTION_PROMPT, summarize_tool_calls, summarize_tool_results


def test_compaction_prompt_is_string():
    assert isinstance(COMPACTION_PROMPT, str)
    assert len(COMPACTION_PROMPT) > 50
    assert "Summarize" in COMPACTION_PROMPT


def test_summarize_tool_calls_empty():
    assert summarize_tool_calls([]) == "(no tool calls found)"


def test_summarize_tool_calls_basic():
    history = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "fetch_data",
                    "input": {"dataset": "ACE_MAG"},
                }
            ],
        },
    ]
    result = summarize_tool_calls(history)
    assert "fetch_data" in result
    assert "ACE_MAG" in result


def test_summarize_tool_calls_no_assistant():
    history = [{"role": "user", "content": "hello"}]
    assert summarize_tool_calls(history) == "(no tool calls found)"


def test_summarize_tool_results_string():
    assert summarize_tool_results("some result")[:11] == "some result"


def test_summarize_tool_results_empty():
    assert summarize_tool_results([]) == "(tool results not available)"


def test_summarize_tool_results_dict():
    msg = {"content": "Fetched 100 rows of data"}
    result = summarize_tool_results(msg)
    assert "Fetched 100 rows" in result
