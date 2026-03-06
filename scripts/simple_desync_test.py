#!/usr/bin/env python
"""Simple test for history desync - directly uses the adapter."""

import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import config
import config as cfg
from agent.llm import LLMAdapter
from agent.logging import setup_logging, get_logger


def create_adapter() -> LLMAdapter:
    """Create the LLM adapter based on config."""
    provider = config.LLM_PROVIDER.lower()
    api_key = config.get_api_key()
    if provider == "openai":
        from agent.llm import OpenAIAdapter
        return OpenAIAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
    elif provider == "anthropic":
        from agent.llm import AnthropicAdapter
        return AnthropicAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
    elif provider == "minimax":
        from agent.llm import MiniMaxAdapter
        return MiniMaxAdapter(api_key=api_key, base_url=config.LLM_BASE_URL)
    else:
        from agent.llm import GeminiAdapter
        return GeminiAdapter(api_key=api_key)

setup_logging(verbose=True)
logger = get_logger()


def test_conversation():
    """Test a conversation with tool calls to trigger desync."""
    print("Creating adapter...")
    adapter = create_adapter()
    model = config.SUB_AGENT_MODEL

    print(f"Creating chat with model: {model}")

    # Create a chat session
    chat = adapter.create_chat(
        model=model,
        system_prompt="You are a helpful data assistant.",
        tools=None,  # No tools for simplicity
    )

    print("Chat created")

    # Simple conversation
    requests = [
        "Hello, how are you?",
        "What's 2+2?",
        "What is pandas?",
        "Explain machine learning",
        "What is Python?",
    ]

    for i in range(10):
        request = random.choice(requests)
        print(f"\n[{i+1}] Request: {request[:50]}...")

        try:
            response = chat.send(request)
            print(f"    Response: {response.text[:100]}...")

            # Check history
            history = chat.get_history()
            print(f"    History: {len(history)} messages")

        except Exception as e:
            error_msg = str(e)
            print(f"    ERROR: {error_msg[:200]}")

            if "tool call result does not follow" in error_msg or "2013" in error_msg:
                print("    >>> DESYNC ERROR!")
                # Save debug info
                debug_file = REPO / f"/tmp/desync_direct_{i}.json"
                with open(debug_file, "w") as f:
                    json.dump({
                        "iteration": i,
                        "request": request,
                        "error": error_msg,
                        "history": chat.get_history(),
                    }, f, indent=2, default=str)

                # Reset chat
                chat = adapter.create_chat(
                    model=model,
                    system_prompt="You are a helpful data assistant.",
                    tools=None,
                )
                print("    Chat reset")

        time.sleep(0.5)

    print("\n=== Done ===")


def test_with_tools():
    """Test with actual tool calls."""
    print("\n=== Testing with tools ===")

    from agent.tools import get_function_schemas

    adapter = create_adapter()
    model = config.SUB_AGENT_MODEL
    tool_schemas = get_function_schemas(names=["list_fetched_data"])

    chat = adapter.create_chat(
        model=model,
        system_prompt="You are a helpful data assistant with access to tools.",
        tools=tool_schemas,
    )

    requests = [
        "List fetched data",
        "Show me available data",
        "What data do we have?",
    ]

    for i in range(15):
        request = random.choice(requests)
        print(f"\n[{i+1}] Request: {request}")

        try:
            response = chat.send(request)
            print(f"    Text: {response.text[:100]}...")
            print(f"    Tool calls: {len(response.tool_calls)}")

            # If there are tool calls, execute them and send results
            if response.tool_calls:
                print(f"    Executing {len(response.tool_calls)} tool call(s)")

                # Get tool results (mock)
                tool_results = []
                for tc in response.tool_calls:
                    result = {"status": "success", "message": "Mock result"}
                    tool_result_msg = adapter.make_tool_result_message(
                        tc.name, result, tool_call_id=tc.id
                    )
                    tool_results.append(tool_result_msg)

                # Send tool results back
                response2 = chat.send(tool_results)
                print(f"    After tool: {response2.text[:100]}...")

            history = chat.get_history()
            print(f"    History: {len(history)} messages")

        except Exception as e:
            error_msg = str(e)
            print(f"    ERROR: {error_msg[:200]}")

            if "tool call result does not follow" in error_msg or "2013" in error_msg:
                print("    >>> DESYNC ERROR!")
                debug_file = REPO / f"/tmp/desync_tools_{i}.json"
                with open(debug_file, "w") as f:
                    json.dump({
                        "iteration": i,
                        "request": request,
                        "error": error_msg,
                        "history": chat.get_history(),
                    }, f, indent=2, default=str)

                # Reset
                chat = adapter.create_chat(
                    model=model,
                    system_prompt="You are a helpful data assistant with access to tools.",
                    tools=tool_schemas,
                )

        time.sleep(0.3)

    print("\n=== Done with tools ===")


if __name__ == "__main__":
    test_conversation()
    print("\n" + "="*50)
    test_with_tools()
