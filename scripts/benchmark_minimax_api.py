#!/usr/bin/env python3
"""Benchmark MiniMax vs Anthropic API behavioral differences.

Probes MiniMax's Anthropic-compatible API on 10 known or suspected
divergence points. Produces a structured pass/fail report showing
exactly where the APIs differ.

Usage:
    python scripts/benchmark_minimax_api.py
    python scripts/benchmark_minimax_api.py --verbose   # show full API responses
    python scripts/benchmark_minimax_api.py --delay 3    # custom delay between tests (default: 2)
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import anthropic

BASE_URL = "https://api.minimaxi.com/anthropic"
MODEL = "MiniMax-M2.5-highspeed"

VERBOSE = False
DELAY = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> anthropic.Anthropic:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("ERROR: MINIMAX_API_KEY not set in .env or environment")
        sys.exit(1)
    return anthropic.Anthropic(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=60.0,
    )


SIMPLE_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
}


def rate_limit():
    """Sleep between API calls to respect rate limits."""
    time.sleep(DELAY)


def log_verbose(label: str, data):
    if VERBOSE:
        print(f"    [{label}]")
        if hasattr(data, "__dict__"):
            # Anthropic response objects
            try:
                print(f"    {json.dumps(json.loads(data.model_dump_json()), indent=2)}")
            except Exception:
                print(f"    {data}")
        elif isinstance(data, dict):
            print(f"    {json.dumps(data, indent=2, default=str)}")
        else:
            print(f"    {data}")


# ---------------------------------------------------------------------------
# Test results tracking
# ---------------------------------------------------------------------------


class TestResult:
    def __init__(self, name: str, status: str, expected: str, actual: str, details: str = ""):
        self.name = name
        self.status = status  # PASS, FAIL, INFO, ERROR
        self.expected = expected
        self.actual = actual
        self.details = details


results: list[TestResult] = []


def record(name: str, status: str, expected: str, actual: str, details: str = ""):
    results.append(TestResult(name, status, expected, actual, details))


# ---------------------------------------------------------------------------
# Test 1: Thinking Block Replay
# ---------------------------------------------------------------------------


def test_1_thinking_block_replay():
    """Send a message with thinking enabled, get response with thinking blocks + signatures,
    then replay the history (including thinking blocks) in a follow-up."""
    print("\n--- Test 1: Thinking Block Replay ---")
    client = make_client()

    try:
        # Step 1: Get a response with thinking
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048,
            },
            messages=[{"role": "user", "content": "What is 2+2? Think carefully."}],
        )
        log_verbose("resp1", resp1)

        # Extract thinking blocks from response
        thinking_blocks = []
        text_blocks = []
        for block in resp1.content:
            if block.type == "thinking":
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", ""),
                })
            elif block.type == "text":
                text_blocks.append({"type": "text", "text": block.text})

        print(f"  Step 1: Got response with {len(thinking_blocks)} thinking block(s), {len(text_blocks)} text block(s)")
        if thinking_blocks:
            sig = thinking_blocks[0].get("signature", "")
            print(f"  Thinking signature present: {bool(sig)} (len={len(sig)})")

        # Step 2: Replay history WITH thinking blocks
        history = [
            {"role": "user", "content": "What is 2+2? Think carefully."},
            {"role": "assistant", "content": thinking_blocks + text_blocks},
        ]

        rate_limit()

        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048,
            },
            messages=history + [{"role": "user", "content": "What is 3+3?"}],
        )
        log_verbose("resp2", resp2)
        print(f"  Step 2: Follow-up with thinking blocks in history SUCCEEDED")
        print(f"  Response: {resp2.content[-1].text[:100] if resp2.content else '(empty)'}")
        record(
            "1: Thinking Block Replay",
            "PASS",
            "Accept or reject thinking blocks with signatures in history",
            "ACCEPTED — thinking blocks with signatures are replayed without error",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  Step 2: Follow-up with thinking blocks in history FAILED")
        print(f"  Error: {err_msg[:300]}")
        record(
            "1: Thinking Block Replay",
            "INFO",
            "Accept or reject thinking blocks with signatures in history",
            f"REJECTED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 2: Thinking Block Replay (Stripped)
# ---------------------------------------------------------------------------


def test_2_thinking_stripped():
    """Same as Test 1, but strip thinking blocks from history before follow-up."""
    print("\n--- Test 2: Thinking Block Replay (Stripped) ---")
    client = make_client()

    try:
        # Step 1: Get a response with thinking
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048,
            },
            messages=[{"role": "user", "content": "What is 2+2? Think carefully."}],
        )
        log_verbose("resp1", resp1)

        # Strip thinking blocks — keep only text
        text_blocks = []
        for block in resp1.content:
            if block.type == "text":
                text_blocks.append({"type": "text", "text": block.text})
        if not text_blocks:
            text_blocks = [{"type": "text", "text": ""}]

        print(f"  Step 1: Got response, stripped thinking, kept {len(text_blocks)} text block(s)")

        # Step 2: Replay history WITHOUT thinking blocks
        history = [
            {"role": "user", "content": "What is 2+2? Think carefully."},
            {"role": "assistant", "content": text_blocks},
        ]

        rate_limit()

        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048,
            },
            messages=history + [{"role": "user", "content": "What is 3+3?"}],
        )
        log_verbose("resp2", resp2)
        print(f"  Step 2: Follow-up with stripped history SUCCEEDED")
        print(f"  Response: {resp2.content[-1].text[:100] if resp2.content else '(empty)'}")
        record(
            "2: Thinking Stripped Replay",
            "PASS",
            "Should work (our current workaround)",
            "Works correctly — stripped history accepted",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  Step 2: Follow-up with stripped history FAILED")
        print(f"  Error: {err_msg[:300]}")
        record(
            "2: Thinking Stripped Replay",
            "FAIL",
            "Should work (our current workaround)",
            f"FAILED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 3: Prompt Caching (cache_control: ephemeral)
# ---------------------------------------------------------------------------


def test_3_cache_control():
    """Send a request with cache_control markers on system prompt and tools."""
    print("\n--- Test 3: Prompt Caching (cache_control: ephemeral) ---")
    client = make_client()

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You are a helpful weather assistant. Always respond concisely.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    **SIMPLE_TOOL,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        )
        log_verbose("resp", resp)

        # Check usage for cache fields
        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cache_creation_input_tokens", None)
        print(f"  Response received successfully")
        print(f"  cache_read_input_tokens: {cache_read}")
        print(f"  cache_creation_input_tokens: {cache_write}")
        print(f"  input_tokens: {usage.input_tokens}")
        print(f"  output_tokens: {usage.output_tokens}")
        record(
            "3: cache_control: ephemeral",
            "PASS",
            "Accept, silently ignore, or error on cache_control markers",
            f"ACCEPTED — cache_read={cache_read}, cache_write={cache_write}",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  FAILED: {err_msg[:300]}")
        record(
            "3: cache_control: ephemeral",
            "INFO",
            "Accept, silently ignore, or error on cache_control markers",
            f"REJECTED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 4: Without cache_control
# ---------------------------------------------------------------------------


def test_4_no_cache_control():
    """Same request without cache_control markers."""
    print("\n--- Test 4: Without cache_control ---")
    client = make_client()

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant. Always respond concisely.",
            tools=[SIMPLE_TOOL],
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        )
        log_verbose("resp", resp)

        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_write = getattr(usage, "cache_creation_input_tokens", None)
        print(f"  Response received successfully")
        print(f"  cache_read_input_tokens: {cache_read}")
        print(f"  cache_creation_input_tokens: {cache_write}")
        print(f"  input_tokens: {usage.input_tokens}")
        print(f"  output_tokens: {usage.output_tokens}")
        record(
            "4: No cache_control",
            "PASS",
            "Should work normally",
            f"Works — cache_read={cache_read}, cache_write={cache_write}",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  FAILED: {err_msg[:300]}")
        record(
            "4: No cache_control",
            "FAIL",
            "Should work normally",
            f"FAILED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 5: Tool Use/Result Pairing (Correct)
# ---------------------------------------------------------------------------


def test_5_tool_result_correct():
    """Send a message that triggers a tool call, then send back a properly paired tool_result."""
    print("\n--- Test 5: Tool Use/Result Pairing (Correct) ---")
    client = make_client()

    try:
        # Step 1: Trigger a tool call
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
            tools=[SIMPLE_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        )
        log_verbose("resp1", resp1)

        tool_use_block = None
        for block in resp1.content:
            if block.type == "tool_use":
                tool_use_block = block
                break

        if not tool_use_block:
            print(f"  No tool_use block in response — model didn't call a tool")
            record(
                "5: Tool Result Pairing",
                "INFO",
                "Normal tool call + result flow",
                "Model did not call a tool despite tool_choice=any",
            )
            return

        print(f"  Step 1: Got tool call: {tool_use_block.name}(id={tool_use_block.id})")

        # Build history with assistant's tool call
        assistant_content = []
        for block in resp1.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        rate_limit()

        # Step 2: Send back correctly paired tool_result
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant.",
            tools=[SIMPLE_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps({"temperature": "22°C", "condition": "Sunny"}),
                        }
                    ],
                },
            ],
        )
        log_verbose("resp2", resp2)

        response_text = ""
        for block in resp2.content:
            if block.type == "text":
                response_text += block.text
        print(f"  Step 2: Tool result accepted, response: {response_text[:100]}")
        record(
            "5: Tool Result Pairing",
            "PASS",
            "Normal tool call + result flow should work",
            "Works correctly",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  FAILED: {err_msg[:300]}")
        record(
            "5: Tool Result Pairing",
            "FAIL",
            "Normal tool call + result flow should work",
            f"FAILED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 6: Tool Use/Result Mismatch (Wrong ID)
# ---------------------------------------------------------------------------


def test_6_tool_result_wrong_id():
    """Send a tool_result with a wrong/stale tool_use_id."""
    print("\n--- Test 6: Tool Use/Result Mismatch (Wrong ID) ---")
    client = make_client()

    try:
        # Step 1: Trigger a tool call
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant.",
            tools=[SIMPLE_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
        )
        log_verbose("resp1", resp1)

        tool_use_block = None
        for block in resp1.content:
            if block.type == "tool_use":
                tool_use_block = block
                break

        if not tool_use_block:
            print(f"  No tool_use block — skipping")
            record(
                "6: Wrong tool_use_id",
                "INFO",
                "Should error with specific error message",
                "Model did not call a tool — cannot test",
            )
            return

        print(f"  Step 1: Got tool call: {tool_use_block.name}(id={tool_use_block.id})")

        assistant_content = []
        for block in resp1.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        rate_limit()

        # Step 2: Send tool_result with WRONG ID
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant.",
            tools=[SIMPLE_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in Berlin?"},
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_WRONG_ID_12345",
                            "content": json.dumps({"temperature": "18°C"}),
                        }
                    ],
                },
            ],
        )
        log_verbose("resp2", resp2)
        print(f"  Step 2: Wrong ID was ACCEPTED (unexpected)")
        record(
            "6: Wrong tool_use_id",
            "INFO",
            "Should error with specific error message",
            "ACCEPTED — MiniMax does not validate tool_use_id",
        )
    except anthropic.BadRequestError as e:
        err_msg = str(e)
        print(f"  Step 2: Wrong ID was REJECTED (expected)")
        print(f"  Error message: {err_msg[:300]}")
        record(
            "6: Wrong tool_use_id",
            "PASS",
            "Should error with specific error message",
            f"Rejected with: {err_msg[:200]}",
            details=err_msg,
        )
    except Exception as e:
        err_msg = str(e)
        err_type = type(e).__name__
        print(f"  Step 2: Got {err_type}")
        print(f"  Error message: {err_msg[:300]}")
        record(
            "6: Wrong tool_use_id",
            "INFO",
            "Should error with specific error message",
            f"{err_type}: {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 7: Tool Use/Result Mismatch (Missing Result)
# ---------------------------------------------------------------------------


def test_7_tool_result_missing():
    """Send a plain text user message when the API expects a tool_result."""
    print("\n--- Test 7: Tool Use/Result Missing ---")
    client = make_client()

    try:
        # Step 1: Trigger a tool call
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant.",
            tools=[SIMPLE_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": "What is the weather in London?"}],
        )
        log_verbose("resp1", resp1)

        tool_use_block = None
        for block in resp1.content:
            if block.type == "tool_use":
                tool_use_block = block
                break

        if not tool_use_block:
            print(f"  No tool_use block — skipping")
            record(
                "7: Missing tool_result",
                "INFO",
                "Should error when tool_result is missing",
                "Model did not call a tool — cannot test",
            )
            return

        print(f"  Step 1: Got tool call: {tool_use_block.name}(id={tool_use_block.id})")

        assistant_content = []
        for block in resp1.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        rate_limit()

        # Step 2: Send plain text instead of tool_result
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="You are a helpful weather assistant.",
            tools=[SIMPLE_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "Never mind, just say hello."},
            ],
        )
        log_verbose("resp2", resp2)

        response_text = ""
        for block in resp2.content:
            if block.type == "text":
                response_text += block.text
        print(f"  Step 2: Plain text was ACCEPTED (no tool_result required)")
        print(f"  Response: {response_text[:100]}")
        record(
            "7: Missing tool_result",
            "INFO",
            "Should error when tool_result is missing after tool_use",
            "ACCEPTED — MiniMax does not enforce tool_result after tool_use",
        )
    except anthropic.BadRequestError as e:
        err_msg = str(e)
        print(f"  Step 2: Plain text was REJECTED (tool_result required)")
        print(f"  Error message: {err_msg[:300]}")
        record(
            "7: Missing tool_result",
            "PASS",
            "Should error when tool_result is missing after tool_use",
            f"Rejected with: {err_msg[:200]}",
            details=err_msg,
        )
    except Exception as e:
        err_msg = str(e)
        err_type = type(e).__name__
        print(f"  Step 2: Got {err_type}")
        print(f"  Error message: {err_msg[:300]}")
        record(
            "7: Missing tool_result",
            "INFO",
            "Should error when tool_result is missing after tool_use",
            f"{err_type}: {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 8: Structured Output via Forced Tool Call
# ---------------------------------------------------------------------------


def test_8_structured_output():
    """Test json_schema -> forced tool call pattern (how planner works)."""
    print("\n--- Test 8: Structured Output via Forced Tool Call ---")
    client = make_client()

    schema_tool = {
        "name": "structured_output",
        "description": "Return the structured response matching the required schema.",
        "input_schema": {
            "type": "object",
            "title": "structured_output",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["description", "priority"],
                    },
                }
            },
            "required": ["tasks"],
        },
    }

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system="You are a task planner. Break down the user's request into structured tasks.",
            tools=[schema_tool],
            tool_choice={"type": "tool", "name": "structured_output"},
            messages=[
                {"role": "user", "content": "I need to: 1) fetch solar wind data, 2) plot it, 3) analyze trends."}
            ],
        )
        log_verbose("resp", resp)

        # Check where the JSON ended up
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "input": block.input,
                    "id": block.id,
                })

        if tool_calls:
            tc = tool_calls[0]
            print(f"  JSON in tool_calls[0].input (name={tc['name']})")
            print(f"  Data: {json.dumps(tc['input'], indent=2)[:500]}")
            json_location = "tool_calls[0].input"
        elif text_parts:
            combined = "\n".join(text_parts)
            print(f"  JSON in response.text")
            print(f"  Text: {combined[:500]}")
            json_location = "response.text"
        else:
            print(f"  No text or tool_calls in response")
            json_location = "empty"

        record(
            "8: Structured Output (Forced Tool)",
            "PASS",
            "JSON should be in tool_calls[0].input (like Anthropic)",
            f"JSON location: {json_location}",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  FAILED: {err_msg[:300]}")
        record(
            "8: Structured Output (Forced Tool)",
            "FAIL",
            "JSON should be in tool_calls[0].input",
            f"FAILED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 9: Consecutive Same-Role Messages
# ---------------------------------------------------------------------------


def test_9_consecutive_same_role():
    """Send two consecutive user messages without merging."""
    print("\n--- Test 9: Consecutive Same-Role Messages ---")
    client = make_client()

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        log_verbose("resp", resp)

        response_text = ""
        for block in resp.content:
            if block.type == "text":
                response_text += block.text
        print(f"  Consecutive user messages ACCEPTED")
        print(f"  Response: {response_text[:150]}")
        record(
            "9: Consecutive Same-Role Messages",
            "INFO",
            "Anthropic rejects this (strict alternation). MiniMax may accept it.",
            f"ACCEPTED — response: {response_text[:100]}",
        )
    except anthropic.BadRequestError as e:
        err_msg = str(e)
        print(f"  Consecutive user messages REJECTED")
        print(f"  Error: {err_msg[:300]}")
        record(
            "9: Consecutive Same-Role Messages",
            "INFO",
            "Anthropic rejects this (strict alternation). MiniMax may accept it.",
            f"REJECTED — {err_msg[:200]}",
            details=err_msg,
        )
    except Exception as e:
        err_msg = str(e)
        err_type = type(e).__name__
        print(f"  Got {err_type}: {err_msg[:300]}")
        record(
            "9: Consecutive Same-Role Messages",
            "INFO",
            "Anthropic rejects this (strict alternation). MiniMax may accept it.",
            f"{err_type}: {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Test 10: Usage Metadata / Cache Fields
# ---------------------------------------------------------------------------


def test_10_usage_metadata():
    """Check what cache-related and other fields exist in response.usage."""
    print("\n--- Test 10: Usage Metadata / Cache Fields ---")
    client = make_client()

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        log_verbose("resp", resp)

        usage = resp.usage
        fields = {}

        # Standard fields
        for field in [
            "input_tokens",
            "output_tokens",
        ]:
            fields[field] = getattr(usage, field, "MISSING")

        # Cache-related fields
        for field in [
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ]:
            val = getattr(usage, field, "MISSING")
            fields[field] = val

        # Thinking-related fields
        for field in [
            "thinking_tokens",
        ]:
            val = getattr(usage, field, "MISSING")
            fields[field] = val

        print(f"  Usage fields:")
        for k, v in fields.items():
            present = "PRESENT" if v != "MISSING" else "MISSING"
            print(f"    {k}: {v} ({present})")

        # Check the raw dict if available
        try:
            raw_dict = json.loads(usage.model_dump_json())
            print(f"  Raw usage dict keys: {sorted(raw_dict.keys())}")
            extra_keys = set(raw_dict.keys()) - set(fields.keys())
            if extra_keys:
                print(f"  Extra keys not checked: {extra_keys}")
                for k in extra_keys:
                    fields[k] = raw_dict[k]
        except Exception:
            pass

        record(
            "10: Usage Metadata",
            "PASS",
            "Check which cache/thinking fields exist in usage",
            f"Fields: {json.dumps(fields, default=str)}",
        )
    except Exception as e:
        err_msg = str(e)
        print(f"  FAILED: {err_msg[:300]}")
        record(
            "10: Usage Metadata",
            "FAIL",
            "Check which cache/thinking fields exist",
            f"FAILED — {err_msg[:200]}",
            details=err_msg,
        )


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


def print_summary():
    print("\n")
    print("=" * 80)
    print("  MINIMAX vs ANTHROPIC API BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Model: {MODEL}")
    print(f"  Endpoint: {BASE_URL}")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for r in results:
        status_icon = {
            "PASS": "OK  ",
            "FAIL": "FAIL",
            "INFO": "INFO",
            "ERROR": "ERR ",
        }.get(r.status, "????")

        print(f"\n  [{status_icon}] {r.name}")
        print(f"         Expected: {r.expected}")
        print(f"         Actual:   {r.actual}")
        if r.details and VERBOSE:
            print(f"         Details:  {r.details[:500]}")

    # Actionable summary
    print("\n" + "=" * 80)
    print("  ACTIONABLE FINDINGS")
    print("=" * 80)

    # Thinking blocks
    t1 = next((r for r in results if r.name.startswith("1:")), None)
    t2 = next((r for r in results if r.name.startswith("2:")), None)
    if t1 and "ACCEPTED" in t1.actual:
        print("\n  [Thinking Blocks]")
        print("    MiniMax ACCEPTS thinking blocks with signatures in history.")
        print("    -> Stripping may be unnecessary (but still safe to do).")
    elif t1 and "REJECTED" in t1.actual:
        print("\n  [Thinking Blocks]")
        print("    MiniMax REJECTS thinking blocks in history.")
        print("    -> Current stripping workaround is REQUIRED.")

    # Cache control
    t3 = next((r for r in results if r.name.startswith("3:")), None)
    if t3 and "ACCEPTED" in t3.actual:
        print("\n  [Prompt Caching]")
        print("    MiniMax ACCEPTS cache_control markers (may silently ignore).")
        print("    -> No need to strip cache_control for MiniMax.")
    elif t3 and "REJECTED" in t3.actual:
        print("\n  [Prompt Caching]")
        print("    MiniMax REJECTS cache_control markers.")
        print("    -> Must strip cache_control for MiniMax adapter.")

    # Tool result errors
    t6 = next((r for r in results if r.name.startswith("6:")), None)
    t7 = next((r for r in results if r.name.startswith("7:")), None)
    if t6 or t7:
        print("\n  [Tool Use Errors]")
        if t6:
            if "ACCEPTED" in t6.actual:
                print("    Wrong tool_use_id: ACCEPTED (no validation)")
            else:
                print(f"    Wrong tool_use_id error: {t6.actual[:150]}")
        if t7:
            if "ACCEPTED" in t7.actual:
                print("    Missing tool_result: ACCEPTED (no enforcement)")
            else:
                print(f"    Missing tool_result error: {t7.actual[:150]}")
        print("    -> Note any MiniMax-specific history validation patterns if needed.")

    # Structured output
    t8 = next((r for r in results if r.name.startswith("8:")), None)
    if t8:
        print("\n  [Structured Output]")
        print(f"    {t8.actual}")
        if "tool_calls" in t8.actual:
            print("    -> Same as Anthropic: JSON in tool_calls[0].input. No changes needed.")
        elif "response.text" in t8.actual:
            print("    -> DIFFERENT: JSON in response.text, not tool_calls. Planner needs fix.")

    # Alternation
    t9 = next((r for r in results if r.name.startswith("9:")), None)
    if t9:
        print("\n  [Message Alternation]")
        if "ACCEPTED" in t9.actual:
            print("    MiniMax is LENIENT — accepts consecutive same-role messages.")
            print("    -> _ensure_alternation() is still good practice but not required for MiniMax.")
        else:
            print("    MiniMax enforces STRICT alternation (same as Anthropic).")
            print("    -> _ensure_alternation() is required for both.")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


ALL_TESTS = [
    test_1_thinking_block_replay,
    test_2_thinking_stripped,
    test_3_cache_control,
    test_4_no_cache_control,
    test_5_tool_result_correct,
    test_6_tool_result_wrong_id,
    test_7_tool_result_missing,
    test_8_structured_output,
    test_9_consecutive_same_role,
    test_10_usage_metadata,
]


def main():
    global VERBOSE, DELAY

    parser = argparse.ArgumentParser(
        description="Benchmark MiniMax vs Anthropic API behavioral differences"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full API responses")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls (default: 2)")
    parser.add_argument("--test", type=int, help="Run only a specific test number (1-10)")
    args = parser.parse_args()

    VERBOSE = args.verbose
    DELAY = args.delay

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("ERROR: MINIMAX_API_KEY not set in .env or environment")
        sys.exit(1)

    print("=" * 80)
    print("  MINIMAX vs ANTHROPIC API BENCHMARK")
    print("=" * 80)
    print(f"  Model: {MODEL}")
    print(f"  Endpoint: {BASE_URL}")
    print(f"  Delay between calls: {DELAY}s")
    print(f"  Verbose: {VERBOSE}")
    print("=" * 80)

    if args.test:
        if 1 <= args.test <= len(ALL_TESTS):
            tests_to_run = [ALL_TESTS[args.test - 1]]
        else:
            print(f"ERROR: --test must be between 1 and {len(ALL_TESTS)}")
            sys.exit(1)
    else:
        tests_to_run = ALL_TESTS

    for i, test_fn in enumerate(tests_to_run):
        try:
            test_fn()
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.")
            break
        except Exception as e:
            test_name = test_fn.__name__
            print(f"\n  UNHANDLED ERROR in {test_name}: {e}")
            traceback.print_exc()
            record(
                test_name,
                "ERROR",
                "Should not crash",
                f"Unhandled {type(e).__name__}: {str(e)[:200]}",
                details=traceback.format_exc(),
            )

        # Rate limit between tests
        if i < len(tests_to_run) - 1:
            rate_limit()

    print_summary()


if __name__ == "__main__":
    main()
