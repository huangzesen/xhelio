#!/usr/bin/env python
"""Debug script to reproduce history desync errors.

Usage:
    python scripts/debug_history_desync.py [--count N] [--delay SECONDS]

This script sends random requests to a DataOpsAgent to trigger the
"tool call result does not follow tool call" (error 2013) issue.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add repo to path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from agent.data_ops_agent import DataOpsAgent
from agent.event_bus import get_event_bus
from agent.logging import setup_logging, get_logger

logger = get_logger()


def random_code_snippet() -> str:
    """Generate random but valid-ish pandas code."""
    operations = [
        "df.head()",
        "df.tail()",
        "df.describe()",
        "df.shape",
        "df.columns.tolist()",
        "len(df)",
        "df.dtypes",
        "df.isnull().sum()",
        "df.dropna()",
        "df.fillna(0)",
        "df['new_col'] = df.iloc[:, 0] * 2",
        "df.sort_values(df.columns[0])",
        "df.mean()",
        "df.std()",
    ]
    return random.choice(operations)


def run_test(count: int = 20, delay: float = 1.0):
    """Run the debug test."""
    setup_logging(verbose=True)
    logger.info(f"Starting history desync debug test: {count} iterations")

    event_bus = get_event_bus()

    # Create a DataOpsAgent
    agent = DataOpsAgent(
        session_id="debug_desync_test",
        event_bus=event_bus,
    )

    # Track errors
    errors = []
    success_count = 0

    for i in range(count):
        print(f"\n--- Iteration {i+1}/{count} ---")

        # Generate a random code snippet
        code = random_code_snippet()

        try:
            # Send a random request that will trigger tool execution
            # We need to trigger actual tool calls
            request = f"Run this code: {code}"

            # Also try triggering different tools
            tool_triggers = [
                f"Run this code: {code}",
                "list available data",
                "describe the data",
                "show me the columns",
            ]
            request = random.choice(tool_triggers)

            response = agent.send_message(request)
            print(f"Response: {response.get('text', '')[:200]}...")
            success_count += 1

        except Exception as e:
            err_msg = str(e)
            print(f"ERROR: {err_msg}")
            errors.append((i, err_msg))

            # Check if it's the desync error
            if "tool call result does not follow tool call" in err_msg:
                print(">>> HIT HISTORY DESYNC ERROR! <<<")
                # Save debug info
                debug_file = REPO / f"/tmp/desync_debug_{i}.json"
                debug_file.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_file, "w") as f:
                    json.dump({
                        "iteration": i,
                        "request": request,
                        "error": err_msg,
                        "history": agent._chat.get_history() if hasattr(agent, '_chat') and agent._chat else None,
                    }, f, indent=2, default=str)
                print(f"Saved debug info to {debug_file}")

        time.sleep(delay)

    print(f"\n=== Results ===")
    print(f"Total: {count}")
    print(f"Success: {success_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for i, err in errors:
            print(f"  [{i}] {err[:100]}")

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug history desync errors")
    parser.add_argument("--count", type=int, default=20, help="Number of iterations")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    args = parser.parse_args()

    run_test(args.count, args.delay)
