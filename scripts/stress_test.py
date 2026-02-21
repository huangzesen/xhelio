#!/usr/bin/env python3
"""
Stress test / exploratory test for the agent server.
Sends diverse commands and logs all results to a JSON file.
"""

import json
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_server import send_msg, recv_msg, PORT_FILE

RESULTS_FILE = PROJECT_ROOT / "tests" / "stress_test_results.json"

def connect():
    port = int(PORT_FILE.read_text().strip())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    sock.connect(("127.0.0.1", port))
    return sock

def send(message: str) -> dict:
    sock = connect()
    try:
        send_msg(sock, {"action": "send", "message": message})
        response = recv_msg(sock)
    finally:
        sock.close()
    return response or {"response": "", "error": "No response", "tool_calls": []}

def reset():
    sock = connect()
    try:
        send_msg(sock, {"action": "reset"})
        recv_msg(sock)
    finally:
        sock.close()

# ---- Test scenarios ----

TEST_COMMANDS = [
    # Group 1: Basic discovery
    {
        "group": "Discovery",
        "commands": [
            "What missions do you support?",
            "Search for solar wind speed datasets",
            "What datasets does Parker Solar Probe have?",
        ],
        "reset_before": True,
    },
    # Group 2: Edge cases - ambiguous/vague requests
    {
        "group": "Ambiguous Requests",
        "commands": [
            "Show me some data",
            "Plot something interesting",
            "What happened on the sun last week?",
        ],
        "reset_before": True,
    },
    # Group 3: Fetch + compute pipeline
    {
        "group": "Fetch & Compute",
        "commands": [
            "Fetch ACE magnetic field data for 2024-01-01 to 2024-01-07",
            "Compute the magnitude of the magnetic field",
            "Now compute a 2-hour running average of the magnitude",
            "Describe the smoothed data",
        ],
        "reset_before": True,
    },
    # Group 4: Error handling - bad inputs
    {
        "group": "Error Handling",
        "commands": [
            "Fetch data from FAKE_DATASET_999 for last week",
            "Plot data for the year 3000",
            "Compute derivative of data that doesn't exist",
            "Search for datasets from the Hubble Space Telescope",
        ],
        "reset_before": True,
    },
    # Group 5: Multi-mission & plotting
    {
        "group": "Multi-Mission Plot",
        "commands": [
            "Plot OMNI magnetic field magnitude for January 2024",
            "Now also fetch Wind magnetic field magnitude for the same period",
            "Can you overlay both datasets?",
        ],
        "reset_before": True,
    },
    # Group 6: Complex multi-step natural language
    {
        "group": "Complex NL Requests",
        "commands": [
            "I want to compare the solar wind speed from ACE and DSCOVR during a solar storm in December 2023. Can you find and plot both?",
        ],
        "reset_before": True,
    },
    # Group 7: Conversation memory & context
    {
        "group": "Context Memory",
        "commands": [
            "Fetch STEREO-A magnetic field data for February 2024",
            "What did I just fetch?",
            "Describe it",
            "Plot it",
        ],
        "reset_before": True,
    },
    # Group 8: Time parsing edge cases
    {
        "group": "Time Parsing",
        "commands": [
            "Show me ACE data for yesterday",
            "Fetch MMS magnetic field data for the last 3 hours",
            "Plot Wind data from Christmas 2023 to New Year 2024",
        ],
        "reset_before": True,
    },
]


def main():
    all_results = []
    total_start = time.time()

    for group_idx, group in enumerate(TEST_COMMANDS):
        group_name = group["group"]
        print(f"\n{'='*60}")
        print(f"  Group {group_idx+1}: {group_name}")
        print(f"{'='*60}")

        if group.get("reset_before"):
            reset()
            print("  [reset conversation]")

        group_results = []

        for cmd_idx, cmd in enumerate(group["commands"]):
            print(f"\n  [{group_idx+1}.{cmd_idx+1}] Sending: {cmd}")
            start = time.time()

            try:
                r = send(cmd)
                elapsed = round(time.time() - start, 2)

                result = {
                    "group": group_name,
                    "command": cmd,
                    "response": r.get("response", ""),
                    "error": r.get("error"),
                    "tool_calls": [tc["name"] for tc in r.get("tool_calls", [])],
                    "tool_details": r.get("tool_calls", []),
                    "elapsed": elapsed,
                    "tokens": r.get("tokens", {}),
                    "timestamp": datetime.now().isoformat(),
                }

                # Print summary
                resp_preview = (r.get("response") or "")[:200]
                tools = result["tool_calls"]
                error = r.get("error")

                if error:
                    print(f"  ERROR: {error}")
                print(f"  Tools: {', '.join(tools) if tools else 'none'}")
                print(f"  Response: {resp_preview}...")
                print(f"  Time: {elapsed}s")

                group_results.append(result)

            except Exception as e:
                print(f"  EXCEPTION: {e}")
                group_results.append({
                    "group": group_name,
                    "command": cmd,
                    "response": "",
                    "error": str(e),
                    "tool_calls": [],
                    "tool_details": [],
                    "elapsed": round(time.time() - start, 2),
                    "tokens": {},
                    "timestamp": datetime.now().isoformat(),
                })

        all_results.extend(group_results)

    total_elapsed = round(time.time() - total_start, 1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    errors = [r for r in all_results if r["error"]]
    slow = [r for r in all_results if r["elapsed"] > 60]
    no_tools = [r for r in all_results if not r["tool_calls"] and not r["error"]]

    print(f"  Total commands: {len(all_results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Slow (>60s): {len(slow)}")
    print(f"  No tool calls: {len(no_tools)}")
    print(f"  Total time: {total_elapsed}s")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_commands": len(all_results),
        "total_elapsed": total_elapsed,
        "error_count": len(errors),
        "slow_count": len(slow),
        "no_tool_count": len(no_tools),
        "results": all_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
