#!/usr/bin/env python
"""Full session test using the API to trigger history desync."""

import requests
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


def check_server():
    """Check if the backend server is running."""
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def send_message(message: str) -> dict:
    """Send a message to the agent via API."""
    resp = requests.post(
        "http://localhost:8000/chat",
        json={"message": message, "stream": False},
        timeout=120,
    )
    return resp.json()


def run_api_session():
    """Run a session via the API."""
    print("Checking server...")
    if not check_server():
        print("Server not running! Starting it...")

        # Start the server
        import subprocess
        import os
(REPO)
        subprocess.Popen(
            ["venv/bin/python", "-m", "uvicorn", "api.server        os.chdir:app", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to start
        for i in range(20):
            time.sleep(1)
            if check_server():
                print("Server started!")
                break
        else:
            print("Server failed to start!")
            return

    print("Server is running!")

    # Test queries that trigger tool calls
    test_queries = [
        "What data is available?",
        "Show me PSP datasets",
        "What does ACE have?",
        "List parameters for PSP_FLD_L2_MAG_RTN_1MIN",
        "What missions are available?",
        "Show me available data sources",
    ]

    for i, query in enumerate(test_queries * 3):
        print(f"\n=== Turn {i+1}: {query} ===")

        try:
            result = send_message(query)
            print(f"Response: {result.get('text', '')[:200]}...")

            if 'error' in result:
                print(f"ERROR: {result.get('error')}")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(1)

    print("\n=== Done ===")


if __name__ == "__main__":
    run_api_session()
