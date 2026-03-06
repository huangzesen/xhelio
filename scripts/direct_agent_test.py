#!/usr/bin/env python
"""Direct sub-agent test to trigger history desync."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from agent.logging import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger()


def run_direct_test():
    """Run a direct test with sub-agents."""
    from agent.core import OrchestratorAgent

    print("Creating orchestrator...")
    orch = OrchestratorAgent(verbose=True, gui_mode=False)

    # Get the envoy agent directly
    print("Getting PSP envoy agent...")
    envoy = orch._get_or_create_envoy_agent("PSP")

    # Start the agent
    print("Starting envoy agent...")
    envoy.start()

    # Wait for agent to start
    time.sleep(1)

    # Test queries
    queries = [
        "What PSP datasets are available?",
        "Browse datasets for PSP",
        "What magnetic field data exists?",
    ]

    # Run multiple turns
    for i, query in enumerate(queries * 5):
        print(f"\n=== Turn {i+1}: {query} ===")

        try:
            # Send to envoy
            response = envoy.send(query, wait=True)
            print(f"Response: {str(response.get('text', ''))[:150]}...")

        except Exception as e:
            print(f"ERROR: {e}")

            # Check for desync error
            if "tool call result does not follow" in str(e):
                print(">>> DESYNC ERROR DETECTED!")

                # Save debug info
                debug_file = REPO / f"/tmp/desync_turn_{i}.json"
                with open(debug_file, "w") as f:
                    json.dump({
                        "turn": i,
                        "query": query,
                        "error": str(e),
                    }, f, indent=2)

        time.sleep(0.5)

    print("\n=== Done ===")


if __name__ == "__main__":
    import json
    run_direct_test()
