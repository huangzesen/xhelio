#!/usr/bin/env python
"""Full session test with real orchestrator to trigger history desync."""

import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from agent.logging import setup_logging, get_logger

setup_logging(verbose=True)
logger = get_logger()


def run_full_session():
    """Run a full session with the orchestrator."""
    from agent.core import OrchestratorAgent

    print("Creating orchestrator...")
    orch = OrchestratorAgent(verbose=True, gui_mode=False)

    print("Starting orchestrator thread...")
    orch.start()

    # Wait for startup
    time.sleep(2)

    test_queries = [
        "What data is available?",
        "Show me available PSP datasets",
        "What datasets does ACE have?",
        "List parameters for PSP_FLD_L2_MAG_RTN_1MIN",
        "What missions are available?",
        "Show me available data sources",
    ]

    for i, query in enumerate(test_queries * 3):
        print(f"\n=== Turn {i+1}: {query} ===")

        try:
            orch.input_queue.put((query, None))

            # Wait for response (with timeout)
            # The orchestrator prints to console, we just wait a bit
            time.sleep(8)

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n=== Stopping orchestrator ===")
    orch.stop()
    orch.join(timeout=10)

    print("=== Done ===")


if __name__ == "__main__":
    run_full_session()
