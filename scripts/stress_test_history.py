#!/usr/bin/env python
"""Aggressive stress test for history desync.

This creates multiple agents and hammers them with requests to try to trigger
the race condition that causes error 2013.
"""

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from agent.data_ops_agent import DataOpsAgent
from agent.event_bus import get_event_bus
from agent.logging import setup_logging, get_logger
from agent.memory_agent import MemoryAgent
from agent.viz_plotly_agent import VizPlotlyAgent

logger = get_logger()


# Sample requests that trigger tool calls
REQUESTS = [
    "list available data",
    "show me the columns of the data",
    "describe the data",
    "what data do we have?",
    "run df.head()",
    "run df.describe()",
    "calculate the mean of the first column",
    "show data summary",
    "list fetched data",
    "what datasets are loaded?",
]


def create_agent(agent_type: str, session_id: str):
    """Create an agent of the specified type."""
    event_bus = get_event_bus()

    if agent_type == "dataops":
        return DataOpsAgent(session_id=session_id, event_bus=event_bus)
    elif agent_type == "viz":
        return VizPlotlyAgent(session_id=session_id, event_bus=event_bus)
    elif agent_type == "memory":
        return MemoryAgent(session_id=session_id, event_bus=event_bus)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_single_request(agent_type: str, request: str, iteration: int, results: dict):
    """Run a single request and track results."""
    session_id = f"stress_{agent_type}_{iteration}_{random.randint(1000, 9999)}"

    try:
        agent = create_agent(agent_type, session_id)
        response = agent.send_message(request)
        results["success"] += 1
        return {"status": "success", "text": response.get("text", "")[:100]}
    except Exception as e:
        error_msg = str(e)
        results["errors"] += 1

        if "tool call result does not follow" in error_msg:
            results["desync_errors"] += 1
            # Save debug info
            debug_file = REPO / f"/tmp/desync_stress_{iteration}.json"
            with open(debug_file, "w") as f:
                json.dump({
                    "iteration": iteration,
                    "agent_type": agent_type,
                    "request": request,
                    "error": error_msg,
                }, f, indent=2)
            return {"status": "desync", "error": error_msg}

        return {"status": "error", "error": error_msg}


def run_stress_test(
    agent_type: str = "dataops",
    count: int = 50,
    concurrency: int = 3,
    delay: float = 0.5,
):
    """Run stress test with multiple concurrent requests."""
    setup_logging(verbose=False)
    logger.info(f"Starting stress test: {count} requests, concurrency={concurrency}")

    results = {"success": 0, "errors": 0, "desync_errors": 0}
    results_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []

        for i in range(count):
            request = random.choice(REQUESTS)
            future = executor.submit(
                run_single_request,
                agent_type,
                request,
                i,
                results,
            )
            futures.append(future)

            # Small delay between submissions
            time.sleep(delay)

        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result["status"] == "success":
                    print(".", end="", flush=True)
                elif result["status"] == "desync":
                    print("D", end="", flush=True)
                else:
                    print("E", end="", flush=True)
            except Exception as e:
                print(f"X", end="", flush=True)
                print(f"\nFuture error: {e}")

    print(f"\n=== Results ===")
    print(f"Total: {count}")
    print(f"Success: {results['success']}")
    print(f"Errors: {results['errors']}")
    print(f"Desync errors: {results['desync_errors']}")

    return results


def run_sequential_test(agent_type: str = "dataops", count: int = 30):
    """Run sequential requests to same agent (more likely to build up history issues)."""
    setup_logging(verbose=False)
    logger.info(f"Starting sequential test: {count} requests to same agent")

    session_id = f"seq_{agent_type}_{int(time.time())}"
    agent = create_agent(agent_type, session_id)

    results = {"success": 0, "errors": 0, "desync_errors": 0}

    for i in range(count):
        request = random.choice(REQUESTS)
        print(f"\n[{i+1}/{count}] {request[:50]}")

        try:
            response = agent.send_message(request)
            results["success"] += 1
            print(f"  -> success")

            # Check history
            if hasattr(agent, '_chat') and agent._chat:
                history = agent._chat.get_history()
                print(f"  -> history length: {len(history)}")

        except Exception as e:
            error_msg = str(e)
            results["errors"] += 1
            print(f"  -> ERROR: {error_msg[:100]}")

            if "tool call result does not follow" in error_msg:
                results["desync_errors"] += 1
                print(f"  -> DESYNC ERROR!")

                # Save debug
                debug_file = REPO / f"/tmp/desync_seq_{i}.json"
                with open(debug_file, "w") as f:
                    json.dump({
                        "iteration": i,
                        "request": request,
                        "error": error_msg,
                        "history": agent._chat.get_history() if hasattr(agent, '_chat') and agent._chat else None,
                    }, f, indent=2, default=str)

                # Try to recover by continuing (agent will have reset)
                try:
                    agent = create_agent(agent_type, session_id)
                    print(f"  -> Recovered with new agent")
                except:
                    print(f"  -> Failed to recover")

        time.sleep(0.3)

    print(f"\n=== Sequential Results ===")
    print(f"Total: {count}")
    print(f"Success: {results['success']}")
    print(f"Errors: {results['errors']}")
    print(f"Desync errors: {results['desync_errors']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test for history desync")
    parser.add_argument("--type", choices=["dataops", "viz", "memory"], default="dataops")
    parser.add_argument("--count", type=int, default=30, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests (1 = sequential)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between requests")
    args = parser.parse_args()

    if args.concurrency > 1:
        run_stress_test(args.type, args.count, args.concurrency, args.delay)
    else:
        run_sequential_test(args.type, args.count)
