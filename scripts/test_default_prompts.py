#!/usr/bin/env python3
"""
Test the three default UI example prompts end-to-end.

These are open-ended science discovery queries that exercise the full agent
pipeline: search → fetch → compute → plot → explain.

Usage:
    python scripts/test_default_prompts.py              # all 3 prompts
    python scripts/test_default_prompts.py --verbose     # with tool call logging
    python scripts/test_default_prompts.py --test 2      # single prompt
    python scripts/test_default_prompts.py --no-server   # use already-running server
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_server import send_msg, recv_msg, PORT_FILE, _cleanup_port_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLOTS_DIR = PROJECT_ROOT / "tests" / "plots"
SERVER_STARTUP_TIMEOUT = 60  # seconds
REQUEST_TIMEOUT = 600  # 10 min per request — these are complex multi-step prompts


# ---------------------------------------------------------------------------
# Server management (reused from run_agent_tests.py)
# ---------------------------------------------------------------------------

def start_server(verbose: bool = False) -> subprocess.Popen:
    """Launch agent_server.py serve as a subprocess, wait for port file."""
    _cleanup_port_file()

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "agent_server.py"), "serve"]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    print("Starting agent server...", end="", flush=True)
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read()
            print(f"\nServer exited unexpectedly (code {proc.returncode}):")
            print(output)
            sys.exit(1)
        if PORT_FILE.exists():
            try:
                port = int(PORT_FILE.read_text().strip())
                if port > 0:
                    print(f" ready (port {port})")
                    return proc
            except (ValueError, OSError):
                pass
        time.sleep(1)
        print(".", end="", flush=True)

    print("\nTimeout waiting for server to start.")
    proc.kill()
    sys.exit(1)


def _connect() -> socket.socket:
    """Connect to the running server."""
    port = int(PORT_FILE.read_text().strip())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(REQUEST_TIMEOUT)
    sock.connect(("127.0.0.1", port))
    return sock


def send(message: str) -> dict:
    """Send a message and return the full response dict."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "send", "message": message})
        response = recv_msg(sock)
    finally:
        sock.close()
    return response or {"response": "", "error": "No response", "tool_calls": []}


def reset():
    """Reset the agent conversation."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "reset"})
        recv_msg(sock)
    finally:
        sock.close()


def stop_server():
    """Send stop command to the server."""
    try:
        sock = _connect()
        try:
            send_msg(sock, {"action": "stop"})
            recv_msg(sock)
        finally:
            sock.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

class TestResults:
    def __init__(self):
        self.scenarios = []
        self._current_checks = []
        self._current_name = ""

    def start_scenario(self, name: str):
        self._current_name = name
        self._current_checks = []

    def check(self, label: str, condition: bool, detail: str = ""):
        self._current_checks.append((label, condition, detail))
        marker = "  [+]" if condition else "  [-]"
        print(f"{marker} {label}" + (f"  ({detail})" if detail and not condition else ""))

    def end_scenario(self):
        self.scenarios.append((self._current_name, list(self._current_checks)))

    def summary(self) -> str:
        lines = ["\n=== Default Prompt Test Report ==="]
        total_pass = 0
        total_checks = 0
        for name, checks in self.scenarios:
            passed = sum(1 for _, ok, _ in checks if ok)
            total = len(checks)
            total_pass += passed
            total_checks += total
            status = "PASS" if passed == total else "FAIL"
            lines.append(f"  {name:<50s} {status} ({passed}/{total} checks)")
        lines.append("===")
        lines.append(f"Total: {total_pass}/{total_checks} checks passed")
        return "\n".join(lines)

    @property
    def all_passed(self) -> bool:
        return all(ok for _, checks in self.scenarios for _, ok, _ in checks)


def tool_names(response: dict) -> list[str]:
    return [tc["name"] for tc in response.get("tool_calls", [])]


def has_tool(response: dict, name: str) -> bool:
    return name in tool_names(response)


def has_any_tool(response: dict, names: set[str]) -> bool:
    return bool(set(tool_names(response)) & names)


def resp_text(response: dict) -> str:
    return (response.get("response") or "").lower()


def resp_error(response: dict) -> str | None:
    return response.get("error")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

FETCH_TOOLS = {"fetch_data", "delegate_to_mission"}
VIZ_TOOLS = {"plot_data", "delegate_to_visualization"}
SEARCH_TOOLS = {"search_datasets", "browse_datasets", "search_full_catalog", "google_search"}


def test_1_voyager(results: TestResults):
    """Prompt 1: Voyager 1 heliopause crossing."""
    results.start_scenario("Prompt 1: Voyager 1 Heliopause")
    print("\n--- Prompt 1: How did scientists prove Voyager 1 left the solar system? ---")
    reset()

    print("\n  Sending prompt...")
    t0 = time.time()
    r = send("How did scientists prove Voyager 1 left the solar system? Show me the data.")
    elapsed = round(time.time() - t0, 1)
    print(f"  Response received in {elapsed}s")

    text = resp_text(r)
    tools = tool_names(r)
    print(f"  Tools called: {tools}")
    print(f"  Response length: {len(text)} chars")
    print(f"  Response preview: {text[:300]}...")

    # Check 1: No crash
    results.check(
        "No error",
        resp_error(r) is None,
        str(resp_error(r)),
    )

    # Check 2: Tools were called
    results.check(
        "Agent called tools",
        len(tools) > 0,
        f"tools: {tools}",
    )

    # Check 3: Data fetch attempted (delegate_to_mission, fetch_data, or search)
    results.check(
        "Data access attempted",
        has_any_tool(r, FETCH_TOOLS | SEARCH_TOOLS),
        f"tools: {tools}",
    )

    # Check 4: Visualization attempted
    results.check(
        "Visualization attempted",
        has_any_tool(r, VIZ_TOOLS),
        f"tools: {tools}",
    )

    # Check 5: Response mentions Voyager
    results.check(
        "Response mentions Voyager",
        "voyager" in text,
        f"snippet: {text[:200]}",
    )

    # Check 6: Response discusses the science (heliopause/interstellar)
    results.check(
        "Response discusses heliopause or interstellar space",
        any(w in text for w in ["heliopause", "interstellar", "heliosphere", "solar wind",
                                 "cosmic ray", "plasma", "electron"]),
        f"snippet: {text[:300]}",
    )

    results.end_scenario()


def test_2_psp_corona(results: TestResults):
    """Prompt 2: Parker Solar Probe corona entry."""
    results.start_scenario("Prompt 2: PSP Corona Entry")
    print("\n--- Prompt 2: When did Parker Solar Probe first enter the solar corona? ---")
    reset()

    print("\n  Sending prompt...")
    t0 = time.time()
    r = send("When did Parker Solar Probe first enter the solar corona? Show me what happened.")
    elapsed = round(time.time() - t0, 1)
    print(f"  Response received in {elapsed}s")

    text = resp_text(r)
    tools = tool_names(r)
    print(f"  Tools called: {tools}")
    print(f"  Response length: {len(text)} chars")
    print(f"  Response preview: {text[:300]}...")

    # Check 1: No crash
    results.check(
        "No error",
        resp_error(r) is None,
        str(resp_error(r)),
    )

    # Check 2: Tools were called
    results.check(
        "Agent called tools",
        len(tools) > 0,
        f"tools: {tools}",
    )

    # Check 3: Data fetch attempted
    results.check(
        "Data access attempted",
        has_any_tool(r, FETCH_TOOLS | SEARCH_TOOLS),
        f"tools: {tools}",
    )

    # Check 4: Visualization attempted
    results.check(
        "Visualization attempted",
        has_any_tool(r, VIZ_TOOLS),
        f"tools: {tools}",
    )

    # Check 5: Response mentions Parker Solar Probe
    results.check(
        "Response mentions Parker Solar Probe",
        "parker" in text or "psp" in text,
        f"snippet: {text[:200]}",
    )

    # Check 6: Response discusses corona entry
    results.check(
        "Response discusses corona or Alfvén surface",
        any(w in text for w in ["corona", "alfvén", "alfven", "perihelion",
                                 "sun", "solar", "magnetic"]),
        f"snippet: {text[:300]}",
    )

    results.end_scenario()


def test_3_cme(results: TestResults):
    """Prompt 3: CME hitting Earth."""
    results.start_scenario("Prompt 3: CME Hitting Earth")
    print("\n--- Prompt 3: Show me a powerful CME hitting Earth ---")
    reset()

    print("\n  Sending prompt...")
    t0 = time.time()
    r = send("Show me a powerful coronal mass ejection hitting Earth. What did it look like in the data?")
    elapsed = round(time.time() - t0, 1)
    print(f"  Response received in {elapsed}s")

    text = resp_text(r)
    tools = tool_names(r)
    print(f"  Tools called: {tools}")
    print(f"  Response length: {len(text)} chars")
    print(f"  Response preview: {text[:300]}...")

    # Check 1: No crash
    results.check(
        "No error",
        resp_error(r) is None,
        str(resp_error(r)),
    )

    # Check 2: Tools were called
    results.check(
        "Agent called tools",
        len(tools) > 0,
        f"tools: {tools}",
    )

    # Check 3: Data fetch attempted
    results.check(
        "Data access attempted",
        has_any_tool(r, FETCH_TOOLS | SEARCH_TOOLS),
        f"tools: {tools}",
    )

    # Check 4: Visualization attempted
    results.check(
        "Visualization attempted",
        has_any_tool(r, VIZ_TOOLS),
        f"tools: {tools}",
    )

    # Check 5: Response mentions CME or geomagnetic storm
    results.check(
        "Response mentions CME or storm",
        any(w in text for w in ["cme", "coronal mass ejection", "geomagnetic",
                                 "storm", "solar wind", "magnetic"]),
        f"snippet: {text[:200]}",
    )

    # Check 6: Response mentions a spacecraft used for observation
    earth_monitors = ["ace", "dscovr", "wind", "goes", "omni", "stereo", "soho"]
    results.check(
        "Response mentions an Earth-monitoring spacecraft",
        any(w in text for w in earth_monitors),
        f"snippet: {text[:300]}",
    )

    results.end_scenario()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    1: ("Voyager 1 Heliopause", test_1_voyager),
    2: ("PSP Corona Entry", test_2_psp_corona),
    3: ("CME Hitting Earth", test_3_cme),
}


def main():
    parser = argparse.ArgumentParser(description="Test default UI prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Start server in verbose mode")
    parser.add_argument("--test", "-t", type=int, help="Run only this test number (1-3)")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start/stop server (assume already running)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results = TestResults()
    proc = None
    start_time = time.time()

    try:
        # Start server
        if not args.no_server:
            proc = start_server(verbose=args.verbose)
        else:
            if not PORT_FILE.exists():
                print("Error: --no-server specified but server is not running.")
                sys.exit(1)
            print("Using existing server.")

        # Select tests to run
        if args.test:
            if args.test not in ALL_TESTS:
                print(f"Invalid test number: {args.test}. Choose 1-{len(ALL_TESTS)}.")
                sys.exit(1)
            tests_to_run = {args.test: ALL_TESTS[args.test]}
        else:
            tests_to_run = ALL_TESTS

        # Run tests
        for num, (name, func) in sorted(tests_to_run.items()):
            try:
                func(results)
            except Exception as e:
                print(f"\n  [!] Prompt {num} ({name}) crashed: {e}")
                results.start_scenario(f"Prompt {num}: {name}")
                results.check("Test execution", False, str(e))
                results.end_scenario()

        # Summary
        elapsed = round(time.time() - start_time, 1)
        report = results.summary()
        report += f"\nElapsed: {elapsed}s"
        print(report)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = PLOTS_DIR / f"default_prompts_report_{timestamp}.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved: {report_path}")

    finally:
        # Stop server
        if not args.no_server:
            print("\nStopping server...")
            stop_server()
            if proc:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    sys.exit(0 if results.all_passed else 1)


if __name__ == "__main__":
    main()
