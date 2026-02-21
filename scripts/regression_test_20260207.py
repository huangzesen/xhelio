#!/usr/bin/env python3
"""
Regression Test Suite — 2026-02-07 Issue Fixes

Multi-turn conversation scenarios that stress-test the 15 fixes from commit
87114d3.  Each scenario is a coherent 3-10 turn conversation exercising
delegation chains, label propagation, compute pipelines, and Autoplot state.

Usage:
    python scripts/regression_test_20260207.py                 # full run
    python scripts/regression_test_20260207.py --no-server      # existing server
    python scripts/regression_test_20260207.py --scenario 2     # single scenario
    python scripts/regression_test_20260207.py --verbose         # verbose server
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

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_server import send_msg, recv_msg, PORT_FILE, _cleanup_port_file

PLOTS_DIR = PROJECT_ROOT / "tests" / "plots"
SERVER_STARTUP_TIMEOUT = 90   # seconds — JVM can be slow
REQUEST_TIMEOUT = 300         # 5 min per request


# ============================================================================
# Server management  (reused from run_agent_tests.py)
# ============================================================================

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


def server_alive() -> bool:
    """Check if the server is still responding."""
    try:
        r = send("ping")
        return r.get("error") is None or r.get("response", "") != ""
    except Exception:
        return False


# ============================================================================
# Test infrastructure
# ============================================================================

class TestResults:
    """Collects per-turn, per-check results across all scenarios."""

    def __init__(self):
        self.scenarios: list[dict] = []
        self._current: dict | None = None

    def start_scenario(self, name: str, description: str, issues: list[str]):
        self._current = {
            "name": name,
            "description": description,
            "issues_tested": issues,
            "turns": [],
            "started_at": datetime.now().isoformat(),
        }

    def start_turn(self, turn_num: int, message: str):
        self._current_turn = {
            "turn": turn_num,
            "message": message,
            "checks": [],
            "response": None,
        }

    def record_response(self, response: dict):
        self._current_turn["response"] = {
            "text": (response.get("response") or "")[:500],
            "error": response.get("error"),
            "tool_calls": [tc["name"] for tc in response.get("tool_calls", [])],
            "elapsed": response.get("elapsed", 0),
        }

    def check(self, label: str, passed: bool, detail: str = ""):
        self._current_turn["checks"].append({
            "label": label,
            "passed": passed,
            "detail": detail,
        })
        marker = "    [PASS]" if passed else "    [FAIL]"
        line = f"{marker} {label}"
        if detail and not passed:
            line += f"  ({detail})"
        print(line)

    def end_turn(self):
        self._current["turns"].append(self._current_turn)

    def end_scenario(self):
        self._current["ended_at"] = datetime.now().isoformat()
        self.scenarios.append(self._current)
        self._current = None

    @property
    def total_checks(self) -> int:
        return sum(
            len(t["checks"])
            for s in self.scenarios
            for t in s["turns"]
        )

    @property
    def total_passed(self) -> int:
        return sum(
            1 for s in self.scenarios
            for t in s["turns"]
            for c in t["checks"]
            if c["passed"]
        )

    @property
    def all_passed(self) -> bool:
        return self.total_passed == self.total_checks

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "  REGRESSION TEST REPORT — 2026-02-07 Fixes", "=" * 60]
        for s in self.scenarios:
            turns = s["turns"]
            passed = sum(1 for t in turns for c in t["checks"] if c["passed"])
            total = sum(len(t["checks"]) for t in turns)
            status = "PASS" if passed == total else "FAIL"
            issues = ", ".join(s["issues_tested"]) if s["issues_tested"] else "general"
            lines.append(f"  [{status}] {s['name']:<45s} {passed}/{total}  ({issues})")
        lines.append("-" * 60)
        lines.append(f"  Total: {self.total_passed}/{self.total_checks} checks passed")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "suite": "regression_test_20260207",
            "timestamp": datetime.now().isoformat(),
            "total_checks": self.total_checks,
            "total_passed": self.total_passed,
            "all_passed": self.all_passed,
            "scenarios": self.scenarios,
        }


# ============================================================================
# Check helpers
# ============================================================================

def tool_names(response: dict) -> list[str]:
    return [tc["name"] for tc in response.get("tool_calls", [])]


def has_tool(response: dict, name: str) -> bool:
    return name in tool_names(response)


def resp_text(response: dict) -> str:
    return (response.get("response") or "").lower()


def resp_error(response: dict) -> str | None:
    return response.get("error")


def text_contains_any(text: str, needles: list[str]) -> bool:
    text_lower = text.lower()
    return any(n.lower() in text_lower for n in needles)


# ============================================================================
# Scenario definitions
# ============================================================================

def scenario_1_ace_analysis(results: TestResults):
    """Full ACE Analysis Pipeline — fetch, compute, plot, customize, export."""
    results.start_scenario(
        "Scenario 1: Full ACE Analysis Pipeline",
        "Multi-step fetch → compute → plot → customize → export",
        ["ISSUE-03", "ISSUE-05", "ISSUE-07", "ISSUE-09", "ISSUE-10", "ISSUE-11", "ISSUE-12"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 1: Full ACE Analysis Pipeline")
    print("  Issues: ISSUE-03, -05, -07, -09, -10, -11, -12")
    print("=" * 60)
    reset()

    # Turn 1: Fetch ACE vector data
    print("\n  Turn 1: Fetch ACE magnetic field GSE components")
    results.start_turn(1, "Fetch ACE magnetic field GSE components (AC_H2_MFI, BGSEc) for January 15-20, 2024")
    r = send("Fetch ACE magnetic field GSE components (AC_H2_MFI, BGSEc) for January 15-20, 2024")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response confirms data stored (ISSUE-09: labels)",
        text_contains_any(resp_text(r), ["stored", "AC_H2_MFI", "label", "points", "fetched"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 2: 2-hour running average
    print("\n  Turn 2: Compute 2-hour running average (ISSUE-03: DatetimeIndex)")
    results.start_turn(2, "Compute a 2-hour running average of the data you just fetched")
    r = send("Compute a 2-hour running average of the data you just fetched")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "no 'not compatible' error (ISSUE-03 fix)",
        not text_contains_any(resp_text(r), ["not compatible"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 3: Compute magnitude
    print("\n  Turn 3: Compute magnitude of original data")
    results.start_turn(3, "Also compute the magnitude of the original (non-smoothed) data")
    r = send("Also compute the magnitude of the original (non-smoothed) data")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "mentions magnitude/Bmag",
        text_contains_any(resp_text(r), ["magnitude", "bmag", "stored", "computed"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 4: Plot
    print("\n  Turn 4: Plot magnitude and smoothed data together")
    results.start_turn(4, "Plot the magnitude and the smoothed data together")
    r = send("Plot the magnitude and the smoothed data together")
    results.record_response(r)
    results.check("has delegate_to_visualization", has_tool(r, "delegate_to_visualization"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Set title (ISSUE-10)
    print("\n  Turn 5: Set plot title (ISSUE-10: DOM title API)")
    results.start_turn(5, "Set the title to 'ACE MFI Analysis - Jan 2024'")
    r = send("Set the title to 'ACE MFI Analysis - Jan 2024'")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "confirms title set",
        text_contains_any(resp_text(r), ["title", "set", "updated"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 6: Change render type (ISSUE-07)
    print("\n  Turn 6: Change render type to fill_to_zero (ISSUE-07: snake_case)")
    results.start_turn(6, "Change the render type to fill_to_zero")
    r = send("Change the render type to fill_to_zero")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "mentions fill/render/changed",
        text_contains_any(resp_text(r), ["fill", "render", "changed", "updated"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 7: Color table on line plot (ISSUE-12)
    print("\n  Turn 7: Set color table on line plot (ISSUE-12: informative error)")
    results.start_turn(7, "Try setting the color table to matlab_jet")
    r = send("Try setting the color table to matlab_jet")
    results.record_response(r)
    results.check(
        "informative response about color table applicability",
        text_contains_any(resp_text(r), ["spectrogram", "line", "render type", "color table", "apply", "not"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 8: Export (ISSUE-05, ISSUE-11)
    print("\n  Turn 8: Export PNG + VAP (ISSUE-05, -11: relative paths)")
    results.start_turn(8, "Export the plot as tests/plots/ace_analysis.png and save the session to tests/plots/ace_session.vap")
    r = send("Export the plot as tests/plots/ace_analysis.png and save the session to tests/plots/ace_session.vap")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "ace_analysis.png"
    vap_path = PROJECT_ROOT / "tests" / "plots" / "ace_session.vap"
    results.check("PNG file exists (ISSUE-05)", png_path.exists(), str(png_path))
    results.check("VAP file exists (ISSUE-11)", vap_path.exists(), str(vap_path))
    results.end_turn()

    results.end_scenario()


def scenario_2_crash_guard(results: TestResults):
    """CDAWeb Crash Guard + Recovery."""
    results.start_scenario(
        "Scenario 2: CDAWeb Crash Guard + Recovery",
        "Invalid parameters and 4-panel guard, verify no JVM crash",
        ["ISSUE-01", "ISSUE-02", "ISSUE-15"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 2: CDAWeb Crash Guard + Recovery")
    print("  Issues: ISSUE-01, -02, -15")
    print("=" * 60)
    reset()

    # Turn 1: Valid plot
    print("\n  Turn 1: Plot ACE magnetic field magnitude")
    results.start_turn(1, "Show me ACE solar wind magnetic field magnitude (AC_H2_MFI, Magnitude) for January 2024")
    r = send("Show me ACE solar wind magnetic field magnitude (AC_H2_MFI, Magnitude) for January 2024")
    results.record_response(r)
    results.check(
        "delegation tool called",
        has_tool(r, "delegate_to_mission") or has_tool(r, "delegate_to_visualization"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 2: Invalid parameter (ISSUE-02)
    print("\n  Turn 2: Plot with invalid parameter (ISSUE-02: no JVM crash)")
    results.start_turn(2, "Now show me the same plot but use dataset WI_H0_MFI with parameter NONEXISTENT_PARAM for the same period")
    r = send("Now show me the same plot but use dataset WI_H0_MFI with parameter NONEXISTENT_PARAM for the same period")
    results.record_response(r)
    results.check("server alive after invalid param (ISSUE-02)", server_alive(), "server ping failed")
    results.check(
        "error message in response",
        text_contains_any(resp_text(r), ["not found", "error", "available", "parameter", "invalid", "does not"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 3: Verify recovery
    print("\n  Turn 3: Check memory to verify server recovery")
    results.start_turn(3, "What data do I have in memory?")
    r = send("What data do I have in memory?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check("server alive", server_alive(), "server ping failed")
    results.end_turn()

    # Turn 4: Fetch OMNI data
    print("\n  Turn 4: Fetch OMNI data for 4-panel test")
    results.start_turn(4, "Fetch OMNI magnetic field (OMNI_HRO_1MIN, F), solar wind speed (flow_speed), and proton density (proton_density) for January 15-17, 2024")
    r = send("Fetch OMNI magnetic field (OMNI_HRO_1MIN, F), solar wind speed (flow_speed), and proton density (proton_density) for January 15-17, 2024")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: 4-panel attempt (ISSUE-01)
    print("\n  Turn 5: Request 4-panel plot (ISSUE-01: should be guarded)")
    results.start_turn(5, "Create a 4-panel dashboard with magnetic field, speed, density, and a duplicate of magnetic field, one in each panel")
    r = send("Create a 4-panel dashboard with magnetic field, speed, density, and a duplicate of magnetic field, one in each panel")
    results.record_response(r)
    results.check("server alive after 4-panel request (ISSUE-01)", server_alive(), "server ping failed")
    results.check(
        "response mentions limit or alternative",
        text_contains_any(resp_text(r), ["maximum", "3 panel", "overlay", "limit", "max", "three", "too many"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 6: 3-panel fallback
    print("\n  Turn 6: Plot 3 panels instead")
    results.start_turn(6, "OK, just plot the first three (magnetic field, speed, density) on separate panels instead")
    r = send("OK, just plot the first three (magnetic field, speed, density) on separate panels instead")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check("server alive (ISSUE-15)", server_alive(), "server ping failed")
    results.end_turn()

    results.end_scenario()


def scenario_3_mms_access(results: TestResults):
    """MMS Data Access — @0 suffix handling."""
    results.start_scenario(
        "Scenario 3: MMS Data Access",
        "MMS FGM dataset access, @0/@1 suffix, describe, plot, export",
        ["ISSUE-06", "ISSUE-08"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 3: MMS Data Access")
    print("  Issues: ISSUE-06, -08")
    print("=" * 60)
    reset()

    # Turn 1: Fetch MMS FGM
    print("\n  Turn 1: Fetch MMS1 FGM survey data (ISSUE-06: @0 suffix)")
    results.start_turn(1, "Fetch MMS1 FGM survey magnetic field data for 2024-01-10 to 2024-01-11")
    r = send("Fetch MMS1 FGM survey magnetic field data for 2024-01-10 to 2024-01-11")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response mentions MMS/stored",
        text_contains_any(resp_text(r), ["points", "stored", "mms", "magnetic", "fetched"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 2: List parameters
    print("\n  Turn 2: What parameters does this dataset have?")
    results.start_turn(2, "What parameters does this dataset have?")
    r = send("What parameters does this dataset have?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Describe
    print("\n  Turn 3: Describe the fetched data")
    results.start_turn(3, "Describe the data you fetched")
    r = send("Describe the data you fetched")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response has statistics",
        text_contains_any(resp_text(r), ["mean", "min", "max", "statistics", "points", "range"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 4: Plot
    print("\n  Turn 4: Plot it")
    results.start_turn(4, "Plot it")
    r = send("Plot it")
    results.record_response(r)
    results.check("has delegate_to_visualization", has_tool(r, "delegate_to_visualization"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Export
    print("\n  Turn 5: Export (ISSUE-05: relative path)")
    results.start_turn(5, "Export as tests/plots/mms_fgm.png")
    r = send("Export as tests/plots/mms_fgm.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "mms_fgm.png"
    results.check("PNG file exists (ISSUE-05)", png_path.exists(), str(png_path))
    results.end_turn()

    results.end_scenario()


def scenario_4_cross_mission(results: TestResults):
    """Cross-Mission Comparison — delegation chaining, label propagation."""
    results.start_scenario(
        "Scenario 4: Cross-Mission Comparison",
        "Fetch from 3 missions, overlay plot, customize, export",
        [],
    )
    print("\n" + "=" * 60)
    print("  Scenario 4: Cross-Mission Comparison")
    print("  Issues: delegation chaining, label propagation")
    print("=" * 60)
    reset()

    # Turn 1: ACE
    print("\n  Turn 1: Fetch ACE magnetic field magnitude")
    results.start_turn(1, "Fetch ACE magnetic field magnitude (AC_H2_MFI, Magnitude) for January 15-20, 2024")
    r = send("Fetch ACE magnetic field magnitude (AC_H2_MFI, Magnitude) for January 15-20, 2024")
    results.record_response(r)
    results.check(
        "data fetch tool called",
        has_tool(r, "delegate_to_mission") or has_tool(r, "fetch_data"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 2: Wind
    print("\n  Turn 2: Fetch Wind magnetic field magnitude")
    results.start_turn(2, "Also fetch Wind magnetic field magnitude (WI_H0_MFI, BF1) for the same period")
    r = send("Also fetch Wind magnetic field magnitude (WI_H0_MFI, BF1) for the same period")
    results.record_response(r)
    results.check(
        "data fetch tool called",
        has_tool(r, "delegate_to_mission") or has_tool(r, "fetch_data"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: DSCOVR
    print("\n  Turn 3: Fetch DSCOVR magnetic field magnitude")
    results.start_turn(3, "And fetch DSCOVR magnetic field magnitude (DSCOVR_H0_MAG, B1F1) for the same period")
    r = send("And fetch DSCOVR magnetic field magnitude (DSCOVR_H0_MAG, B1F1) for the same period")
    results.record_response(r)
    results.check(
        "data fetch tool called",
        has_tool(r, "delegate_to_mission") or has_tool(r, "fetch_data"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 4: List memory
    print("\n  Turn 4: What data is in memory?")
    results.start_turn(4, "What data do I have in memory right now?")
    r = send("What data do I have in memory right now?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    text = resp_text(r)
    results.check("mentions ACE data", text_contains_any(text, ["ac_h2_mfi", "ace"]), f"text: {text[:120]}")
    results.check("mentions Wind data", text_contains_any(text, ["wi_h0_mfi", "wind"]), f"text: {text[:120]}")
    results.check("mentions DSCOVR data", text_contains_any(text, ["dscovr", "b1f1"]), f"text: {text[:120]}")
    results.end_turn()

    # Turn 5: Overlay plot
    print("\n  Turn 5: Plot all three together")
    results.start_turn(5, "Plot all three magnetic field magnitudes together with title 'L1 Magnetic Field Comparison'")
    r = send("Plot all three magnetic field magnitudes together with title 'L1 Magnetic Field Comparison'")
    results.record_response(r)
    results.check("has delegate_to_visualization", has_tool(r, "delegate_to_visualization"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 6: Axis customization
    print("\n  Turn 6: Set y-axis label and range")
    results.start_turn(6, "Set the y-axis label to 'B [nT]' and the y-axis range from 0 to 25")
    r = send("Set the y-axis label to 'B [nT]' and the y-axis range from 0 to 25")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 7: Export
    print("\n  Turn 7: Export")
    results.start_turn(7, "Export to tests/plots/l1_comparison.png")
    r = send("Export to tests/plots/l1_comparison.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "l1_comparison.png"
    results.check("PNG file exists", png_path.exists(), str(png_path))
    results.end_turn()

    results.end_scenario()


def scenario_5_vector_data(results: TestResults):
    """Vector Data Handling — component access, decomposition."""
    results.start_scenario(
        "Scenario 5: Vector Data Handling",
        "Vector field fetch, component access, multi-panel, magnitude",
        ["ISSUE-04", "ISSUE-09"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 5: Vector Data Handling")
    print("  Issues: ISSUE-04, -09")
    print("=" * 60)
    reset()

    # Turn 1: Fetch vector data
    print("\n  Turn 1: Fetch ACE BGSEc vector data")
    results.start_turn(1, "Fetch ACE magnetic field GSE components (AC_H2_MFI, BGSEc) for January 15-17, 2024")
    r = send("Fetch ACE magnetic field GSE components (AC_H2_MFI, BGSEc) for January 15-17, 2024")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response mentions stored/label (ISSUE-09)",
        text_contains_any(resp_text(r), ["stored", "label", "bgsec", "fetched", "points"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 2: Plot single component
    print("\n  Turn 2: Plot Bx component (ISSUE-04: component=0)")
    results.start_turn(2, "Plot the Bx component (first component) of the data")
    r = send("Plot the Bx component (first component) of the data")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Multi-panel components
    print("\n  Turn 3: Plot all 3 components on separate panels")
    results.start_turn(3, "Now plot all three components (Bx, By, Bz) on separate panels")
    r = send("Now plot all three components (Bx, By, Bz) on separate panels")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 4: Per-panel titles
    print("\n  Turn 4: Set per-panel titles (ISSUE-10)")
    results.start_turn(4, "Set panel titles: 'Bx GSE', 'By GSE', 'Bz GSE'")
    r = send("Set panel titles: 'Bx GSE', 'By GSE', 'Bz GSE'")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Compute magnitude
    print("\n  Turn 5: Compute magnitude of vector field")
    results.start_turn(5, "Also compute the magnitude of the BGSEc vector field")
    r = send("Also compute the magnitude of the BGSEc vector field")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "mentions magnitude/bmag",
        text_contains_any(resp_text(r), ["magnitude", "bmag", "computed", "stored"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 6: Export
    print("\n  Turn 6: Export")
    results.start_turn(6, "Export to tests/plots/ace_components.png")
    r = send("Export to tests/plots/ace_components.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "ace_components.png"
    results.check("PNG file exists", png_path.exists(), str(png_path))
    results.end_turn()

    results.end_scenario()


def scenario_6_discovery(results: TestResults):
    """STEREO-A and Wind Dataset Discovery."""
    results.start_scenario(
        "Scenario 6: STEREO-A and Wind Discovery",
        "Dataset browsing, cadence awareness, correct dataset selection",
        ["ISSUE-13", "ISSUE-14"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 6: STEREO-A and Wind Discovery")
    print("  Issues: ISSUE-13, -14")
    print("=" * 60)
    reset()

    # Turn 1: STEREO-A datasets
    print("\n  Turn 1: What magnetic field datasets for STEREO-A? (ISSUE-13)")
    results.start_turn(1, "What magnetic field datasets are available for STEREO-A?")
    r = send("What magnetic field datasets are available for STEREO-A?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "mentions STEREO-A mag datasets",
        text_contains_any(resp_text(r), ["sta_l1_mag", "sta_l2_mag", "mag", "magnetic"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    # Turn 2: Fetch STEREO-A
    print("\n  Turn 2: Fetch STEREO-A magnetic field data")
    results.start_turn(2, "Fetch STEREO-A magnetic field data for January 2024")
    r = send("Fetch STEREO-A magnetic field data for January 2024")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Wind with cadence awareness
    print("\n  Turn 3: Wind 5-day fetch (ISSUE-14: cadence warning)")
    results.start_turn(3, "What about Wind? I need 5 days of magnetic field data from January 15-20, 2024")
    r = send("What about Wind? I need 5 days of magnetic field data from January 15-20, 2024")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "mentions cadence or resolution",
        text_contains_any(resp_text(r), ["wi_h0_mfi", "1-min", "cadence", "3-second", "high", "resolution", "minute"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 4: Use 1-minute
    print("\n  Turn 4: Use 1-minute resolution dataset")
    results.start_turn(4, "Use the 1-minute resolution dataset please")
    r = send("Use the 1-minute resolution dataset please")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Describe Wind data
    print("\n  Turn 5: Describe Wind data")
    results.start_turn(5, "Describe the Wind data")
    r = send("Describe the Wind data")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response has statistics",
        text_contains_any(resp_text(r), ["mean", "points", "statistics", "min", "max"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    results.end_scenario()


def scenario_7_explicit_params(results: TestResults):
    """Explicit Parameters — no unnecessary clarification."""
    results.start_scenario(
        "Scenario 7: Explicit Parameters",
        "Agent should not ask for clarification when dataset/parameter are specified",
        ["ISSUE-08"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 7: Explicit Parameters")
    print("  Issues: ISSUE-08")
    print("=" * 60)
    reset()

    # Turn 1: Explicit ACE
    print("\n  Turn 1: Fetch AC_H2_MFI Magnitude (ISSUE-08: no clarification)")
    results.start_turn(1, "Fetch dataset AC_H2_MFI parameter Magnitude for 2024-01-15 to 2024-01-20")
    r = send("Fetch dataset AC_H2_MFI parameter Magnitude for 2024-01-15 to 2024-01-20")
    results.record_response(r)
    results.check(
        "no ask_clarification (ISSUE-08)",
        not has_tool(r, "ask_clarification"),
        f"tools: {tool_names(r)}",
    )
    results.check(
        "data fetch tool called",
        has_tool(r, "delegate_to_mission") or has_tool(r, "fetch_data"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 2: Explicit OMNI
    print("\n  Turn 2: Fetch OMNI_HRO_1MIN flow_speed (ISSUE-08)")
    results.start_turn(2, "Fetch dataset OMNI_HRO_1MIN parameter flow_speed for 2024-01-15 to 2024-01-20")
    r = send("Fetch dataset OMNI_HRO_1MIN parameter flow_speed for 2024-01-15 to 2024-01-20")
    results.record_response(r)
    results.check(
        "no ask_clarification (ISSUE-08)",
        not has_tool(r, "ask_clarification"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Explicit PSP
    print("\n  Turn 3: Fetch PSP_FLD_L2_MAG_RTN_1MIN (ISSUE-08)")
    results.start_turn(3, "Fetch PSP_FLD_L2_MAG_RTN_1MIN for 2024-01-01 to 2024-01-07")
    r = send("Fetch PSP_FLD_L2_MAG_RTN_1MIN for 2024-01-01 to 2024-01-07")
    results.record_response(r)
    results.check(
        "no ask_clarification (ISSUE-08)",
        not has_tool(r, "ask_clarification"),
        f"tools: {tool_names(r)}",
    )
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 4: Memory check
    print("\n  Turn 4: What data do I have?")
    results.start_turn(4, "What data do I have?")
    r = send("What data do I have?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    text = resp_text(r)
    results.check("mentions ACE data", text_contains_any(text, ["ac_h2_mfi", "ace"]), f"text: {text[:120]}")
    results.check("mentions OMNI data", text_contains_any(text, ["omni_hro_1min", "omni"]), f"text: {text[:120]}")
    results.end_turn()

    results.end_scenario()


def scenario_8_progressive_refinement(results: TestResults):
    """Progressive Plot Refinement — context persistence, autoplot state."""
    results.start_scenario(
        "Scenario 8: Progressive Plot Refinement",
        "8-turn plot refinement chain testing context persistence",
        [],
    )
    print("\n" + "=" * 60)
    print("  Scenario 8: Progressive Plot Refinement")
    print("  Issues: context persistence, delegation chain depth")
    print("=" * 60)
    reset()

    # Turn 1: Initial plot
    print("\n  Turn 1: Show OMNI magnetic field magnitude")
    results.start_turn(1, "Show me OMNI solar wind magnetic field magnitude for January 2024")
    r = send("Show me OMNI solar wind magnetic field magnitude for January 2024")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 2: Log scale
    print("\n  Turn 2: Set y-axis to logarithmic scale")
    results.start_turn(2, "Set the y-axis to logarithmic scale")
    r = send("Set the y-axis to logarithmic scale")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Y-axis range
    print("\n  Turn 3: Set y-axis range")
    results.start_turn(3, "Set the y-axis range from 0.5 to 50")
    r = send("Set the y-axis range from 0.5 to 50")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 4: Y-axis label
    print("\n  Turn 4: Set y-axis label")
    results.start_turn(4, "Set the y-axis label to '|B| [nT]'")
    r = send("Set the y-axis label to '|B| [nT]'")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Render type
    print("\n  Turn 5: Change render type to scatter")
    results.start_turn(5, "Change the render type to scatter")
    r = send("Change the render type to scatter")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 6: Title
    print("\n  Turn 6: Set title")
    results.start_turn(6, "Set the title to 'OMNI IMF Magnitude'")
    r = send("Set the title to 'OMNI IMF Magnitude'")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 7: Canvas size
    print("\n  Turn 7: Set canvas size")
    results.start_turn(7, "Set the canvas size to 1920 by 1080")
    r = send("Set the canvas size to 1920 by 1080")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 8: Export
    print("\n  Turn 8: Export")
    results.start_turn(8, "Export the final result as tests/plots/omni_refined.png")
    r = send("Export the final result as tests/plots/omni_refined.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "omni_refined.png"
    results.check("PNG file exists", png_path.exists(), str(png_path))
    results.end_turn()

    results.end_scenario()


def scenario_9_error_recovery(results: TestResults):
    """Error Recovery Chain — handle failures, maintain conversation."""
    results.start_scenario(
        "Scenario 9: Error Recovery Chain",
        "Error handling and conversation continuity after failures",
        ["ISSUE-03", "ISSUE-07"],
    )
    print("\n" + "=" * 60)
    print("  Scenario 9: Error Recovery Chain")
    print("  Issues: ISSUE-03, -07")
    print("=" * 60)
    reset()

    # Turn 1: Bad dataset
    print("\n  Turn 1: Fetch nonexistent dataset")
    results.start_turn(1, "Fetch data from dataset FAKE_DATASET_999 for last week")
    r = send("Fetch data from dataset FAKE_DATASET_999 for last week")
    results.record_response(r)
    results.check(
        "error message in response",
        text_contains_any(resp_text(r), ["not found", "error", "no matching", "unknown", "don't", "does not", "couldn't", "not a recognized", "not recognized"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 2: Recovery — valid fetch
    print("\n  Turn 2: Recover with valid ACE fetch")
    results.start_turn(2, "OK, try fetching ACE magnetic field magnitude instead for last week")
    r = send("OK, try fetching ACE magnetic field magnitude instead for last week")
    results.record_response(r)
    results.check("has delegate_to_mission", has_tool(r, "delegate_to_mission"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 3: Rolling std (ISSUE-03)
    print("\n  Turn 3: Rolling std with 30-min window (ISSUE-03)")
    results.start_turn(3, "Compute the standard deviation with a 30-minute window")
    r = send("Compute the standard deviation with a 30-minute window")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 4: Overlay plot
    print("\n  Turn 4: Plot original + rolling std")
    results.start_turn(4, "Plot the original data and the rolling std together")
    r = send("Plot the original data and the rolling std together")
    results.record_response(r)
    results.check("has delegate_to_visualization", has_tool(r, "delegate_to_visualization"), f"tools: {tool_names(r)}")
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 5: Render type staircase (ISSUE-07)
    print("\n  Turn 5: Change render type to staircase (ISSUE-07: stairstep mapping)")
    results.start_turn(5, "Change render type to staircase")
    r = send("Change render type to staircase")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.end_turn()

    # Turn 6: Export
    print("\n  Turn 6: Export")
    results.start_turn(6, "Export to tests/plots/recovery_test.png")
    r = send("Export to tests/plots/recovery_test.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "recovery_test.png"
    results.check("PNG file exists", png_path.exists(), str(png_path))
    results.end_turn()

    results.end_scenario()


def scenario_10_complex_decomposition(results: TestResults):
    """Complex Single-Prompt Decomposition — multi-delegation in one turn."""
    results.start_scenario(
        "Scenario 10: Complex Single-Prompt Decomposition",
        "Orchestrator chains multiple delegations in a single turn",
        [],
    )
    print("\n" + "=" * 60)
    print("  Scenario 10: Complex Single-Prompt Decomposition")
    print("  Issues: multi-delegation chaining")
    print("=" * 60)
    reset()

    # Turn 1: Complex multi-step request
    print("\n  Turn 1: Complex request — fetch 2 datasets, compute difference, plot all")
    msg = (
        "Compare ACE and Wind magnetic field magnitude for January 15-20, 2024. "
        "Fetch both datasets, compute the difference between them, and plot the "
        "original ACE data, the original Wind data, and the difference all together."
    )
    results.start_turn(1, msg)
    r = send(msg)
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response mentions ACE/Wind/plot/difference",
        text_contains_any(resp_text(r), ["ace", "wind", "plot", "difference"]),
        f"text: {resp_text(r)[:150]}",
    )
    results.end_turn()

    # Turn 2: Title + export
    print("\n  Turn 2: Set title and export")
    results.start_turn(2, "Set the title to 'ACE vs Wind Magnetic Field' and export as tests/plots/ace_wind_comparison.png")
    r = send("Set the title to 'ACE vs Wind Magnetic Field' and export as tests/plots/ace_wind_comparison.png")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    time.sleep(2)
    png_path = PROJECT_ROOT / "tests" / "plots" / "ace_wind_comparison.png"
    results.check("PNG file exists", png_path.exists(), str(png_path))
    results.end_turn()

    # Turn 3: Describe computed data
    print("\n  Turn 3: Describe the difference data")
    results.start_turn(3, "Describe the difference data - what's the mean and standard deviation?")
    r = send("Describe the difference data - what's the mean and standard deviation?")
    results.record_response(r)
    results.check("no error", resp_error(r) is None, str(resp_error(r)))
    results.check(
        "response has statistics",
        text_contains_any(resp_text(r), ["mean", "std", "statistics", "average", "deviation"]),
        f"text: {resp_text(r)[:120]}",
    )
    results.end_turn()

    results.end_scenario()


# ============================================================================
# Scenario registry
# ============================================================================

ALL_SCENARIOS = {
    1:  ("Full ACE Analysis Pipeline",    scenario_1_ace_analysis),
    2:  ("CDAWeb Crash Guard + Recovery",  scenario_2_crash_guard),
    3:  ("MMS Data Access",                scenario_3_mms_access),
    4:  ("Cross-Mission Comparison",       scenario_4_cross_mission),
    5:  ("Vector Data Handling",           scenario_5_vector_data),
    6:  ("STEREO-A and Wind Discovery",    scenario_6_discovery),
    7:  ("Explicit Parameters",            scenario_7_explicit_params),
    8:  ("Progressive Plot Refinement",    scenario_8_progressive_refinement),
    9:  ("Error Recovery Chain",           scenario_9_error_recovery),
    10: ("Complex Decomposition",          scenario_10_complex_decomposition),
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Regression test suite for 2026-02-07 issue fixes"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Start server in verbose mode")
    parser.add_argument("--scenario", "-s", type=int,
                        help="Run only this scenario number (1-10)")
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

        # Select scenarios
        if args.scenario:
            if args.scenario not in ALL_SCENARIOS:
                print(f"Invalid scenario number: {args.scenario}. Choose 1-{len(ALL_SCENARIOS)}.")
                sys.exit(1)
            scenarios_to_run = {args.scenario: ALL_SCENARIOS[args.scenario]}
        else:
            scenarios_to_run = ALL_SCENARIOS

        # Run scenarios
        for num, (name, func) in sorted(scenarios_to_run.items()):
            try:
                func(results)
            except ConnectionRefusedError:
                print(f"\n  [!] Scenario {num} ({name}): SERVER CRASHED — ConnectionRefusedError")
                results.start_scenario(f"Scenario {num}: {name} (CRASHED)", "Server crash", [])
                results.start_turn(0, "N/A")
                results.record_response({"response": "", "error": "Server crashed", "tool_calls": []})
                results.check("Server survived scenario", False, "ConnectionRefusedError")
                results.end_turn()
                results.end_scenario()

                # Try to restart if we own the server
                if not args.no_server and proc:
                    print("  Attempting server restart...")
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    try:
                        proc = start_server(verbose=args.verbose)
                    except SystemExit:
                        print("  Server restart failed. Aborting remaining scenarios.")
                        break
                else:
                    print("  Cannot restart server (--no-server). Aborting remaining scenarios.")
                    break

            except Exception as e:
                print(f"\n  [!] Scenario {num} ({name}) error: {e}")
                results.start_scenario(f"Scenario {num}: {name} (ERROR)", str(e), [])
                results.start_turn(0, "N/A")
                results.record_response({"response": "", "error": str(e), "tool_calls": []})
                results.check("Scenario execution", False, str(e))
                results.end_turn()
                results.end_scenario()

        # Summary
        elapsed = round(time.time() - start_time, 1)
        report = results.summary()
        report += f"\n  Elapsed: {elapsed}s"
        print(report)

        # Save JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = PROJECT_ROOT / "tests" / f"regression_results_20260207_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results.to_json(), f, indent=2, default=str)
        print(f"\nResults saved: {json_path}")

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
