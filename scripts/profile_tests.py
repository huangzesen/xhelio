#!/usr/bin/env python3
"""scripts/profile_tests.py — Test suite profiler and bottleneck analyzer.

Runs the full test suite with profiling, identifies bottleneck tests,
detects and diagnoses failures, and generates a structured report.

Usage:
    python scripts/profile_tests.py                    # Profile all tests
    python scripts/profile_tests.py --slow             # Include @pytest.mark.slow tests
    python scripts/profile_tests.py --save DIR         # Save report to DIR/
    python scripts/profile_tests.py --top 20           # Show top N slowest tests
    python scripts/profile_tests.py --threshold 1.0    # Flag tests slower than N seconds
    python scripts/profile_tests.py --pattern test_mem # Only profile matching files
    python scripts/profile_tests.py --fix-report       # Include fix proposals for failures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """A single test's profiling data."""
    nodeid: str             # e.g. tests/test_foo.py::TestBar::test_baz
    outcome: str            # "passed", "failed", "skipped", "error"
    duration: float         # seconds
    file: str = ""
    classname: str = ""
    testname: str = ""
    failure_message: str = ""
    failure_short: str = ""

    def __post_init__(self):
        # Parse nodeid into components
        parts = self.nodeid.split("::")
        self.file = parts[0] if parts else ""
        if len(parts) == 3:
            self.classname = parts[1]
            self.testname = parts[2]
        elif len(parts) == 2:
            self.testname = parts[1]


@dataclass
class FileProfile:
    """Aggregated stats for a test file."""
    path: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration: float = 0.0
    slowest_test: Optional[TestResult] = None
    failures: list[TestResult] = field(default_factory=list)


@dataclass
class ProfileReport:
    """Complete profiling report."""
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_duration: float = 0.0
    tests: list[TestResult] = field(default_factory=list)
    files: dict[str, FileProfile] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------

def run_pytest(
    test_dir: str,
    *,
    slow: bool = False,
    pattern: str = "",
    extra_args: list[str] | None = None,
) -> tuple[str, str, int]:
    """Run pytest with JSON report and timing, return (stdout, stderr, returncode)."""

    repo = Path(__file__).resolve().parent.parent
    venv_python = repo / "venv" / "bin" / "python"
    if not venv_python.exists():
        # Windows fallback
        venv_python = repo / "venv" / "Scripts" / "python.exe"

    cmd = [
        str(venv_python), "-m", "pytest",
        test_dir,
        "--tb=short",
        "--durations=0",
        "-q",
        "--no-header",
        "--json-report",
        "--json-report-file=-",  # stdout
    ]

    if slow:
        cmd.append("--slow")

    if pattern:
        cmd.extend(["-k", pattern])

    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=600,
    )
    return result.stdout, result.stderr, result.returncode


def run_pytest_basic(
    test_dir: str,
    *,
    slow: bool = False,
    pattern: str = "",
) -> tuple[str, int]:
    """Fallback: run pytest without json-report plugin, parse --durations output."""

    repo = Path(__file__).resolve().parent.parent
    venv_python = repo / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo / "venv" / "Scripts" / "python.exe"

    cmd = [
        str(venv_python), "-m", "pytest",
        test_dir,
        "--tb=short",
        "--durations=0",
        "-v",
    ]

    if slow:
        cmd.append("--slow")

    if pattern:
        cmd.extend(["-k", pattern])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=600,
    )
    return result.stdout + "\n" + result.stderr, result.returncode


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_json_report(raw: str) -> Optional[dict]:
    """Extract JSON report from pytest-json-report output."""
    # The JSON report is printed to stdout — find it
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("{") and '"tests"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Try the whole output
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_verbose_output(output: str) -> ProfileReport:
    """Parse pytest -v --durations=0 output into a ProfileReport."""
    report = ProfileReport(timestamp=datetime.now().isoformat())
    tests = []

    # Parse test lines: tests/test_foo.py::TestBar::test_baz PASSED/FAILED
    test_line_re = re.compile(
        r"^(tests/\S+::\S+)\s+"
        r"(PASSED|FAILED|SKIPPED|ERROR)"
        r"\s*(?:\[\s*\d+%\])?\s*$"
    )

    # Parse duration lines: 1.23s call     tests/test_foo.py::TestBar::test_baz
    duration_re = re.compile(
        r"^\s*([\d.]+)s\s+(?:call|setup|teardown)\s+(tests/\S+::\S+)"
    )

    # Parse failure header: ____ TestFoo.test_bar ____  or FAILED tests/...
    failure_header_re = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}$")
    failure_summary_re = re.compile(r"^FAILED\s+(tests/\S+::\S+)")

    durations: dict[str, float] = {}
    outcomes: dict[str, str] = {}
    failure_messages: dict[str, str] = {}

    current_failure_id = ""
    in_failures = False
    in_summary = False
    failure_lines: list[str] = []

    lines_list = output.split("\n")
    for line in lines_list:
        # Test result line
        m = test_line_re.match(line)
        if m:
            nodeid, outcome = m.group(1), m.group(2).lower()
            outcomes[nodeid] = outcome
            continue

        # Duration line
        m = duration_re.match(line)
        if m:
            dur, nodeid = float(m.group(1)), m.group(2)
            durations[nodeid] = max(durations.get(nodeid, 0), dur)
            if nodeid not in outcomes:
                outcomes[nodeid] = "passed"  # default if not seen
            continue

        # Detect sections
        if "= FAILURES =" in line:
            in_failures = True
            in_summary = False
            continue
        if "= short test summary" in line:
            # Flush current failure
            if current_failure_id and failure_lines:
                failure_messages[current_failure_id] = "\n".join(failure_lines)
            in_failures = False
            in_summary = True
            current_failure_id = ""
            failure_lines = []
            continue
        if "= warnings summary" in line or line.startswith("===="):
            if current_failure_id and failure_lines:
                failure_messages[current_failure_id] = "\n".join(failure_lines)
            in_failures = False
            in_summary = False
            current_failure_id = ""
            failure_lines = []
            continue

        # Inside FAILURES section — parse individual failure blocks
        if in_failures:
            m = failure_header_re.match(line)
            if m:
                # Flush previous failure
                if current_failure_id and failure_lines:
                    failure_messages[current_failure_id] = "\n".join(failure_lines)
                # The header has the test name but not the full nodeid;
                # we'll match it to nodeids later
                current_failure_id = m.group(1).strip()
                failure_lines = []
            elif current_failure_id:
                failure_lines.append(line)

        # Inside short test summary — pick up FAILED lines
        if in_summary:
            m = failure_summary_re.match(line)
            if m:
                nodeid = m.group(1)
                outcomes[nodeid] = "failed"

    # Flush last failure
    if current_failure_id and failure_lines:
        failure_messages[current_failure_id] = "\n".join(failure_lines)

    # Map failure_messages from short names to full nodeids
    mapped_failures: dict[str, str] = {}
    all_nodeids = set(outcomes.keys()) | set(durations.keys())
    for short_name, msg in failure_messages.items():
        # Try exact match first
        for nodeid in all_nodeids:
            # Match by class::method or just the tail
            if nodeid.endswith(short_name) or short_name in nodeid:
                mapped_failures[nodeid] = msg
                break
        else:
            # Fuzzy: try matching individual parts
            parts = short_name.replace(".", "::").split("::")
            for nodeid in all_nodeids:
                if all(p in nodeid for p in parts):
                    mapped_failures[nodeid] = msg
                    break

    # Parse summary line: "14 failed, 1939 passed, 18 skipped"
    summary_re = re.compile(r"(\d+)\s+(failed|passed|skipped|error|warning)")
    for m in summary_re.finditer(output):
        count, kind = int(m.group(1)), m.group(2)
        if kind == "passed":
            report.passed = count
        elif kind == "failed":
            report.failed = count
        elif kind == "skipped":
            report.skipped = count
        elif kind == "error":
            report.errors = count

    # Parse total time: "in 63.99s"
    time_re = re.compile(r"in\s+([\d.]+)s")
    m = time_re.search(output)
    if m:
        report.total_duration = float(m.group(1))

    # Build TestResult objects
    for nodeid in sorted(all_nodeids):
        outcome = outcomes.get(nodeid, "passed")
        duration = durations.get(nodeid, 0.0)
        failure_msg = mapped_failures.get(nodeid, "")

        tr = TestResult(
            nodeid=nodeid,
            outcome=outcome,
            duration=duration,
            failure_message=failure_msg,
        )
        tests.append(tr)

    report.tests = tests
    report.total_tests = len(tests)

    # Aggregate by file
    for tr in tests:
        if tr.file not in report.files:
            report.files[tr.file] = FileProfile(path=tr.file)
        fp = report.files[tr.file]
        fp.total_tests += 1
        fp.total_duration += tr.duration
        if tr.outcome == "passed":
            fp.passed += 1
        elif tr.outcome == "failed":
            fp.failed += 1
            fp.failures.append(tr)
        elif tr.outcome == "skipped":
            fp.skipped += 1
        else:
            fp.errors += 1
        if fp.slowest_test is None or tr.duration > fp.slowest_test.duration:
            fp.slowest_test = tr

    return report


def parse_json_to_report(data: dict) -> ProfileReport:
    """Convert pytest-json-report JSON to ProfileReport."""
    report = ProfileReport(
        timestamp=datetime.now().isoformat(),
        total_duration=data.get("duration", 0.0),
    )

    summary = data.get("summary", {})
    report.passed = summary.get("passed", 0)
    report.failed = summary.get("failed", 0)
    report.skipped = summary.get("skipped", 0)
    report.errors = summary.get("error", 0)
    report.total_tests = summary.get("total", 0)

    for t in data.get("tests", []):
        nodeid = t.get("nodeid", "")
        outcome = t.get("outcome", "")
        duration = t.get("call", {}).get("duration", 0.0) if t.get("call") else 0.0

        failure_msg = ""
        if outcome == "failed" and t.get("call", {}).get("longrepr"):
            failure_msg = t["call"]["longrepr"]

        tr = TestResult(
            nodeid=nodeid,
            outcome=outcome,
            duration=duration,
            failure_message=failure_msg,
        )
        report.tests.append(tr)

        # Aggregate by file
        if tr.file not in report.files:
            report.files[tr.file] = FileProfile(path=tr.file)
        fp = report.files[tr.file]
        fp.total_tests += 1
        fp.total_duration += tr.duration
        if outcome == "passed":
            fp.passed += 1
        elif outcome == "failed":
            fp.failed += 1
            fp.failures.append(tr)
        elif outcome == "skipped":
            fp.skipped += 1
        else:
            fp.errors += 1
        if fp.slowest_test is None or tr.duration > fp.slowest_test.duration:
            fp.slowest_test = tr

    return report


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def classify_failure(tr: TestResult) -> dict:
    """Classify a test failure and propose a fix direction."""
    msg = tr.failure_message.lower()
    nodeid = tr.nodeid

    info = {
        "nodeid": nodeid,
        "category": "unknown",
        "summary": "",
        "fix_direction": "",
    }

    if "assertionerror" in msg or "assert " in msg:
        info["category"] = "assertion"
        # Check common patterns — more specific first
        if "called" in msg and "times" in msg:
            info["summary"] = "Mock called wrong number of times"
            info["fix_direction"] = (
                "The code path triggers the callback multiple times. "
                "Either fix the code to avoid duplicate calls, "
                "or update the test to expect multiple calls."
            )
        elif ("get_limit" in msg or "get_item_limit" in msg
              or ("assert" in msg and "==" in msg and re.search(r"\b500\b.*\b\d{2,3}\b", msg))):
            info["summary"] = "Default value mismatch — code DEFAULTS changed but test not updated"
            info["fix_direction"] = (
                "The DEFAULTS dict in agent/truncation.py has values that don't match test expectations. "
                "Update the DEFAULTS dict to match the test-specified values (the tests define the contract)."
            )
        elif "endswith" in msg and "..." in msg:
            info["summary"] = "Truncation not triggering — limit too high for test input"
            info["fix_direction"] = (
                "Lower the default limit so truncation triggers on the test input, "
                "or increase the test input length to exceed the current limit."
            )
        elif "'...' in" in msg or '"..." in' in msg or "assert '...' in" in msg:
            info["summary"] = "Expected truncation marker missing — text not being truncated"
            info["fix_direction"] = (
                "The truncation limit is higher than the test input. "
                "Fix the limit or adjust the test input length."
            )
        elif "len(" in msg and "==" in msg:
            info["summary"] = "Length mismatch — truncation limit changed"
            info["fix_direction"] = (
                "The truncation limit doesn't match the expected output length. "
                "Update the default limit or the test expectation."
            )
        else:
            info["summary"] = "Assertion mismatch"
            info["fix_direction"] = "Compare expected vs actual values and update accordingly."
    elif "keyerror" in msg:
        info["category"] = "missing_key"
        info["summary"] = "Missing dict key or registry entry"
        info["fix_direction"] = "Add the missing key to the relevant dictionary or registry."
    elif "nameerror" in msg:
        info["category"] = "name_error"
        info["summary"] = "Undefined variable or function"
        info["fix_direction"] = "Import or define the missing name."
    elif "importerror" in msg or "modulenotfounderror" in msg:
        info["category"] = "import_error"
        info["summary"] = "Missing module or import"
        info["fix_direction"] = "Install the missing package or fix the import path."
    elif "typeerror" in msg:
        info["category"] = "type_error"
        info["summary"] = "Wrong argument types or count"
        info["fix_direction"] = "Check function signatures and call sites."
    elif "timeout" in msg or "timed out" in msg:
        info["category"] = "timeout"
        info["summary"] = "Test timed out"
        info["fix_direction"] = "Investigate blocking I/O or infinite loops."
    elif "attributeerror" in msg:
        info["category"] = "attribute_error"
        info["summary"] = "Missing attribute on object"
        info["fix_direction"] = "Check the class/object interface — attribute may have been renamed or removed."
    else:
        info["summary"] = "Unclassified failure"
        info["fix_direction"] = "Review the failure message manually."

    return info


def group_failures(failures: list[TestResult]) -> dict[str, list[dict]]:
    """Group failures by root cause pattern."""
    groups: dict[str, list[dict]] = {}
    for tr in failures:
        info = classify_failure(tr)
        key = info["summary"]
        if key not in groups:
            groups[key] = []
        groups[key].append(info)
    return groups


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown(
    report: ProfileReport,
    *,
    top_n: int = 20,
    threshold: float = 1.0,
    include_fixes: bool = True,
) -> str:
    """Generate a Markdown profiling report."""
    lines = []
    lines.append(f"# Test Profile Report")
    lines.append(f"")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Total duration:** {report.total_duration:.2f}s")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tests | {report.total_tests} |")
    lines.append(f"| Passed | {report.passed} |")
    lines.append(f"| Failed | {report.failed} |")
    lines.append(f"| Skipped | {report.skipped} |")
    lines.append(f"| Errors | {report.errors} |")

    if report.total_tests > 0:
        pass_rate = (report.passed / report.total_tests) * 100
        lines.append(f"| Pass rate | {pass_rate:.1f}% |")

    # --- Bottleneck tests ---
    sorted_tests = sorted(report.tests, key=lambda t: t.duration, reverse=True)
    slow_tests = [t for t in sorted_tests if t.duration >= threshold]
    top_tests = sorted_tests[:top_n]

    lines.append(f"")
    lines.append(f"## Top {top_n} Slowest Tests")
    lines.append(f"")
    lines.append(f"| # | Duration | Test |")
    lines.append(f"|---|----------|------|")
    for i, t in enumerate(top_tests, 1):
        flag = " **SLOW**" if t.duration >= threshold else ""
        lines.append(f"| {i} | {t.duration:.3f}s{flag} | `{t.nodeid}` |")

    # --- Slow tests above threshold ---
    if slow_tests:
        lines.append(f"")
        lines.append(f"## Tests Above {threshold}s Threshold ({len(slow_tests)} found)")
        lines.append(f"")
        lines.append(f"These tests are bottlenecks and may benefit from optimization:")
        lines.append(f"")
        for t in slow_tests:
            lines.append(f"- **{t.duration:.3f}s** `{t.nodeid}`")

    # --- File-level analysis ---
    sorted_files = sorted(report.files.values(), key=lambda f: f.total_duration, reverse=True)
    lines.append(f"")
    lines.append(f"## Slowest Test Files")
    lines.append(f"")
    lines.append(f"| # | Duration | Tests | Failed | File |")
    lines.append(f"|---|----------|-------|--------|------|")
    for i, fp in enumerate(sorted_files[:15], 1):
        fail_str = str(fp.failed) if fp.failed else "-"
        lines.append(
            f"| {i} | {fp.total_duration:.2f}s | {fp.total_tests} | {fail_str} | `{fp.path}` |"
        )

    # --- Failures ---
    failed_tests = [t for t in report.tests if t.outcome == "failed"]
    if failed_tests:
        lines.append(f"")
        lines.append(f"## Failed Tests ({len(failed_tests)})")
        lines.append(f"")

        # Group by root cause
        if include_fixes:
            groups = group_failures(failed_tests)
            for cause, items in groups.items():
                lines.append(f"### {cause} ({len(items)} tests)")
                lines.append(f"")
                lines.append(f"**Fix direction:** {items[0]['fix_direction']}")
                lines.append(f"")
                for item in items:
                    lines.append(f"- `{item['nodeid']}`")
                lines.append(f"")

        # Full failure details
        lines.append(f"### Failure Details")
        lines.append(f"")
        for t in failed_tests:
            lines.append(f"#### `{t.nodeid}`")
            lines.append(f"")
            if t.failure_message:
                lines.append(f"```")
                # Limit failure message to reasonable length for report
                msg_lines = t.failure_message.strip().split("\n")
                for msg_line in msg_lines[:30]:
                    lines.append(msg_line)
                if len(msg_lines) > 30:
                    lines.append(f"... ({len(msg_lines) - 30} more lines)")
                lines.append(f"```")
            lines.append(f"")

    # --- Optimization proposals ---
    if slow_tests:
        lines.append(f"## Optimization Proposals")
        lines.append(f"")

        # Group slow tests by file
        slow_by_file: dict[str, list[TestResult]] = {}
        for t in slow_tests:
            slow_by_file.setdefault(t.file, []).append(t)

        for filepath, tests in sorted(slow_by_file.items(), key=lambda x: -sum(t.duration for t in x[1])):
            total = sum(t.duration for t in tests)
            lines.append(f"### `{filepath}` — {total:.2f}s in {len(tests)} slow tests")
            lines.append(f"")
            lines.append(f"**Proposals:**")
            lines.append(f"")

            # Common optimization suggestions based on test characteristics
            if any("setup" in t.testname.lower() or "init" in t.testname.lower() for t in tests):
                lines.append(f"- Consider `@pytest.fixture(scope='class')` or `scope='module'` for expensive setup")
            if len(tests) >= 5:
                lines.append(f"- Many slow tests in this file — consider `pytest-xdist` for parallel execution")
            if any(t.duration > 5.0 for t in tests):
                lines.append(f"- Tests over 5s may indicate real I/O or network calls — consider mocking")
            if any("sleep" in t.testname.lower() for t in tests):
                lines.append(f"- Tests with 'sleep' in name may be waiting unnecessarily")
            lines.append(f"- Review for redundant setup/teardown between tests")
            lines.append(f"- Profile individual tests with `pytest --profile` or cProfile")
            lines.append(f"")

    # --- Distribution ---
    if report.tests:
        lines.append(f"## Duration Distribution")
        lines.append(f"")
        buckets = [
            (0.0, 0.01, "<10ms"),
            (0.01, 0.1, "10-100ms"),
            (0.1, 0.5, "100-500ms"),
            (0.5, 1.0, "500ms-1s"),
            (1.0, 5.0, "1-5s"),
            (5.0, float("inf"), ">5s"),
        ]
        lines.append(f"| Range | Count | % |")
        lines.append(f"|-------|-------|---|")
        for lo, hi, label in buckets:
            count = sum(1 for t in report.tests if lo <= t.duration < hi)
            pct = (count / len(report.tests) * 100) if report.tests else 0
            bar = "#" * int(pct / 2)
            lines.append(f"| {label} | {count} | {pct:.1f}% {bar} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile test suite and identify bottlenecks")
    parser.add_argument("test_dir", nargs="?", default="tests/",
                        help="Test directory to profile (default: tests/)")
    parser.add_argument("--slow", action="store_true",
                        help="Include @pytest.mark.slow tests")
    parser.add_argument("--save", metavar="DIR",
                        help="Save report to DIR/test-profile-TIMESTAMP.md")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N slowest tests (default: 20)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Flag tests slower than N seconds (default: 1.0)")
    parser.add_argument("--pattern", "-k", default="",
                        help="Only profile tests matching this pattern")
    parser.add_argument("--fix-report", action="store_true", default=True,
                        help="Include fix proposals for failures (default: on)")
    parser.add_argument("--no-fix-report", action="store_false", dest="fix_report",
                        help="Exclude fix proposals")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON data instead of markdown")
    args = parser.parse_args()

    print(f"Profiling tests in {args.test_dir}...")
    print(f"  slow tests: {'included' if args.slow else 'excluded'}")
    print(f"  threshold: {args.threshold}s")
    if args.pattern:
        print(f"  pattern: {args.pattern}")
    print()

    start = time.time()

    # Check if pytest-json-report is available before trying it
    report = None
    has_json_report = False
    try:
        repo = Path(__file__).resolve().parent.parent
        check = subprocess.run(
            [str(repo / "venv" / "bin" / "python"), "-c",
             "import pytest_jsonreport; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        has_json_report = check.returncode == 0
    except Exception:
        pass

    if has_json_report:
        try:
            stdout, stderr, returncode = run_pytest(
                args.test_dir,
                slow=args.slow,
                pattern=args.pattern,
            )
            data = parse_json_report(stdout)
            if data:
                report = parse_json_to_report(data)
        except Exception as e:
            print(f"JSON report failed ({e}), falling back to verbose parser...")

    if report is None:
        output, returncode = run_pytest_basic(
            args.test_dir,
            slow=args.slow,
            pattern=args.pattern,
        )
        report = parse_verbose_output(output)

    elapsed = time.time() - start
    print(f"Test run completed in {elapsed:.1f}s")
    print(f"  {report.passed} passed, {report.failed} failed, "
          f"{report.skipped} skipped, {report.errors} errors")
    print()

    if args.json:
        # JSON output mode
        json_data = {
            "timestamp": report.timestamp,
            "summary": {
                "total": report.total_tests,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "errors": report.errors,
                "duration": report.total_duration,
            },
            "tests": [
                {
                    "nodeid": t.nodeid,
                    "outcome": t.outcome,
                    "duration": t.duration,
                    "file": t.file,
                    "failure_message": t.failure_message,
                }
                for t in report.tests
            ],
        }
        print(json.dumps(json_data, indent=2))
        return returncode

    # Generate markdown report
    md = generate_markdown(
        report,
        top_n=args.top,
        threshold=args.threshold,
        include_fixes=args.fix_report,
    )

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filepath = save_dir / f"test-profile-{ts}.md"
        filepath.write_text(md)
        print(f"Report saved to {filepath}")

        # Also write a "latest" symlink/copy
        latest = save_dir / "test-profile-latest.md"
        latest.write_text(md)
        print(f"Latest report: {latest}")
    else:
        print(md)

    return returncode


if __name__ == "__main__":
    sys.exit(main() or 0)
