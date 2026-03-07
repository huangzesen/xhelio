# Regression Test Guide — 2026-02-07 Fixes

This document describes the multi-turn regression test suite that validates the 15 issue fixes from commit `87114d3`.

## Overview

The suite in `scripts/regression_test_20260207.py` runs **10 multi-turn scenarios** (~58 commands total) against a live agent server. Each scenario is a coherent conversation chain that stresses delegation chains, label propagation, compute pipelines, and Autoplot state management.

Unlike the single-shot tests in `scripts/run_agent_tests.py`, these scenarios use 3-10 turn conversations that build on prior context — testing how fixes hold up under realistic interactive use.

## Quick Start

```bash
# Full run (starts and stops server automatically)
python scripts/regression_test_20260207.py

# Use an already-running server
python scripts/agent_server.py serve --verbose   # terminal 1
python scripts/regression_test_20260207.py --no-server   # terminal 2

# Single scenario
python scripts/regression_test_20260207.py --scenario 2
```

## Scenario Reference

| # | Name | Turns | Issues Tested | What It Validates |
|---|------|-------|---------------|-------------------|
| 1 | Full ACE Analysis Pipeline | 8 | 03, 05, 07, 09, 10, 11, 12 | Fetch vector data, rolling average, magnitude, plot, title, render type, color table error, export PNG+VAP |
| 2 | CDAWeb Crash Guard + Recovery | 6 | 01, 02, 15 | Invalid parameter handling, 4-panel guard, server stability |
| 3 | MMS Data Access | 5 | 06, 08 | MMS FGM @0 suffix, describe, plot, relative-path export |
| 4 | Cross-Mission Comparison | 7 | — | 3-mission fetch (ACE/Wind/DSCOVR), memory listing, overlay, axis customization, export |
| 5 | Vector Data Handling | 6 | 04, 09 | Vector fetch, single component, 3-panel decomposition, per-panel titles, magnitude |
| 6 | STEREO-A and Wind Discovery | 5 | 13, 14 | Dataset browsing, cadence awareness, resolution selection |
| 7 | Explicit Parameters | 4 | 08 | No unnecessary clarification when dataset+parameter given |
| 8 | Progressive Plot Refinement | 8 | — | 8-step refinement: log scale, range, label, scatter, title, canvas size, export |
| 9 | Error Recovery Chain | 6 | 03, 07 | Bad dataset recovery, rolling std, staircase render type, export |
| 10 | Complex Decomposition | 3 | — | Multi-delegation in single turn: fetch 2 datasets, compute difference, plot, export |

## Per-Issue Fix Reference

### ISSUE-01: 4-panel plot crashes JVM
- **Trigger**: `sc.plot(0, ...); sc.plot(1, ...); sc.plot(2, ...); sc.plot(3, ...)`
- **Old behavior**: JVM crashes with `DasCanvas.waitUntilIdle` thread synchronization error
- **New behavior**: Panel count guard limits to 3 panels max; agent explains limitation
- **Tested in**: Scenario 2 (Turn 5)

### ISSUE-02: Invalid CDAWeb parameter crashes JVM
- **Trigger**: Plotting with nonexistent parameter name (e.g., `NONEXISTENT_PARAM`)
- **Old behavior**: `java.lang.IllegalArgumentException` propagates to `ExitExceptionHandler`, kills JVM
- **New behavior**: Error caught, returns informative message, server stays alive
- **Tested in**: Scenario 2 (Turn 2)

### ISSUE-03: Time-based rolling window fails
- **Trigger**: `df.rolling('2H', center=True).mean()`
- **Old behavior**: `ValueError: passed window 2H is not compatible with a datetimelike index`
- **New behavior**: DataFrame index is DatetimeIndex; time-based windows work
- **Tested in**: Scenario 1 (Turn 2), Scenario 9 (Turn 3)

### ISSUE-04: `to_qdataset()` fails on vector data
- **Trigger**: `to_qdataset('AC_H2_MFI.BGSEc')` on 3-column dataset
- **Old behavior**: Error requiring component= argument
- **New behavior**: Component access supported; agent decomposes vectors
- **Tested in**: Scenario 5 (Turns 2-3)

### ISSUE-05: Relative file paths rejected by Autoplot export
- **Trigger**: `export_png("tests/plots/file.png")`
- **Old behavior**: `IllegalArgumentException: something is wrong with the specified filename`
- **New behavior**: Relative paths resolved to absolute before passing to Autoplot
- **Tested in**: Scenario 1 (Turn 8), Scenario 3 (Turn 5), Scenario 4 (Turn 7), Scenario 8 (Turn 8), Scenario 9 (Turn 6), Scenario 10 (Turn 2)

### ISSUE-06: MMS FGM dataset not found
- **Trigger**: Fetching `MMS1_FGM_SRVY_L2` (HAPI uses `@0`/`@1` suffixes)
- **Old behavior**: Dataset not found, agent says "I do not see any datasets containing FGM"
- **New behavior**: Agent understands `@0`/`@1` sub-datasets, fetches correctly
- **Tested in**: Scenario 3 (Turn 1)

### ISSUE-07: `fill_to_zero` render type rejected
- **Trigger**: "Change render type to fill_to_zero"
- **Old behavior**: Agent says render type is unsupported
- **New behavior**: Snake-case render type names mapped to Java enum values
- **Tested in**: Scenario 1 (Turn 6), Scenario 9 (Turn 5 — staircase)

### ISSUE-08: Unnecessary clarification with explicit parameters
- **Trigger**: "Fetch PSP_FLD_L2_MAG_RTN_1MIN" — exact dataset ID given
- **Old behavior**: Agent asks "Which parameter would you like?"
- **New behavior**: Agent uses provided IDs directly without clarification
- **Tested in**: Scenario 7 (Turns 1-3)

### ISSUE-09: Fetched vector data stored under unexpected label
- **Trigger**: Delegate to mission agent to fetch `AC_H2_MFI.BGSEc`
- **Old behavior**: Follow-up operations fail with "Label not found"
- **New behavior**: Mission agent returns exact stored label
- **Tested in**: Scenario 1 (Turn 1), Scenario 5 (Turn 1)

### ISSUE-10: `dom.setTitle()` doesn't exist
- **Trigger**: Agent generates `dom.setTitle('...')`
- **Old behavior**: `'Application' object has no attribute 'setTitle'`
- **New behavior**: Correct API used: `dom.getPlots(0).setTitle(title)`
- **Tested in**: Scenario 1 (Turn 5), Scenario 5 (Turn 4)

### ISSUE-11: Session save uses relative path
- **Trigger**: "Save session to tests/plots/test_session.vap"
- **Old behavior**: File saved to unknown location
- **New behavior**: Path resolved to absolute (same fix as ISSUE-05)
- **Tested in**: Scenario 1 (Turn 8)

### ISSUE-12: Color table fails on line plots
- **Trigger**: Set color table on a series/line plot
- **Old behavior**: Retries multiple times then reports failure
- **New behavior**: Explains that color tables only apply to spectrograms
- **Tested in**: Scenario 1 (Turn 7)

### ISSUE-13: STEREO-A dataset not found directly
- **Trigger**: "Fetch STEREO-A magnetic field (STA_L2_MAG_RTN)"
- **Old behavior**: Dataset not in recommended list, extra round-trip needed
- **New behavior**: Added to mission JSON recommended datasets
- **Tested in**: Scenario 6 (Turn 1)

### ISSUE-14: Wind 5-day high-cadence fetch very slow
- **Trigger**: Fetching WI_H2_MFI (3-second cadence) for 5 days
- **Old behavior**: 67-76 seconds, no warning
- **New behavior**: Agent warns about cadence, suggests lower-resolution dataset
- **Tested in**: Scenario 6 (Turn 3)

### ISSUE-15: Autoplot "strange bug" in waitUntilIdle
- **Trigger**: Multi-panel or overlay plots
- **Old behavior**: Intermittent race condition warning, sometimes leads to crash
- **New behavior**: Monitored; panel limit from ISSUE-01 fix reduces exposure
- **Tested in**: Scenario 2 (Turn 6)

## Check Types

Each turn has one or more checks that produce PASS/FAIL:

| Check | What It Validates |
|-------|-------------------|
| `no error` | `response["error"]` is None |
| `has <tool>` | Tool was called (e.g., `delegate_to_envoy`) |
| `no ask_clarification` | Agent did NOT ask for unnecessary clarification |
| `text contains` | Response text includes expected substring (case-insensitive) |
| `text not contains` | Response text does NOT include error string |
| `server alive` | Follow-up ping succeeds (critical after crash-prone operations) |
| `file exists` | Exported file was created on disk |

## Result Interpretation

### Console Output

```
============================================================
  Scenario 1: Full ACE Analysis Pipeline
  Issues: ISSUE-03, -05, -07, -09, -10, -11, -12
============================================================

  Turn 1: Fetch ACE magnetic field GSE components
    [PASS] has delegate_to_envoy
    [PASS] no error
    [PASS] response confirms data stored (ISSUE-09: labels)
  ...
```

### JSON Output

Results are saved to `tests/regression_results_20260207_{timestamp}.json`:

```json
{
  "suite": "regression_test_20260207",
  "timestamp": "2026-02-07T...",
  "total_checks": 85,
  "total_passed": 82,
  "all_passed": false,
  "scenarios": [
    {
      "name": "Scenario 1: Full ACE Analysis Pipeline",
      "issues_tested": ["ISSUE-03", "ISSUE-05", ...],
      "turns": [
        {
          "turn": 1,
          "message": "Fetch ACE magnetic field...",
          "response": {
            "text": "I've fetched...",
            "error": null,
            "tool_calls": ["delegate_to_envoy"],
            "elapsed": 12.3
          },
          "checks": [
            {"label": "has delegate_to_envoy", "passed": true, "detail": ""},
            ...
          ]
        }
      ]
    }
  ]
}
```

### Debugging Failures

1. **`[FAIL] server alive`** — The JVM crashed. Check the server terminal for Java stack traces. This indicates a regression in ISSUE-01 or ISSUE-02 fixes.

2. **`[FAIL] has <tool>`** — The LLM chose a different tool path. This may be non-deterministic; re-run to confirm. If consistent, check system prompts for missing routing instructions.

3. **`[FAIL] text contains`** — The LLM's response wording changed. Check if the semantic meaning is correct even if the exact string doesn't match. If so, widen the check's keyword list.

4. **`[FAIL] file exists`** — Export path resolution may have regressed. Check ISSUE-05 fix in `autoplot_bridge/commands.py`.

5. **`ConnectionRefusedError` during scenario** — Server crashed mid-scenario. The script attempts to restart and continue with remaining scenarios.

## Adding New Regression Tests

### Add a new scenario

1. Define a function following the pattern:

```python
def scenario_N_name(results: TestResults):
    results.start_scenario("Scenario N: Name", "Description", ["ISSUE-XX"])
    reset()

    # Turn 1
    results.start_turn(1, "User message")
    r = send("User message")
    results.record_response(r)
    results.check("label", condition, "detail")
    results.end_turn()

    results.end_scenario()
```

2. Register in `ALL_SCENARIOS`:

```python
ALL_SCENARIOS = {
    ...
    11: ("New Scenario", scenario_N_name),
}
```

### Add a turn to an existing scenario

Insert a new turn block between existing turns, adjusting turn numbers.

### Add a check to an existing turn

Add `results.check(...)` calls between `results.record_response(r)` and `results.end_turn()`.

## Known Limitations

- **HAPI server availability**: Tests depend on CDAWeb HAPI being reachable. Network outages cause data fetch failures that aren't code bugs.
- **LLM non-determinism**: Gemini may choose different tool paths or wordings across runs. Checks use flexible matching (`text_contains_any` with multiple keywords) to accommodate this.
- **Timing**: Slow network or server startup can cause timeouts. The server startup timeout is 90 seconds; individual requests timeout at 5 minutes.
- **State leakage**: Scenarios that don't call `reset()` at the start carry over state from the previous scenario. All current scenarios reset at the beginning.
- **Export checks**: File existence checks wait 2 seconds after the export command. On very slow systems, this may not be enough.
