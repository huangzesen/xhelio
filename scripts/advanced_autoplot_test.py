#!/usr/bin/env python3
"""
Advanced Autoplot Capability Test Suite

Exercises multi-panel plotting, diverse datasets from all 8 missions,
render types, axis manipulation, canvas sizing, exports, and complex workflows.
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

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = PROJECT_ROOT / "tests" / f"advanced_autoplot_results_{timestamp_str}.json"
EXPORT_DIR = PROJECT_ROOT / "tests" / "plots"

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


TEST_GROUPS = [
    # ---- GROUP 1: Multi-panel plotting ----
    {
        "group": "Multi-Panel Plotting",
        "description": "Test multi-panel layouts with different datasets on separate panels",
        "commands": [
            # Fetch two datasets, then request them on separate panels
            "Fetch ACE magnetic field magnitude (AC_H2_MFI, Magnitude) for 2024-01-15 to 2024-01-20",
            "Also fetch ACE solar wind speed (AC_H0_SWE, Vp) for the same period",
            "Plot the magnetic field magnitude on the top panel and the solar wind speed on a separate bottom panel",
        ],
        "reset_before": True,
    },

    # ---- GROUP 2: Cross-mission L1 magnetic field comparison ----
    {
        "group": "L1 Mag Field Comparison (ACE vs Wind vs DSCOVR)",
        "description": "Overlay magnetic field from 3 L1 monitors on same plot",
        "commands": [
            "Fetch magnetic field magnitude from ACE (AC_H2_MFI, Magnitude), Wind (WI_H2_MFI, BF1), and DSCOVR (DSCOVR_H0_MAG, B1F1) all for January 15-20, 2024",
            "Overlay all three magnetic field magnitudes on a single plot with title 'L1 Magnetic Field Comparison'",
        ],
        "reset_before": True,
    },

    # ---- GROUP 3: Inner heliosphere comparison (PSP vs SolO vs STEREO-A) ----
    {
        "group": "Inner Heliosphere Mag (PSP vs SolO vs STEREO-A)",
        "description": "Compare magnetic field from inner heliosphere missions in RTN coordinates",
        "commands": [
            "Fetch PSP magnetic field (PSP_FLD_L2_MAG_RTN_1MIN) for 2024-01-01 to 2024-01-07",
            "Also fetch Solar Orbiter magnetic field (SOLO_L2_MAG-RTN-NORMAL-1-MINUTE) for the same period",
            "And fetch STEREO-A magnetic field (STA_L2_MAG_RTN) for the same period too",
            "Plot all three mission magnetic fields together and export to tests/plots/inner_helio_mag.png",
        ],
        "reset_before": True,
    },

    # ---- GROUP 4: OMNI multi-parameter (mag + plasma on separate panels) ----
    {
        "group": "OMNI Multi-Parameter Dashboard",
        "description": "Create a dashboard with OMNI B-field, speed, density, and temperature",
        "commands": [
            "Fetch OMNI 1-minute data: magnetic field magnitude (F), solar wind speed (flow_speed), proton density (proton_density), and temperature (T) from OMNI_HRO_1MIN for January 2024",
            "Create a 4-panel plot: magnetic field on top, then speed, then density, then temperature at the bottom. Export to tests/plots/omni_dashboard.png",
        ],
        "reset_before": True,
    },

    # ---- GROUP 5: Render type changes ----
    {
        "group": "Render Type Variations",
        "description": "Test scatter, fill_to_zero, and staircase render types",
        "commands": [
            "Fetch Wind magnetic field magnitude (WI_H2_MFI, BF1) for 2024-01-15 to 2024-01-16",
            "Plot it and then change the render type to scatter",
            "Now change it to fill_to_zero",
            "Change it back to series (normal line plot)",
        ],
        "reset_before": True,
    },

    # ---- GROUP 6: Axis manipulation ----
    {
        "group": "Axis Manipulation",
        "description": "Test log scale, axis range, axis labels, and title setting",
        "commands": [
            "Plot ACE magnetic field magnitude (AC_H2_MFI, Magnitude) for January 2024",
            "Set the y-axis label to 'B magnitude [nT]'",
            "Set the y-axis range from 0 to 30",
            "Enable logarithmic scale on the y-axis",
            "Set the plot title to 'ACE MFI - January 2024'",
        ],
        "reset_before": True,
    },

    # ---- GROUP 7: Canvas sizing + high-res export ----
    {
        "group": "Canvas Sizing & Export",
        "description": "Resize canvas and export PNG + PDF",
        "commands": [
            "Plot OMNI magnetic field magnitude for January 2024",
            "Set the canvas size to 1920x1080 pixels for a high-resolution export",
            "Export the plot as a PNG to tests/plots/omni_hires.png",
            "Also export as PDF to tests/plots/omni_hires.pdf",
        ],
        "reset_before": True,
    },

    # ---- GROUP 8: MMS magnetospheric data ----
    {
        "group": "MMS Magnetospheric Data",
        "description": "Load MMS magnetic field and ion data",
        "commands": [
            "Fetch MMS1 magnetic field (MMS1_FGM_SRVY_L2) for 2024-01-10 to 2024-01-11",
            "Also fetch MMS1 ion density from FPI (MMS1_FPI_FAST_L2_DIS-MOMS) for the same period",
            "Plot the magnetic field on the top panel and ion density on a bottom panel",
        ],
        "reset_before": True,
    },

    # ---- GROUP 9: PSP perihelion encounter data ----
    {
        "group": "PSP Perihelion Encounter",
        "description": "Load PSP data during a close solar encounter with multiple instruments",
        "commands": [
            "Fetch PSP magnetic field magnitude (PSP_FLD_L2_MAG_RTN_1MIN) for 2024-09-25 to 2024-10-05",
            "Also fetch PSP solar wind proton speed from SPC (PSP_SWP_SPC_L3I, vp_moment_RTN) for the same period",
            "Plot the magnetic field on top and proton speed below. Title: 'PSP Encounter 21'",
        ],
        "reset_before": True,
    },

    # ---- GROUP 10: Compute + plot pipeline ----
    {
        "group": "Compute Pipeline + Multi-Panel",
        "description": "Fetch vector data, compute magnitude, smooth it, then multi-panel plot",
        "commands": [
            "Fetch ACE magnetic field GSE components (AC_H2_MFI, BGSEc) for January 15-20, 2024",
            "Compute the magnitude of the magnetic field vector",
            "Compute a running average of the magnitude with a window of 100 points",
            "Plot the original magnitude and smoothed magnitude overlaid on the top panel, and the raw components on a bottom panel",
        ],
        "reset_before": True,
    },

    # ---- GROUP 11: Direct CDAWeb plot (no fetch) ----
    {
        "group": "Direct CDAWeb Plot (No Pre-Fetch)",
        "description": "Use plot_cdaweb to plot directly without fetching first",
        "commands": [
            "Plot the dataset WI_H0_MFI parameter BZ3GSE directly from CDAWeb for January 2024 without fetching first",
            "Now zoom in to January 10-15, 2024",
            "What's currently plotted?",
        ],
        "reset_before": True,
    },

    # ---- GROUP 12: Solar Orbiter diverse instruments ----
    {
        "group": "Solar Orbiter Multi-Instrument",
        "description": "Load Solar Orbiter MAG + plasma data",
        "commands": [
            "Fetch Solar Orbiter magnetic field (SOLO_L2_MAG-RTN-NORMAL-1-MINUTE) for 2024-03-01 to 2024-03-07",
            "Also fetch Solar Orbiter proton density from SWA-PAS (SOLO_L2_SWA-PAS-GRND-MOM, N) for the same period",
            "Plot both on separate panels with title 'Solar Orbiter - March 2024'",
        ],
        "reset_before": True,
    },

    # ---- GROUP 13: Energetic particle data ----
    {
        "group": "Energetic Particle Data",
        "description": "Load PSP ISOIS energetic particle data",
        "commands": [
            "Fetch PSP ISOIS HET rates (PSP_ISOIS-EPIHI_L2-HET-RATES60) for 2024-05-01 to 2024-05-15",
            "List what parameters are available in that dataset",
            "Describe the fetched data",
        ],
        "reset_before": True,
    },

    # ---- GROUP 14: STEREO-A plasma + magnetic field ----
    {
        "group": "STEREO-A Plasma & Mag",
        "description": "Load STEREO-A combined data",
        "commands": [
            "Fetch STEREO-A magnetic field (STA_L2_MAG_RTN) for February 2024",
            "Also fetch STEREO-A proton bulk speed from PLASTIC (STA_L2_PLA_1DMAX_1MIN) for the same period",
            "Plot magnetic field on top panel and solar wind speed on bottom panel, and export to tests/plots/stereo_a_overview.png",
        ],
        "reset_before": True,
    },

    # ---- GROUP 15: Session save/load ----
    {
        "group": "Session Save & Load",
        "description": "Save session to .vap file and verify",
        "commands": [
            "Plot ACE magnetic field magnitude for January 2024",
            "Save the current session to tests/plots/test_session.vap",
        ],
        "reset_before": True,
    },

    # ---- GROUP 16: Autoplot script (advanced DOM manipulation) ----
    {
        "group": "Autoplot Script (DOM Manipulation)",
        "description": "Test autoplot_script tool for direct ScriptContext access",
        "commands": [
            "Fetch Wind magnetic field magnitude (WI_H2_MFI, BF1) for January 15-20, 2024",
            "Plot it in Autoplot",
            "Use an autoplot script to set the plot background color to light gray and change the line color to dark blue",
        ],
        "reset_before": True,
    },

    # ---- GROUP 17: Many datasets at once (stress test) ----
    {
        "group": "Many Datasets Stress Test",
        "description": "Try to load data from 5+ missions in one session",
        "commands": [
            "Fetch ACE mag field magnitude (AC_H2_MFI, Magnitude) for Jan 15-20 2024",
            "Fetch Wind mag field magnitude (WI_H2_MFI, BF1) for the same dates",
            "Fetch DSCOVR mag field magnitude (DSCOVR_H0_MAG, B1F1) for the same dates",
            "Fetch OMNI mag field magnitude (OMNI_HRO_1MIN, F) for the same dates",
            "Fetch STEREO-A magnetic field (STA_L2_MAG_RTN) for the same dates",
            "List all data currently in memory",
            "Overlay all 5 magnetic field datasets on a single plot titled '5-Mission Magnetic Field Comparison'",
        ],
        "reset_before": True,
    },

    # ---- GROUP 18: Color table testing ----
    {
        "group": "Color Table",
        "description": "Test setting color tables (mainly relevant for spectrograms but test the API)",
        "commands": [
            "Plot OMNI magnetic field magnitude for January 2024",
            "Try setting the color table to matlab_jet",
        ],
        "reset_before": True,
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Start from group N (1-based)")
    parser.add_argument("--end", type=int, default=len(TEST_GROUPS), help="End at group N (1-based)")
    args = parser.parse_args()

    all_results = []
    total_start = time.time()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for group_idx, group in enumerate(TEST_GROUPS):
        if group_idx + 1 < args.start or group_idx + 1 > args.end:
            continue

        group_name = group["group"]
        print(f"\n{'='*70}")
        print(f"  Group {group_idx+1}/{len(TEST_GROUPS)}: {group_name}")
        print(f"  {group['description']}")
        print(f"{'='*70}")

        if group.get("reset_before"):
            try:
                reset()
                print("  [conversation reset]")
            except (ConnectionRefusedError, OSError) as e:
                print(f"  [FATAL] Server connection failed: {e}")
                all_results.append({
                    "group": group_name,
                    "group_index": group_idx + 1,
                    "command_index": 0,
                    "command": "(reset)",
                    "response": "",
                    "error": f"Server down: {e}",
                    "tool_calls": [],
                    "tool_details": [],
                    "elapsed": 0,
                    "tokens": {},
                    "timestamp": datetime.now().isoformat(),
                    "status": "SERVER_DOWN",
                })
                continue

        group_results = []

        for cmd_idx, cmd in enumerate(group["commands"]):
            print(f"\n  [{group_idx+1}.{cmd_idx+1}] {cmd[:100]}{'...' if len(cmd)>100 else ''}")
            start = time.time()

            try:
                r = send(cmd)
                elapsed = round(time.time() - start, 2)

                result = {
                    "group": group_name,
                    "group_index": group_idx + 1,
                    "command_index": cmd_idx + 1,
                    "command": cmd,
                    "response": r.get("response", ""),
                    "error": r.get("error"),
                    "tool_calls": [tc["name"] for tc in r.get("tool_calls", [])],
                    "tool_details": r.get("tool_calls", []),
                    "elapsed": elapsed,
                    "tokens": r.get("tokens", {}),
                    "timestamp": datetime.now().isoformat(),
                }

                # Classify result
                error = r.get("error")
                resp = r.get("response", "")
                tools = result["tool_calls"]

                if error:
                    result["status"] = "ERROR"
                    print(f"       ERROR: {error}")
                elif "error" in resp.lower() and "no error" not in resp.lower():
                    result["status"] = "SOFT_ERROR"
                    print(f"       SOFT_ERROR in response")
                elif not tools:
                    result["status"] = "NO_TOOLS"
                    print(f"       No tool calls (agent responded with text only)")
                else:
                    result["status"] = "OK"

                print(f"       Tools: {', '.join(tools) if tools else 'none'}")
                print(f"       Response: {resp[:150]}{'...' if len(resp)>150 else ''}")
                print(f"       Time: {elapsed}s")

                group_results.append(result)

            except Exception as e:
                elapsed = round(time.time() - start, 2)
                print(f"       EXCEPTION: {e}")
                group_results.append({
                    "group": group_name,
                    "group_index": group_idx + 1,
                    "command_index": cmd_idx + 1,
                    "command": cmd,
                    "response": "",
                    "error": str(e),
                    "tool_calls": [],
                    "tool_details": [],
                    "elapsed": elapsed,
                    "tokens": {},
                    "timestamp": datetime.now().isoformat(),
                    "status": "EXCEPTION",
                })

        all_results.extend(group_results)

    total_elapsed = round(time.time() - total_start, 1)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    errors = [r for r in all_results if r["status"] == "ERROR"]
    soft_errors = [r for r in all_results if r["status"] == "SOFT_ERROR"]
    exceptions = [r for r in all_results if r["status"] == "EXCEPTION"]
    no_tools = [r for r in all_results if r["status"] == "NO_TOOLS"]
    ok = [r for r in all_results if r["status"] == "OK"]
    slow = [r for r in all_results if r["elapsed"] > 120]

    print(f"  Total commands:     {len(all_results)}")
    print(f"  OK:                 {len(ok)}")
    print(f"  Errors:             {len(errors)}")
    print(f"  Soft errors:        {len(soft_errors)}")
    print(f"  Exceptions:         {len(exceptions)}")
    print(f"  No tool calls:      {len(no_tools)}")
    print(f"  Slow (>120s):       {len(slow)}")
    print(f"  Total time:         {total_elapsed}s")

    if errors:
        print(f"\n  --- Errors ---")
        for r in errors:
            print(f"  [{r['group_index']}.{r['command_index']}] {r['group']}: {r['error'][:100]}")

    if soft_errors:
        print(f"\n  --- Soft Errors (error in response text) ---")
        for r in soft_errors:
            snippet = r['response'][:120]
            print(f"  [{r['group_index']}.{r['command_index']}] {r['group']}: {snippet}")

    if exceptions:
        print(f"\n  --- Exceptions ---")
        for r in exceptions:
            print(f"  [{r['group_index']}.{r['command_index']}] {r['group']}: {r['error'][:100]}")

    if slow:
        print(f"\n  --- Slow Commands (>120s) ---")
        for r in slow:
            print(f"  [{r['group_index']}.{r['command_index']}] {r['group']}: {r['elapsed']}s")

    # ---- Save results ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_commands": len(all_results),
        "total_elapsed": total_elapsed,
        "summary": {
            "ok": len(ok),
            "errors": len(errors),
            "soft_errors": len(soft_errors),
            "exceptions": len(exceptions),
            "no_tools": len(no_tools),
            "slow": len(slow),
        },
        "results": all_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Export dir: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
