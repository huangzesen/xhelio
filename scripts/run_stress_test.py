#!/usr/bin/env python3
"""
Stress test: launch 20 independent agent sessions with complicated prompts,
staggered by 1 second so each gets a distinct log file.

Usage:
    python scripts/run_stress_test.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_ROOT / "venv" / "bin" / "python")

PROMPTS = [
    # 1 — Multi-mission comparison
    "Compare ACE and Wind magnetic field data for the week of 2023-10-01 to 2023-10-07. Plot both on the same panel with different colors and add a legend.",

    # 2 — PSP perihelion deep dive
    "Show Parker Solar Probe magnetic field, solar wind speed, density, and radial distance for its closest perihelion in December 2024. Make a 4-panel plot.",

    # 3 — Solar Orbiter + computation
    "Fetch Solar Orbiter magnetometer data for March 2022 and compute the magnetic field magnitude. Then plot Br, Bt, Bn and |B| together.",

    # 4 — OMNI with derived quantities
    "Get OMNI solar wind data for the Halloween 2003 storm (Oct 28 to Nov 2, 2003). Plot Bz, dynamic pressure, and Dst index.",

    # 5 — Long time range with resampling
    "Show ACE magnetic field magnitude for all of 2023. Resample to daily averages and plot with a 27-day running average overlay.",

    # 6 — Multi-spacecraft timing
    "Fetch Wind and DSCOVR solar wind speed for January 2024. Compute the cross-correlation lag between them.",

    # 7 — MMS burst data
    "Get MMS1 magnetic field data in GSE coordinates for 2024-06-15 from 12:00 to 14:00 UTC. Plot all three components.",

    # 8 — Event search + annotation
    "Find a strong interplanetary shock in ACE data during 2023. Show magnetic field and solar wind speed around the event, and annotate the shock arrival time.",

    # 9 — STEREO-A beacon
    "Show STEREO-A magnetic field and plasma data for the first week of September 2022. Create a multi-panel plot with field components and proton density.",

    # 10 — Coordinate transformation request
    "Fetch PSP magnetic field in RTN coordinates for 2022-02-25 and convert the radial component to show the Parker spiral angle. Plot both.",

    # 11 — Statistical analysis
    "Get ACE solar wind speed for 2023 and create a histogram showing the distribution of fast and slow wind. Mark the 400 km/s boundary.",

    # 12 — Data gaps investigation
    "Show Solar Orbiter magnetometer data for July 2023. Identify and highlight any data gaps longer than 1 hour.",

    # 13 — Alfven speed computation
    "Fetch ACE magnetic field magnitude and proton density for 2024-03-01 to 2024-03-15. Compute and plot the Alfven speed.",

    # 14 — Multi-panel with shared x-axis
    "Create a 5-panel plot for PSP encounter 13 (September 2022): Br, Bt, Bn, |B|, and radial distance. Use shared time axis.",

    # 15 — CME event study
    "Show the July 2012 CME as seen by STEREO-A. Plot magnetic field and solar wind plasma parameters from July 20 to July 26, 2012.",

    # 16 — Sector boundary crossing
    "Find a heliospheric current sheet crossing in Wind data during 2024. Show the magnetic field polarity reversal with Bx component highlighted.",

    # 17 — Proton beta computation
    "Fetch ACE magnetic field and plasma data for April 2024. Compute and plot the proton beta parameter over time.",

    # 18 — Corotating interaction region
    "Show OMNI data for a corotating interaction region in 2023. Plot solar wind speed, density, temperature, and magnetic field magnitude showing the CIR structure.",

    # 19 — Export and styling
    "Get DSCOVR magnetic field data for the Gannon storm in May 2024. Create a publication-quality plot with customized fonts, colors, and export as PNG.",

    # 20 — Complex multi-step pipeline
    "Fetch PSP FIELDS and SWEAP data for encounter 17 (Sept 2023). Compute |B|, |V|, and the magnetic pressure. Plot all in a 4-panel layout with the radial distance as the bottom panel. Add annotations for perihelion.",
]


def main():
    processes = []

    print(f"Launching {len(PROMPTS)} agent sessions (1s apart)...\n")

    for i, prompt in enumerate(PROMPTS):
        label = f"[{i+1:2d}/{len(PROMPTS)}]"
        print(f"{label} Launching: {prompt[:70]}...")

        proc = subprocess.Popen(
            [PYTHON, str(PROJECT_ROOT / "main_direct.py"), prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        processes.append((i + 1, prompt[:70], proc, time.time()))

        # Wait 1 second so each session gets a distinct log filename
        if i < len(PROMPTS) - 1:
            time.sleep(1)

    print(f"\nAll {len(PROMPTS)} sessions launched. Waiting for completion...\n")

    # Wait for all to finish and report
    results = []
    for idx, label, proc, start_time in processes:
        proc.wait()
        elapsed = time.time() - start_time
        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
        results.append((idx, label, status, elapsed, proc.returncode))
        print(f"  [{idx:2d}] {status} ({elapsed:.1f}s) — {label}...")

    # Summary
    ok = sum(1 for r in results if r[4] == 0)
    fail = len(results) - ok
    print(f"\nDone: {ok} succeeded, {fail} failed out of {len(results)} total.")


if __name__ == "__main__":
    main()
