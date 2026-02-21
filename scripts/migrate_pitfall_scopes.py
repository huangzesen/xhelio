#!/usr/bin/env python3
"""One-time migration: assign scopes to existing pitfall memories.

Reads ~/.xhelio/memory.json, backs it up to memory.json.bak,
then assigns scope to each pitfall based on keyword regex matching.

Usage:
    python scripts/migrate_pitfall_scopes.py          # dry-run (default)
    python scripts/migrate_pitfall_scopes.py --apply   # write changes
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path so config is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_data_dir

MEMORY_PATH = get_data_dir() / "memory.json"

# Ordered rules: first match wins
SCOPE_RULES = [
    # Mission-specific patterns
    (re.compile(r"PSP|Parker Solar Probe|SWEAP|SPC|SPI|FIELDS|psp_", re.IGNORECASE), "mission:PSP"),
    # ACE: require dataset IDs or instrument names, not just "ACE" (which appears in routing advice)
    (re.compile(r"AC_H.*MFI|AC_H.*SWE|AC_H.*EPM|ACE.*EPAM|ACE.*SWICS", re.IGNORECASE), "mission:ACE"),
    (re.compile(r"Solar Orbiter|SolO|solo_", re.IGNORECASE), "mission:SOLO"),
    (re.compile(r"\bOMNI\b|OMNI_HRO|omni_", re.IGNORECASE), "mission:OMNI"),
    (re.compile(r"\bWind\b|WI_H|WI_K", re.IGNORECASE), "mission:WIND"),
    (re.compile(r"\bDSCOVR\b|dscovr_", re.IGNORECASE), "mission:DSCOVR"),
    (re.compile(r"\bMMS\b|mms\d|MMS_FPI|MMS_FGM", re.IGNORECASE), "mission:MMS"),
    (re.compile(r"STEREO|STA_|STB_|stereo_", re.IGNORECASE), "mission:STEREO"),
    (re.compile(r"Voyager|VG\d|voyager_", re.IGNORECASE), "mission:VOYAGER1"),
    (re.compile(r"Ulysses|SWOOPS|ulysses_", re.IGNORECASE), "mission:ULYSSES"),
    # Visualization patterns
    (re.compile(
        r"plot_data|style_plot|manage_plot|y_range|panel.*trace|Plotly|"
        r"subplot|axis.*label|legend|render|export.*png|export.*pdf|"
        r"color.*scheme|marker|line.*width",
        re.IGNORECASE,
    ), "visualization"),
]


def detect_scope(content: str) -> str:
    """Return the scope for a pitfall based on keyword matching."""
    for pattern, scope in SCOPE_RULES:
        if pattern.search(content):
            return scope
    return "generic"


def main():
    parser = argparse.ArgumentParser(description="Migrate pitfall scopes")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    args = parser.parse_args()

    if not MEMORY_PATH.exists():
        print(f"No memory file found at {MEMORY_PATH}")
        return

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = data.get("memories", [])
    changes = 0

    for m in memories:
        if m.get("type") != "pitfall":
            # Ensure all non-pitfall memories also have scope field
            if "scope" not in m:
                m["scope"] = "generic"
            continue

        old_scope = m.get("scope", "generic")
        new_scope = detect_scope(m.get("content", ""))

        if old_scope != new_scope or "scope" not in m:
            print(f"  [{old_scope} -> {new_scope}] {m.get('content', '')[:80]}")
            m["scope"] = new_scope
            changes += 1
        else:
            m.setdefault("scope", "generic")

    print(f"\n{changes} pitfall(s) would be updated out of "
          f"{sum(1 for m in memories if m.get('type') == 'pitfall')} total pitfalls.")

    if args.apply and changes > 0:
        # Backup
        backup_path = MEMORY_PATH.with_suffix(".json.bak")
        shutil.copy2(MEMORY_PATH, backup_path)
        print(f"Backup saved to {backup_path}")

        # Write
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {MEMORY_PATH}")
    elif not args.apply and changes > 0:
        print("\nDry run — use --apply to write changes.")


if __name__ == "__main__":
    main()
