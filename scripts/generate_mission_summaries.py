#!/usr/bin/env python3
"""
Generate enriched mission profile descriptions using Gemini + web search.

Reads each mission JSON from knowledge/missions/, extracts dataset info,
uses Gemini with Google Search grounding to research mission context, and
writes a ~2000-token summary into the profile.description field.

Usage:
    venv/bin/python scripts/generate_mission_summaries.py                    # All missions
    venv/bin/python scripts/generate_mission_summaries.py --mission PSP      # Single mission
    venv/bin/python scripts/generate_mission_summaries.py --dry-run          # Show what would be written
    venv/bin/python scripts/generate_mission_summaries.py --force            # Overwrite existing non-empty descriptions
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

MISSIONS_DIR = Path(__file__).parent.parent / "knowledge" / "missions" / "cdaweb"
DEFAULT_MODEL = "gemini-2.5-flash"
PLACEHOLDER_SUFFIX = "data from CDAWeb."
RATE_LIMIT_DELAY = 4  # seconds between API calls


def load_mission_json(mission_stem: str) -> dict:
    """Load a mission JSON file."""
    filepath = MISSIONS_DIR / f"{mission_stem}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Mission file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_mission_json(mission_stem: str, data: dict):
    """Save a mission JSON file preserving key order."""
    filepath = MISSIONS_DIR / f"{mission_stem}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved {filepath}")


def is_placeholder_description(description: str) -> bool:
    """Check if a profile description is the auto-generated placeholder."""
    if not description:
        return True
    return description.strip().endswith(PLACEHOLDER_SUFFIX)


def extract_dataset_summary(mission_data: dict) -> str:
    """Build a text summary of datasets from the mission JSON."""
    mission_name = mission_data.get("name", mission_data.get("id", "Unknown"))
    lines = [f"Mission: {mission_name}"]

    instruments = mission_data.get("instruments", {})
    for inst_id, inst in instruments.items():
        inst_name = inst.get("name", inst_id)
        datasets = inst.get("datasets", {})
        if not datasets:
            continue

        lines.append(f"\nInstrument: {inst_name}")
        for ds_id, ds_info in datasets.items():
            desc = ds_info.get("description", "")
            start = ds_info.get("start_date", "?")[:10]
            stop = ds_info.get("stop_date", "?")[:10]
            n_params = len(ds_info.get("parameters", []))
            lines.append(f"  - {ds_id}: {desc} [{start} to {stop}, {n_params} params]")

            # Include parameter names for context
            params = ds_info.get("parameters", [])
            if params:
                param_names = [p.get("name", "") for p in params[:15]]
                if len(params) > 15:
                    param_names.append(f"... +{len(params) - 15} more")
                lines.append(f"    Parameters: {', '.join(param_names)}")

    return "\n".join(lines)


def generate_summary(client, model: str, mission_data: dict) -> str:
    """Use Gemini with Google Search grounding to generate a mission summary."""
    from google.genai import types

    mission_name = mission_data.get("name", mission_data.get("id", "Unknown"))
    dataset_summary = extract_dataset_summary(mission_data)

    prompt = f"""You are writing a concise mission reference for a data analysis AI assistant.
The AI helps users fetch and plot data from NASA's CDAWeb (Coordinated Data Analysis Web).

Given the following information about the {mission_name} mission, write a dense ~2000-token
summary covering ALL of the following topics:

1. Mission overview: full name, agency (NASA/ESA/JAXA/etc.), launch date, mission status (active/ended), orbit type and parameters
2. Key instruments: what each measures, typical cadences, measurement techniques
3. Coordinate systems: which coordinate frames the data uses (RTN, GSE, GSM, HCI, spacecraft coords, etc.) and when each applies
4. Data date range: when data is available, any significant gaps
5. Known data caveats: fill values, saturation limits, pointing issues, known calibration notes
6. Common science use cases: solar wind studies, CME analysis, magnetic reconnection, SEP events, etc.

CDAWeb dataset information:
{dataset_summary}

IMPORTANT GUIDELINES:
- Write plain text, no markdown headers. Use bullet points sparingly (only for lists of instruments or coordinate systems).
- Be factual and specific — include numbers (cadences in seconds, orbit distances in solar radii or RE, etc.).
- Focus on information useful for someone querying and analyzing the data programmatically.
- Do NOT include general background about space physics — focus on this specific mission's data.
- If the mission has ended, note the final date of data availability.
"""

    # Use Google Search grounding for up-to-date mission information
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            temperature=0.3,
        ),
    )

    if response.candidates and response.candidates[0].content.parts:
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        return "\n".join(text_parts).strip()

    return ""


def process_mission(client, model: str, mission_stem: str, dry_run: bool, force: bool) -> bool:
    """Process a single mission. Returns True if updated."""
    mission_data = load_mission_json(mission_stem)
    mission_name = mission_data.get("name", mission_data.get("id", mission_stem))
    profile = mission_data.get("profile", {})
    current_desc = profile.get("description", "")

    # Check if we should skip
    if not force and not is_placeholder_description(current_desc):
        print(f"  [{mission_stem}] Skipping — already has a description ({len(current_desc)} chars). Use --force to overwrite.")
        return False

    # Check if mission has any datasets
    instruments = mission_data.get("instruments", {})
    total_datasets = sum(len(inst.get("datasets", {})) for inst in instruments.values())
    if total_datasets == 0:
        print(f"  [{mission_stem}] Skipping — no datasets found")
        return False

    print(f"  [{mission_stem}] Generating summary for {mission_name} ({total_datasets} datasets)...")

    if dry_run:
        summary = extract_dataset_summary(mission_data)
        print(f"  [DRY RUN] Would send {len(summary)} chars of dataset context to Gemini")
        print(f"  [DRY RUN] Current description: {current_desc[:80]}...")
        return False

    summary = generate_summary(client, model, mission_data)
    if not summary:
        print(f"  [{mission_stem}] WARNING: Gemini returned empty summary")
        return False

    # Update the profile description
    profile["description"] = summary
    mission_data["profile"] = profile
    save_mission_json(mission_stem, mission_data)
    print(f"  [{mission_stem}] Written {len(summary)} chars")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate enriched mission profile descriptions using Gemini + web search"
    )
    parser.add_argument(
        "--mission",
        type=str,
        help="Process only this mission (e.g., PSP, ACE). Case-insensitive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls or writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-empty descriptions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: GOOGLE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = None
    if not args.dry_run:
        from google import genai
        client = genai.Client(api_key=api_key)

    # Determine which missions to process
    if args.mission:
        stems = [args.mission.lower()]
        # Verify it exists
        filepath = MISSIONS_DIR / f"{stems[0]}.json"
        if not filepath.exists():
            print(f"Error: Mission file not found: {filepath}")
            sys.exit(1)
    else:
        stems = sorted(p.stem for p in MISSIONS_DIR.glob("*.json"))

    print(f"Processing {len(stems)} mission(s)...")
    updated = 0
    skipped = 0

    for i, stem in enumerate(stems):
        try:
            was_updated = process_mission(client, args.model, stem, args.dry_run, args.force)
            if was_updated:
                updated += 1
                # Rate limit between API calls
                if i < len(stems) - 1:
                    time.sleep(RATE_LIMIT_DELAY)
            else:
                skipped += 1
        except FileNotFoundError as e:
            print(f"  [{stem}] Error: {e}")
            skipped += 1
        except Exception as e:
            print(f"  [{stem}] Error: {e}")
            skipped += 1

    print(f"\nDone! Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
