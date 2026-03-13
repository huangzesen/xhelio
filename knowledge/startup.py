"""Mission data startup utilities."""

import json
from pathlib import Path


def get_mission_status() -> dict:
    """Scan envoy JSONs and return a status summary."""
    envoys_dir = Path(__file__).parent / "envoys"
    mission_files = []
    if envoys_dir.exists():
        for kind_dir in sorted(envoys_dir.iterdir()):
            if kind_dir.is_dir() and not kind_dir.name.startswith("_"):
                mission_files.extend(sorted(kind_dir.glob("*.json")))

    names = []
    for f in mission_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            names.append(data.get("id", f.stem))
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "mission_count": len(names),
        "mission_names": names,
    }


def run_background_load() -> None:
    """No-op — envoys are registered at startup, not loaded from JSON."""
    from knowledge.loading_state import get_loading_state, LoadingPhase

    loading = get_loading_state()
    loading.update(phase=LoadingPhase.COMPLETE, pct=100, message="Ready")
