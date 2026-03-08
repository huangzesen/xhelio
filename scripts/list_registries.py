#!/usr/bin/env python3
"""List all registered registries and their entry counts."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force all registry-bearing modules to import
import agent.truncation        # noqa: F401
import agent.turn_limits       # noqa: F401
import agent.fallback_registry # noqa: F401
import agent.tool_handlers     # noqa: F401
import agent.tools             # noqa: F401
import agent.event_bus         # noqa: F401
import agent.event_formatters  # noqa: F401
import agent.observations      # noqa: F401
import agent.agent_registry    # noqa: F401
import rendering.registry      # noqa: F401
import config                  # noqa: F401
try:
    import data_ops.custom_ops  # noqa: F401
except ImportError as e:
    print(f"Warning: could not import data_ops.custom_ops: {e}", file=sys.stderr)

from agent.registry_protocol import list_registries


def main():
    regs = list_registries()
    print(f"\n{'Name':<30} {'Entries':>7}  Description")
    print("-" * 75)
    for name in sorted(regs):
        reg = regs[name]
        count = len(reg.list_all())
        print(f"{name:<30} {count:>7}  {reg.description}")
    print(f"\nTotal: {len(regs)} registries")


if __name__ == "__main__":
    main()
