#!/usr/bin/env python3
"""
Helio AI Agent — direct mode (DEPRECATED).

This runs the agent directly without the FastAPI backend.
Use `python main.py` instead, which auto-starts the API server
and provides a richer CLI experience.

Usage:
    python main_direct.py              # Normal mode
    python main_direct.py --verbose    # Show tool execution details
    python main_direct.py --continue   # Resume most recent session
    python main_direct.py --session ID # Resume specific session
"""

import sys
import argparse
from pathlib import Path
from config import get_data_dir
from knowledge.startup import resolve_refresh_flags

# readline is optional (not available on Windows without pyreadline3)
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

HISTORY_FILE = get_data_dir() / ".history"


def setup_readline():
    """Configure readline for input history."""
    if not READLINE_AVAILABLE:
        return
    readline.set_history_length(500)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def print_welcome():
    """Print welcome message."""
    print("=" * 60)
    print("  Helio AI Agent")
    print("=" * 60)
    print()
    print("I can help you explore and visualize spacecraft data")
    print("from heliophysics missions via CDAWeb.")
    print()
    print("What I can do:")
    print("  Search & plot    - Find datasets and plot them instantly")
    print("  Compute          - Magnitude, smoothing, derivatives, etc.")
    print("  Describe data    - Statistical summaries of fetched data")
    print("  Export           - Save plots to PNG, data to CSV")
    print("  Multi-step tasks - Complex requests broken into steps")
    print()
    print("Examples:")
    print("  'Show me ACE magnetic field data for last week'")
    print("  'Fetch Parker solar wind data and compute the magnitude'")
    print("  'Describe the data'")
    print("  'Save the data to a CSV file'")
    print("  'Compare Wind and ACE magnetic field for January 2024'")
    print()
    print("Commands: quit, reset, status, retry, cancel, errors,")
    print("          capabilities, sessions, help")
    print("-" * 60)
    print()


def print_capabilities():
    """Print detailed capability summary from docs."""
    docs_path = Path(__file__).parent / "docs" / "capability-summary.md"
    if not docs_path.exists():
        print("Capability summary not found.")
        return
    print()
    print("=" * 60)
    with open(docs_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 60)
    print()


def check_incomplete_plans(agent, verbose: bool):
    """Check for incomplete plans from previous sessions and offer to resume."""
    from agent.tasks import get_task_store

    store = get_task_store()
    incomplete = store.get_incomplete_plans()

    if not incomplete:
        return

    # Get the most recent incomplete plan
    plan = sorted(incomplete, key=lambda p: p.created_at, reverse=True)[0]

    print("-" * 60)
    print("Found incomplete plan from previous session:")
    print(f"  Request: {plan.user_request[:60]}...")
    print(f"  Status: {plan.progress_summary()}")
    print()

    while True:
        choice = input("Resume (r), discard (d), or ignore (i)? ").strip().lower()
        if choice in ("r", "resume"):
            print()
            result = agent.resume_plan(plan)
            print(f"Agent: {result}")
            print()
            break
        elif choice in ("d", "discard"):
            result = agent.discard_plan(plan)
            print(result)
            print()
            break
        elif choice in ("i", "ignore"):
            print("Ignoring incomplete plan.")
            print()
            break
        else:
            print("Please enter 'r' to resume, 'd' to discard, or 'i' to ignore.")


def main():
    """Main conversation loop."""
    parser = argparse.ArgumentParser(description="Helio AI Agent")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show tool execution details",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch with visible GUI window for interactive plots",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model name for the orchestrator (overrides 'model' in config.json)",
    )
    parser.add_argument(
        "--continue", "-c",
        dest="resume_latest",
        action="store_true",
        help="Resume the most recent session",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Resume a specific session by ID",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh dataset time ranges (fast — updates start/stop dates only)",
    )
    parser.add_argument(
        "--refresh-full",
        action="store_true",
        help="Full rebuild of primary mission data (re-download everything)",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Download ALL missions from CDAWeb (full rebuild)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Single command to execute (non-interactive mode)",
    )
    args = parser.parse_args()

    # Skip welcome message and readline in single-command mode
    if not args.command:
        setup_readline()
        print_welcome()

        # Mission data menu (skip in single-command mode)
        resolve_refresh_flags(
            refresh=args.refresh,
            refresh_full=args.refresh_full,
            refresh_all=args.refresh_all,
        )

    print("Data backend: CDF (direct file download)")

    # Import here to delay JVM startup until user is ready
    try:
        from agent.core import create_agent
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you've installed dependencies: pip install -r requirements.txt")
        sys.exit(1)

    try:
        agent = create_agent(verbose=args.verbose, gui_mode=args.gui, model=args.model)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure LLM_API_KEY is set in .env file")
        print("  2. Check that the required SDK is installed (google-genai, openai, or anthropic)")
        sys.exit(1)

    # Single command mode (non-interactive)
    if args.command:
        print(f"You: {args.command}\n")
        response = agent.process_message(args.command)
        print(f"Agent: {response}\n")

        # Print token usage and exit
        usage = agent.get_token_usage()
        if usage["api_calls"] > 0:
            print("-" * 60)
            cached = usage.get('cached_tokens', 0)
            cache_str = f", cached: {cached:,}" if cached else ""
            print(f"  Tokens: {usage['total_tokens']:,} (in: {usage['input_tokens']:,}, out: {usage['output_tokens']:,}{cache_str})")
            print("-" * 60)

        agent.close()
        sys.stdout.flush()
        import os
        os._exit(0)

    if args.gui:
        print("GUI Mode: Plot window will appear when plotting.")

    print(f"Model: {agent.model_name}")

    # Session setup
    if args.session:
        try:
            meta, _dlog, _elog = agent.load_session(args.session)
            turns = meta.get("turn_count", 0)
            preview = meta.get("last_message_preview", "")
            from data_ops.store import get_store
            data_count = len(get_store())
            print(f"Resumed session: {args.session}")
            print(f"  {turns} turns, {data_count} data entries in memory")
            if preview:
                print(f"  Last message: {preview}")
        except FileNotFoundError:
            print(f"Session not found: {args.session}")
            print("Starting new session.")
            agent.start_session()
    elif args.resume_latest:
        from agent.session import SessionManager
        sm = SessionManager()
        latest = sm.get_most_recent_session()
        if latest:
            try:
                meta, _dlog, _elog = agent.load_session(latest)
                turns = meta.get("turn_count", 0)
                preview = meta.get("last_message_preview", "")
                from data_ops.store import get_store
                data_count = len(get_store())
                print(f"Resumed session: {latest}")
                print(f"  {turns} turns, {data_count} data entries in memory")
                if preview:
                    print(f"  Last message: {preview}")
            except Exception as e:
                print(f"Could not resume session: {e}")
                print("Starting new session.")
                agent.start_session()
        else:
            print("No previous sessions found. Starting new session.")
            agent.start_session()
    else:
        agent.start_session()
    print()

    # Check for incomplete plans from previous sessions
    check_incomplete_plans(agent, args.verbose)

    print("Agent ready. Type your request:\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation reset.\n")
                continue

            if user_input.lower() == "help":
                print_welcome()
                continue

            if user_input.lower() == "status":
                status = agent.get_plan_status()
                if status:
                    print(status)
                else:
                    print("No active or incomplete plans.")
                print()
                continue

            if user_input.lower() == "retry":
                result = agent.retry_failed_task()
                print(result)
                print()
                continue

            if user_input.lower() == "cancel":
                result = agent.cancel_plan()
                print(result)
                print()
                continue

            if user_input.lower() == "errors":
                from agent.logging import print_recent_errors
                print_recent_errors(days=7, limit=10)
                print()
                continue

            if user_input.lower() in ("capabilities", "caps"):
                print_capabilities()
                continue

            if user_input.lower() == "sessions":
                from agent.session import SessionManager
                sm = SessionManager()
                sessions = sm.list_sessions()[:10]
                if not sessions:
                    print("No saved sessions.")
                else:
                    print(f"Saved sessions (most recent {len(sessions)}):")
                    for s in sessions:
                        sid = s["id"]
                        turns = s.get("turn_count", 0)
                        preview = s.get("last_message_preview", "")[:40]
                        updated = s.get("updated_at", "")[:19]
                        current = " (current)" if sid == agent.get_session_id() else ""
                        print(f"  {sid}  {turns} turns  {updated}  {preview}{current}")
                    print()
                    print("Resume with: python main.py --session <ID>")
                print()
                continue

            # Process the message
            print()
            response = agent.process_message(user_input)
            print(f"Agent: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You can continue the conversation or type 'reset' to start fresh.\n")

    # Print token usage summary and log session end
    usage = agent.get_token_usage()
    if usage["api_calls"] > 0:
        print()
        print("-" * 60)
        print(f"  Session token usage:")
        print(f"    Input tokens:    {usage['input_tokens']:,}")
        print(f"    Output tokens:   {usage['output_tokens']:,}")
        print(f"    Thinking tokens: {usage['thinking_tokens']:,}")
        cached = usage.get('cached_tokens', 0)
        if cached:
            print(f"    Cached tokens:   {cached:,}")
        print(f"    Total tokens:    {usage['total_tokens']:,}")
        print(f"    API calls:       {usage['api_calls']}")
        print("-" * 60)

        # Log session end
        from agent.logging import log_session_end
        log_session_end(usage)

    # Clean shutdown — delete explicit caches to stop storage charges
    agent.close()
    readline.write_history_file(HISTORY_FILE)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
