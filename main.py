#!/usr/bin/env python3
"""Helio AI Agent — main entry point.

Connects to the FastAPI backend over HTTP/SSE. If the server isn't
running, it is started automatically as a background process.

Usage:
    python main.py                          # Interactive mode (auto-starts server)
    python main.py --verbose                # Show tool calls
    python main.py --continue               # Resume most recent session
    python main.py --session ID             # Resume specific session
    python main.py "Show me ACE mag data"   # Single-command mode
    python main.py --url http://host:9000   # Custom server URL
    python main.py --no-color               # Disable ANSI colors

Slash commands (type /help for full list):
    /quit        - Extract memories, print token summary, exit
    /status      - Show session info (tokens, data, plan status)
    /data        - List fetched data entries
    /figure      - Save current figure to HTML and open in browser
    /help        - Show available commands
    /sessions    - List saved sessions on disk
    ... and more. Anything without a leading / is sent as a chat message.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import requests

# readline is optional (not available on Windows without pyreadline3)
try:
    import readline
    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False

# ---- ANSI colors ----

_USE_COLOR = True
_VERBOSE = False

# Accumulated token usage across the session
_total_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return _c("2", text)


def cyan(text: str) -> str:
    return _c("36", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


def bold(text: str) -> str:
    return _c("1", text)


# ---- Readline ----

def _history_path() -> str:
    from config import get_data_dir
    return os.path.join(str(get_data_dir()), ".cli_history")


_SLASH_COMMANDS = [
    ("/branch",   "Fork session into a new branch"),
    ("/cancel",   "Cancel current plan"),
    ("/data",     "List fetched data entries"),
    ("/errors",   "Show recent errors from logs"),
    ("/exit",     "Exit (alias for /quit)"),
    ("/figure",   "Save figure to HTML and open in browser"),
    ("/follow",   "Generate follow-up suggestions"),
    ("/help",     "Show available commands"),
    ("/quit",     "Extract memories, print token summary, exit"),
    ("/reset",    "Reset session (delete + create new)"),
    ("/retry",    "Retry failed plan task"),
    ("/sessions", "List saved sessions on disk"),
    ("/status",   "Show session info (tokens, data, plan)"),
]

_COMMAND_NAME_WIDTH = max(len(c[0]) for c in _SLASH_COMMANDS)


def _slash_completer(text, state):
    """Readline completer for slash commands."""
    if text.startswith("/"):
        matches = [c[0] for c in _SLASH_COMMANDS if c[0].startswith(text)]
    else:
        matches = []
    if state < len(matches):
        return matches[state]
    return None


def _display_matches(substitution, matches, longest_match_length):
    """Display slash command matches in a formatted table below the prompt."""
    buf = readline.get_line_buffer()
    print()
    for name, desc in _SLASH_COMMANDS:
        if name.startswith(buf):
            padded = name.ljust(_COMMAND_NAME_WIDTH + 4)
            if _USE_COLOR:
                print(f"  {_c('1', padded)}{_c('2', desc)}")
            else:
                print(f"  {padded}{desc}")
    # Re-display the prompt and current input
    print(cyan("\n> ") + buf, end="", flush=True)


def setup_readline():
    if not _READLINE_AVAILABLE:
        return
    readline.set_history_length(500)
    readline.set_completer(_slash_completer)
    readline.set_completer_delims(' \t\n')
    readline.parse_and_bind('tab: complete')
    readline.set_completion_display_function(_display_matches)
    try:
        readline.read_history_file(_history_path())
    except FileNotFoundError:
        pass


def save_readline():
    if not _READLINE_AVAILABLE:
        return
    try:
        path = _history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        readline.write_history_file(path)
    except Exception:
        pass


# ---- Server auto-start ----

def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _find_server_script() -> str:
    """Find api_server.py relative to this script."""
    return str(Path(__file__).resolve().parent / "api_server.py")


def ensure_server(url: str) -> bool:
    """If the server isn't running, start it as a detached background process.

    The server survives after the CLI exits (shared resource).
    Returns True if server is available.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    if _is_port_open(host, port):
        return True

    server_script = _find_server_script()
    if not Path(server_script).exists():
        print(red(f"Server script not found: {server_script}"))
        return False

    from config import get_data_dir
    log_path = os.path.join(str(get_data_dir()), "logs", "api_server.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"Server not running. Starting on port {port}...")
    log_file = open(log_path, "a")

    popen_kwargs = {
        "stdout": log_file,
        "stderr": log_file,
    }
    # Detach from CLI's process group so Ctrl+C doesn't kill the server
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, server_script, "--port", str(port), "--host", host],
        **popen_kwargs,
    )

    print(dim(f"  PID {proc.pid} (log: {log_path})"))

    if not _wait_for_port(host, port, timeout=30.0):
        if proc.poll() is not None:
            print(red(f"  Server exited with code {proc.returncode}. Check {log_path}"))
        else:
            print(red(f"  Timed out after 30s. Check {log_path}"))
        return False

    print(f"  Server ready.")
    return True


# ---- SSE parsing ----

def iter_sse_events(response: requests.Response):
    """Parse SSE events from a streaming requests response.

    Yields (event_type, data_dict) tuples.
    """
    event_type = "message"
    data_lines = []

    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue

        if line == "":
            # Empty line = end of event
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}
                yield event_type, data
            event_type = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # Ignore comments (lines starting with ':') and other fields


# ---- API helpers ----

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api{path}"

    def check_server(self) -> dict:
        resp = requests.get(self._url("/status"), timeout=5)
        resp.raise_for_status()
        return resp.json()

    def create_session(self) -> dict:
        resp = requests.post(self._url("/sessions"), timeout=30)
        resp.raise_for_status()
        info = resp.json()
        self.session_id = info["session_id"]
        return info

    def resume_session(self, saved_session_id: str) -> dict:
        resp = requests.post(
            self._url("/sessions/resume"),
            json={"session_id": saved_session_id},
            timeout=60,
        )
        resp.raise_for_status()
        info = resp.json()
        self.session_id = info["session_id"]
        return info

    def delete_session(self):
        if self.session_id:
            try:
                requests.delete(
                    self._url(f"/sessions/{self.session_id}"), timeout=5
                )
            except requests.RequestException:
                pass
            self.session_id = None

    def get_session(self) -> dict:
        resp = requests.get(
            self._url(f"/sessions/{self.session_id}"), timeout=5
        )
        resp.raise_for_status()
        return resp.json()

    def get_data(self) -> list:
        resp = requests.get(
            self._url(f"/sessions/{self.session_id}/data"), timeout=5
        )
        resp.raise_for_status()
        return resp.json()

    def get_figure(self) -> dict:
        resp = requests.get(
            self._url(f"/sessions/{self.session_id}/figure"), timeout=5
        )
        resp.raise_for_status()
        return resp.json()

    def get_follow_ups(self, count: int = 3) -> list:
        resp = requests.post(
            self._url(f"/sessions/{self.session_id}/follow-ups"),
            json={"count": count},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("suggestions", [])

    def get_saved_sessions(self) -> list:
        resp = requests.get(self._url("/sessions/saved"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_plan_status(self) -> dict:
        resp = requests.get(
            self._url(f"/sessions/{self.session_id}/plan"), timeout=5
        )
        resp.raise_for_status()
        return resp.json()

    def retry_plan(self) -> dict:
        resp = requests.post(
            self._url(f"/sessions/{self.session_id}/plan/retry"), timeout=120
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_plan(self) -> dict:
        resp = requests.post(
            self._url(f"/sessions/{self.session_id}/plan/cancel"), timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_errors(self, days: int = 7, limit: int = 10) -> dict:
        resp = requests.get(
            self._url("/errors"),
            params={"days": days, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def execute_command(self, command: str) -> dict:
        """Execute a slash command via the backend endpoint."""
        resp = requests.post(
            self._url(f"/sessions/{self.session_id}/command"),
            json={"command": command},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def chat_stream(self, message: str):
        """Send a chat message and yield SSE events."""
        resp = requests.post(
            self._url(f"/sessions/{self.session_id}/chat"),
            json={"message": message},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        yield from iter_sse_events(resp)


# ---- Event display ----

def display_event(event_type: str, data: dict, client: APIClient = None):
    """Display an SSE event to the terminal."""
    etype = data.get("type", event_type)

    if etype == "tool_call":
        if _VERBOSE:
            name = data.get("tool_name", "?")
            args = data.get("tool_args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(dim(f"  [Tool: {name}({args_str})]"))

    elif etype == "tool_result":
        if _VERBOSE:
            name = data.get("tool_name", "?")
            status = data.get("status", "?")
            marker = green("ok") if status == "success" else red(status)
            print(dim(f"  [Result: {name} -> {marker}]"))

    elif etype == "text_delta":
        text = data.get("text", "")
        if text:
            print(f"\n{text}")

    elif etype == "plot":
        if data.get("available") and client:
            print(yellow("\n  [Plot available -- opening in browser]"))
            cmd_figure(client)

    elif etype == "done":
        usage = data.get("token_usage", {})
        if usage:
            # Accumulate totals
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if key in usage:
                    _total_usage[key] += usage[key]
            if _VERBOSE:
                parts = []
                if "input_tokens" in usage:
                    parts.append(f"in={usage['input_tokens']}")
                if "output_tokens" in usage:
                    parts.append(f"out={usage['output_tokens']}")
                if "total_tokens" in usage:
                    parts.append(f"total={usage['total_tokens']}")
                if parts:
                    print(dim(f"  [{', '.join(parts)}]"))

    elif etype == "error":
        msg = data.get("message", str(data))
        print(red(f"\n  Error: {msg}"))


# ---- Commands ----

def cmd_help():
    print()
    print(bold("Slash commands:"))
    print(f"  {cyan('/quit')}       Exit (extracts memories, shows token summary)")
    print(f"  {cyan('/branch')}     Fork session into a new branch")
    print(f"  {cyan('/reset')}      Reset session (delete + create new)")
    print(f"  {cyan('/status')}     Show session info (tokens, data, plan)")
    print(f"  {cyan('/data')}       List fetched data entries")
    print(f"  {cyan('/figure')}     Save figure to HTML and open in browser")
    print(f"  {cyan('/follow')}     Generate follow-up suggestions")
    print(f"  {cyan('/sessions')}   List saved sessions on disk")
    print(f"  {cyan('/retry')}      Retry failed plan task")
    print(f"  {cyan('/cancel')}     Cancel current plan")
    print(f"  {cyan('/errors')}     Show recent errors from logs")
    print(f"  {cyan('/help')}       Show this message")
    print()
    print(dim("  Anything else is sent as a chat message to the agent."))


def cmd_status(client: APIClient):
    try:
        info = client.get_session()
        print(f"  Session:  {info['session_id']}")
        print(f"  Model:    {info['model']}")
        print(f"  Busy:     {info['busy']}")
        usage = info.get("token_usage", {})
        if usage:
            print(f"  Tokens:   {usage}")
        print(f"  Data:     {info.get('data_entries', 0)} entries")
        # Only show plan status if this session has an active plan
        plan = info.get("plan_status")
        if plan:
            print(f"  Plan:     {plan}")
            try:
                plan_detail = client.get_plan_status()
                plan_text = plan_detail.get("plan_status")
                if plan_text:
                    print()
                    print(plan_text)
            except requests.RequestException:
                pass
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_data(client: APIClient):
    try:
        entries = client.get_data()
        if not entries:
            print("  No data fetched yet.")
            return
        for entry in entries:
            label = entry.get("label", "?")
            shape = entry.get("shape", "")
            source = entry.get("source", "")
            print(f"  {label}  {dim(shape)}  {dim(source)}")
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def _save_figure_html(fig_json: dict, session_id: str) -> str:
    """Write a standalone Plotly HTML file and return the filename."""
    filename = f"figure_{session_id[:8]}.html"
    data_json = json.dumps(fig_json)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
</head><body>
<div id="fig" style="width:100%;height:100vh;"></div>
<script>Plotly.newPlot("fig", {data_json}.data, {data_json}.layout);</script>
</body></html>"""
    with open(filename, "w") as f:
        f.write(html)
    return filename


def cmd_figure(client: APIClient, open_browser: bool = True):
    try:
        result = client.get_figure()
        fig = result.get("figure")
        if fig is None:
            print("  No figure available.")
            return
        filename = _save_figure_html(fig, client.session_id)
        print(f"  Figure saved to {bold(filename)}")
        if open_browser:
            webbrowser.open(f"file://{os.path.abspath(filename)}")
            print(dim("  Opened in browser."))
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_follow(client: APIClient):
    try:
        print(dim("  Generating follow-ups..."))
        suggestions = client.get_follow_ups()
        if not suggestions:
            print("  No suggestions.")
            return
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_branch(client: APIClient):
    try:
        print(dim("  Branching session..."))
        result = client.execute_command("branch")
        new_id = result.get("data", {}).get("session_id")
        if new_id:
            print(f"  Branched to new session: {bold(new_id[:8])}")
        else:
            print(f"  {result.get('content', 'Branched.')}")
    except requests.HTTPError as e:
        print(red(f"  Error: {e}"))


def cmd_reset(client: APIClient):
    print(dim("  Resetting session..."))
    client.delete_session()
    info = client.create_session()
    print(f"  New session: {bold(info['session_id'][:8])}")


def cmd_sessions(client: APIClient):
    try:
        sessions = client.get_saved_sessions()
        if not sessions:
            print("  No saved sessions.")
            return
        shown = sessions[:10]
        print(f"  Saved sessions (showing {len(shown)}):")
        for s in shown:
            sid = s["id"]
            turns = s.get("turn_count", 0)
            preview = s.get("name") or (s.get("last_message_preview") or "")[:40]
            updated = (s.get("updated_at") or "")[:19]
            print(f"    {sid}  {turns} turns  {updated}  {preview}")
        print()
        print("  Resume with: python main.py --session <ID>")
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_retry(client: APIClient):
    try:
        print(dim("  Retrying failed task..."))
        result = client.retry_plan()
        print(f"  {result.get('result', 'Done')}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 409:
            print(red("  Session is busy."))
        else:
            print(red(f"  Error: {e}"))
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_cancel(client: APIClient):
    try:
        result = client.cancel_plan()
        print(f"  {result.get('result', 'Done')}")
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def cmd_errors(client: APIClient):
    try:
        result = client.get_errors(days=7, limit=10)
        errors = result.get("errors", [])
        if not errors:
            print("  No errors found in the last 7 days.")
            return
        print(f"  Recent errors ({len(errors)} found):")
        print("  " + "-" * 56)
        for i, err in enumerate(errors, 1):
            ts = err.get("timestamp", "?")
            level = err.get("level", "?")
            msg = err.get("message", "?")
            print(f"  {i}. [{ts}] {level}")
            print(f"     {msg}")
            details = err.get("details", [])
            for detail in details[:3]:
                print(f"     {detail}")
            if len(details) > 3:
                print(f"     ... and {len(details) - 3} more lines")
        print("  " + "-" * 56)
    except requests.RequestException as e:
        print(red(f"  Error: {e}"))


def print_welcome():
    print()
    print("=" * 60)
    print("  Helio AI Agent (API client)")
    print("=" * 60)
    print()
    print("I can help you explore and visualize spacecraft data")
    print("from heliophysics missions via CDAWeb.")
    print()
    print("Examples:")
    print("  'Show me ACE magnetic field data for last week'")
    print("  'Fetch Parker solar wind data and compute the magnitude'")
    print("  'Compare Wind and ACE magnetic field for January 2024'")
    print()
    print("Type /help for available commands.")
    print("-" * 60)


def print_token_summary():
    if _total_usage["total_tokens"] > 0:
        print()
        print("-" * 60)
        print("  Session token usage:")
        print(f"    Input tokens:  {_total_usage['input_tokens']:,}")
        print(f"    Output tokens: {_total_usage['output_tokens']:,}")
        print(f"    Total tokens:  {_total_usage['total_tokens']:,}")
        print("-" * 60)


# ---- Main ----

def main():
    global _USE_COLOR, _VERBOSE

    parser = argparse.ArgumentParser(
        description="CLI client for helio-agent FastAPI backend"
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        help="Single command to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show tool calls and per-turn token usage",
    )
    parser.add_argument(
        "--continue", "-c", dest="resume_latest", action="store_true",
        help="Resume the most recent saved session",
    )
    parser.add_argument(
        "--session", "-s", default=None,
        help="Resume a specific saved session by ID",
    )
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False
    _VERBOSE = args.verbose

    single_command = args.command
    interactive = single_command is None

    # Setup readline for interactive mode
    if interactive:
        setup_readline()

    client = APIClient(args.url)

    # Check server — auto-start if not running
    if interactive:
        print(f"Connecting to {args.url} ...")
    try:
        status = client.check_server()
        if interactive:
            print(
                f"Server OK -- {status.get('active_sessions', 0)} active sessions, "
                f"uptime {status.get('uptime_seconds', 0):.0f}s"
            )
    except requests.ConnectionError:
        if not ensure_server(args.url):
            print(red("Could not start the server. Exiting."))
            sys.exit(1)
        # Verify it's actually serving
        try:
            status = client.check_server()
            if interactive:
                print(
                    f"Server OK -- {status.get('active_sessions', 0)} active sessions"
                )
        except requests.RequestException as e:
            print(red(f"Server started but not responding: {e}"))
            sys.exit(1)
    except requests.RequestException as e:
        print(red(f"Server error: {e}"))
        sys.exit(1)

    # Session setup: resume or create new
    resumed = False
    if args.session:
        # Resume specific session
        if interactive:
            print(f"Resuming session {args.session} ...")
        try:
            info = client.resume_session(args.session)
            turns = info.get("turn_count", 0)
            preview = info.get("last_message_preview", "")
            if interactive:
                print(
                    f"Resumed session {bold(info['session_id'][:8])} "
                    f"(from {args.session[:8]}..., {turns} turns, model: {info['model']})"
                )
                if preview:
                    print(f"  Last message: {preview}")
            resumed = True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(red(f"Session not found: {args.session}"))
            elif e.response is not None and e.response.status_code == 429:
                print(red("Server at max sessions. Try again later."))
            else:
                print(red(f"Failed to resume session: {e}"))
            if not interactive:
                sys.exit(1)
            print("Starting new session instead.")

    elif args.resume_latest:
        # Resume most recent saved session
        if interactive:
            print("Looking for most recent saved session...")
        try:
            saved = client.get_saved_sessions()
            if saved:
                latest_id = saved[0]["id"]
                info = client.resume_session(latest_id)
                turns = info.get("turn_count", 0)
                preview = info.get("last_message_preview", "")
                if interactive:
                    print(
                        f"Resumed session {bold(info['session_id'][:8])} "
                        f"(from {latest_id[:8]}..., {turns} turns, model: {info['model']})"
                    )
                    if preview:
                        print(f"  Last message: {preview}")
                resumed = True
            else:
                if interactive:
                    print("No saved sessions found. Starting new session.")
        except requests.RequestException as e:
            print(red(f"Failed to resume: {e}"))
            if not interactive:
                sys.exit(1)
            print("Starting new session instead.")

    if not resumed:
        # Create new session
        if interactive:
            print("Creating session...")
        try:
            info = client.create_session()
            if interactive:
                print(
                    f"Session {bold(info['session_id'][:8])} created "
                    f"(model: {info['model']})"
                )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(red("Server at max sessions. Try again later."))
            else:
                print(red(f"Failed to create session: {e}"))
            sys.exit(1)

    # Single-command mode
    if single_command:
        try:
            for event_type, data in client.chat_stream(single_command):
                display_event(event_type, data, client)
        except requests.HTTPError as e:
            print(red(f"Error: {e}"))
            sys.exit(1)
        except requests.ConnectionError:
            print(red("Lost connection to server."))
            sys.exit(1)
        finally:
            print_token_summary()
            client.delete_session()
        return

    # Interactive mode
    if not resumed:
        print_welcome()
    else:
        print()
        print("Type /help for available commands.")
        print("-" * 60)

    try:
        while True:
            try:
                user_input = input(cyan("\n> ")).strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().strip()

                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    cmd_help()
                elif cmd == "branch":
                    cmd_branch(client)
                elif cmd == "reset":
                    cmd_reset(client)
                elif cmd == "status":
                    cmd_status(client)
                elif cmd == "data":
                    cmd_data(client)
                elif cmd == "figure":
                    cmd_figure(client)
                elif cmd == "follow":
                    cmd_follow(client)
                elif cmd == "sessions":
                    cmd_sessions(client)
                elif cmd == "retry":
                    cmd_retry(client)
                elif cmd == "cancel":
                    cmd_cancel(client)
                elif cmd == "errors":
                    cmd_errors(client)
                else:
                    print(red(f"  Unknown command: /{cmd}"))
                    print(dim("  Type /help for available commands."))
                continue

            # Send chat message
            try:
                for event_type, data in client.chat_stream(user_input):
                    display_event(event_type, data, client)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 409:
                    print(red("  Session is busy. Wait for the current request to finish."))
                else:
                    print(red(f"  Error: {e}"))
            except requests.ConnectionError:
                print(red("  Lost connection to server."))
            except KeyboardInterrupt:
                print(yellow("\n  Interrupted."))

    except KeyboardInterrupt:
        pass
    finally:
        print()

        print_token_summary()
        save_readline()
        client.delete_session()
        print("Session deleted. Goodbye.")


if __name__ == "__main__":
    main()
