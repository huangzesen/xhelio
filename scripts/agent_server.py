#!/usr/bin/env python3
"""
Interactive Agent Server for Multi-Turn Testing

Keeps an OrchestratorAgent alive in a background process so Claude Code (or any
client) can drive multi-turn conversations over a TCP socket.

Usage:
    python scripts/agent_server.py serve [--verbose] [--port PORT]
    python scripts/agent_server.py send "Show me ACE mag data for last week"
    python scripts/agent_server.py reset
    python scripts/agent_server.py stop
"""

import argparse
import io
import json
import os
import socket
import struct
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path so we can import agent.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Paths -------------------------------------------------------------------

from config import get_data_dir

HELIO_DIR = get_data_dir()
PORT_FILE = HELIO_DIR / "server.port"
SESSION_DIR = HELIO_DIR / "sessions"

# --- Protocol helpers ---------------------------------------------------------
# Messages are length-prefixed: 4-byte big-endian uint32 followed by JSON bytes.

def send_msg(sock: socket.socket, obj: dict) -> None:
    """Send a JSON object over a socket with length-prefix framing."""
    data = json.dumps(obj, default=str).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock: socket.socket) -> dict | None:
    """Receive a length-prefixed JSON object from a socket."""
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    if msg_len == 0:
        return None
    raw_data = _recv_exact(sock, msg_len)
    if not raw_data:
        return None
    return json.loads(raw_data.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from a socket, or return None on disconnect."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# --- Server -------------------------------------------------------------------

def cmd_serve(args):
    """Start the agent server."""
    # Create dirs
    HELIO_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize agent
    print("Initializing agent...")
    from agent.core import create_agent
    agent = create_agent(verbose=args.verbose, model=args.model)
    print("Agent ready.")

    # Bind socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)

    actual_port = srv.getsockname()[1]
    PORT_FILE.write_text(str(actual_port))
    print(f"Listening on 127.0.0.1:{actual_port}")

    # Session log
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = SESSION_DIR / f"session_{session_ts}.jsonl"
    print(f"Session log: {session_file}")
    sys.stdout.flush()

    turn = 0

    try:
        while True:
            srv.settimeout(None)  # block forever waiting for connections
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            conn.settimeout(300)  # 5 min timeout per request

            try:
                request = recv_msg(conn)
                if not request:
                    conn.close()
                    continue

                action = request.get("action", "")
                response = _handle_action(agent, action, request, args.verbose)

                # Track turns for send actions
                if action == "send":
                    turn += 1
                    response["turn"] = turn

                    # Append to session log
                    log_entry = {
                        "turn": turn,
                        "timestamp": datetime.now().isoformat(),
                        "message": request.get("message", ""),
                        "response": response.get("response", ""),
                        "tool_calls": response.get("tool_calls", []),
                        "follow_ups": response.get("follow_ups", []),
                        "elapsed": response.get("elapsed", 0),
                        "tokens": response.get("tokens", {}),
                        "error": response.get("error"),
                        "has_figure": response.get("figure_json") is not None,
                    }
                    with open(session_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, default=str) + "\n")

                send_msg(conn, response)

                # Handle stop action after sending response
                if action == "stop":
                    conn.close()
                    print("Stop requested. Shutting down.")
                    sys.stdout.flush()
                    _clean_shutdown()

            except socket.timeout:
                try:
                    send_msg(conn, {"error": "Request timed out (300s)", "response": ""})
                except Exception:
                    pass
            except Exception as e:
                try:
                    send_msg(conn, {"error": str(e), "response": ""})
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        srv.close()
        _cleanup_port_file()
        sys.stdout.flush()
        os._exit(0)


def _handle_action(agent, action: str, request: dict, verbose: bool) -> dict:
    """Dispatch a request action and return a response dict."""
    if action == "send":
        return _handle_send(agent, request.get("message", ""), verbose)
    elif action == "reset":
        agent.reset()
        return {"response": "Conversation reset.", "error": None}
    elif action == "new_session":
        return _handle_new_session(agent)
    elif action == "status":
        tokens = agent.get_token_usage()
        plan_status = agent.get_plan_status()
        return {
            "response": plan_status or "No active plan.",
            "tokens": tokens,
            "error": None,
        }
    elif action == "stop":
        return {"response": "Server stopping.", "error": None}
    else:
        return {"error": f"Unknown action: {action}", "response": ""}


def _handle_new_session(agent) -> dict:
    """Reset agent state AND clear the DataStore for a fresh session."""
    agent.reset()
    try:
        from data_ops.store import get_store
        get_store().clear()
    except Exception:
        pass
    return {"response": "New session started (agent + data store cleared).", "error": None}


def _handle_send(agent, message: str, verbose: bool) -> dict:
    """Process a user message through the agent with tool-call tracking."""
    if not message:
        return {"response": "", "error": "Empty message", "tool_calls": [], "elapsed": 0}

    tool_log = []

    # Monkey-patch _execute_tool_safe to track tool calls
    original_execute = agent._execute_tool_safe

    def tracking_execute(name, args):
        start = time.time()
        result = original_execute(name, args)
        tool_log.append({
            "name": name,
            "args": _sanitize_args(args),
            "elapsed": round(time.time() - start, 3),
            "status": result.get("status"),
        })
        return result

    agent._execute_tool_safe = tracking_execute

    try:
        start = time.time()

        # Capture verbose output
        if verbose:
            buf = io.StringIO()
            with redirect_stdout(buf):
                response_text = agent.process_message(message)
            verbose_log = buf.getvalue()
        else:
            response_text = agent.process_message(message)
            verbose_log = ""

        elapsed = round(time.time() - start, 2)
        tokens = agent.get_token_usage()

        # Generate follow-up suggestions
        follow_ups = []
        try:
            follow_ups = agent.generate_follow_ups()
        except Exception:
            pass

        # Snapshot current figure (None if no plot)
        # Skip if figure JSON would be too large (>50MB = likely high-cadence data)
        figure_json = None
        try:
            fig = agent.get_figure()
            if fig is not None:
                raw = fig.to_json()
                if len(raw) < 50_000_000:  # 50MB limit
                    figure_json = raw
        except Exception:
            pass

        return {
            "response": response_text,
            "tool_calls": tool_log,
            "follow_ups": follow_ups,
            "elapsed": elapsed,
            "tokens": tokens,
            "verbose_log": verbose_log,
            "figure_json": figure_json,
            "error": None,
        }

    except Exception as e:
        return {
            "response": "",
            "tool_calls": tool_log,
            "follow_ups": [],
            "elapsed": round(time.time() - start, 2),
            "tokens": agent.get_token_usage(),
            "verbose_log": "",
            "figure_json": None,
            "error": str(e),
        }
    finally:
        agent._execute_tool_safe = original_execute


def _sanitize_args(args: dict) -> dict:
    """Make tool args JSON-serializable (truncate large values)."""
    from agent.truncation import trunc
    sanitized = {}
    for k, v in args.items():
        sanitized[k] = trunc(str(v), "context.tool_args_sanitize")
    return sanitized


def _cleanup_port_file():
    """Remove the port file on shutdown."""
    try:
        PORT_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _clean_shutdown():
    """Shut down cleanly, avoiding JPype JVM hang."""
    _cleanup_port_file()
    sys.stdout.flush()
    os._exit(0)


# --- Client commands ----------------------------------------------------------

def _connect() -> socket.socket:
    """Connect to the running server. Raises SystemExit on failure."""
    if not PORT_FILE.exists():
        print("Server not running. Start with: python scripts/agent_server.py serve")
        sys.exit(1)

    try:
        port = int(PORT_FILE.read_text().strip())
    except (ValueError, OSError):
        print("Invalid port file. Delete it and restart the server.")
        _cleanup_port_file()
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(300)
    try:
        sock.connect(("127.0.0.1", port))
    except ConnectionRefusedError:
        print("Server not responding (stale port file). Start with: python scripts/agent_server.py serve")
        _cleanup_port_file()
        sys.exit(1)

    return sock


def cmd_send(args):
    """Send a message to the agent and print the response."""
    message = " ".join(args.message)
    if not message:
        print("Usage: python scripts/agent_server.py send \"your message\"")
        sys.exit(1)

    sock = _connect()
    try:
        send_msg(sock, {"action": "send", "message": message})
        response = recv_msg(sock)
    finally:
        sock.close()

    if not response:
        print("No response from server.")
        sys.exit(1)

    if response.get("error"):
        print(f"Error: {response['error']}")
        sys.exit(1)

    # Print agent response
    print(f"Agent: {response.get('response', '')}")

    # Print summary
    tool_calls = response.get("tool_calls", [])
    elapsed = response.get("elapsed", 0)
    tokens = response.get("tokens", {})

    if tool_calls:
        names = [tc["name"] for tc in tool_calls]
        print(f"\n  Tools: {', '.join(names)} ({len(names)} call{'s' if len(names) != 1 else ''}, {elapsed}s)")
    else:
        print(f"\n  ({elapsed}s, no tool calls)")

    in_tok = tokens.get("input_tokens", 0)
    out_tok = tokens.get("output_tokens", 0)
    if in_tok or out_tok:
        print(f"  Tokens: {in_tok:,} in / {out_tok:,} out")


def cmd_reset(args):
    """Reset the agent conversation."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "reset"})
        response = recv_msg(sock)
    finally:
        sock.close()

    print(response.get("response", "Reset complete.") if response else "No response.")


def cmd_stop(args):
    """Stop the server."""
    sock = _connect()
    try:
        send_msg(sock, {"action": "stop"})
        response = recv_msg(sock)
    finally:
        sock.close()

    print(response.get("response", "Server stopped.") if response else "Server stopped.")


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Agent Server for multi-turn testing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the agent server")
    p_serve.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    p_serve.add_argument("--model", "-m", default=None, help="Gemini model name (default: gemini-2.5-flash)")
    p_serve.add_argument("--port", type=int, default=0, help="Port (default: auto)")
    p_serve.set_defaults(func=cmd_serve)

    # send
    p_send = subparsers.add_parser("send", help="Send a message to the agent")
    p_send.add_argument("message", nargs="+", help="Message text")
    p_send.set_defaults(func=cmd_send)

    # reset
    p_reset = subparsers.add_parser("reset", help="Reset conversation")
    p_reset.set_defaults(func=cmd_reset)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop the server")
    p_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
