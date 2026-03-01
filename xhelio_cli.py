#!/usr/bin/env python3
"""XHelio CLI — launch XHelio from anywhere.

Usage:
    xhelio              Launch backend + frontend, open browser
    xhelio serve        Backend only (uvicorn on :8000)
    xhelio cli          Text CLI (interactive agent)
    xhelio mcp          MCP server over stdio
    xhelio purge        Delete all sessions, memory, and caches
    xhelio --help       Show help
"""

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Project root is wherever this file lives (works with editable installs)
PROJECT_ROOT = Path(__file__).resolve().parent


def _is_port_open(port: int, host: str = "localhost") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_open(port, host):
            return True
        time.sleep(0.3)
    return False


def _kill_port(port: int):
    """Kill any process listening on the given port and wait for it to be freed."""
    if platform.system() == "Windows":
        return
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip()
        if pids:
            for pid in pids.split("\n"):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ValueError, ProcessLookupError):
                    pass
            # Wait for port to be freed (up to 5s)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _is_port_open(port):
                    return
                time.sleep(0.2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _open_browser(url: str):
    """Open URL in default browser (cross-platform)."""
    import webbrowser
    webbrowser.open(url)


_PROVIDER_ENV_KEYS = {
    "1": ("gemini", "GOOGLE_API_KEY"),
    "2": ("openai", "OPENAI_API_KEY"),
    "3": ("anthropic", "ANTHROPIC_API_KEY"),
}


def _ensure_env():
    """Check .env exists; prompt for provider and API key if missing."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        return
    print()
    print("No .env file found. Creating one — you need at least one API key.")
    print()
    print("Which LLM provider will you use?")
    print("  1. Gemini  (default — https://ai.google.dev/)")
    print("  2. OpenAI / OpenAI-compatible (OpenRouter, DeepSeek, etc.)")
    print("  3. Anthropic Claude")
    print()
    try:
        choice = input("Provider [1]: ").strip() or "1"
        if choice not in _PROVIDER_ENV_KEYS:
            print(f"Invalid choice '{choice}', defaulting to Gemini.")
            choice = "1"
        provider, env_var = _PROVIDER_ENV_KEYS[choice]
        api_key = input(f"Enter your API key ({env_var}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)
    env_path.write_text(f"{env_var}={api_key}\n")
    print(f"Saved to .env (provider: {provider})")
    print()


def _find_python() -> str:
    """Return the Python executable to use.

    Prefer the project venv if it exists, otherwise fall back to the
    Python that is running this script (which is the installed one
    after pip install -e .).
    """
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    # Windows venv
    venv_python_win = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python_win.exists():
        return str(venv_python_win)
    return sys.executable


def _ensure_npm_deps():
    """Install frontend npm deps if missing."""
    frontend_dir = PROJECT_ROOT / "frontend"
    if not frontend_dir.exists():
        print(f"Error: frontend/ directory not found at {frontend_dir}")
        print("The web UI requires the source checkout. Use 'xhelio cli' for text mode.")
        sys.exit(1)
    if (frontend_dir / "node_modules").exists():
        return
    if not shutil.which("npm"):
        print("Error: npm not found. Install Node.js or use 'xhelio cli' for text mode.")
        sys.exit(1)
    print("Installing frontend dependencies ...")
    subprocess.run(
        ["npm", "install"],
        cwd=str(frontend_dir),
        check=True,
    )


def cmd_default(args):
    """Launch backend + frontend, open browser."""
    _ensure_env()

    if not shutil.which("npm"):
        print("Error: npm not found.")
        print("  Install Node.js (https://nodejs.org) for the web UI,")
        print("  or use 'xhelio cli' for the text-mode interface.")
        sys.exit(1)

    _ensure_npm_deps()

    python = _find_python()
    backend_port = args.port
    frontend_port = 5173

    # Kill stale process on backend port and wait for it to be free
    if _is_port_open(backend_port):
        print(f"Port {backend_port} in use — stopping old server ...")
        _kill_port(backend_port)
        # Wait for the port to be fully released (TIME_WAIT)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _is_port_open(backend_port):
            time.sleep(0.5)
        if _is_port_open(backend_port):
            print(f"Error: port {backend_port} still in use after 10s. "
                  f"Kill the process manually or use --port to pick another.")
            sys.exit(1)

    procs = []
    try:
        # Start backend
        print(f"Starting backend on :{backend_port} ...")
        backend = subprocess.Popen(
            [python, str(PROJECT_ROOT / "api_server.py"),
             "--port", str(backend_port), "--host", "0.0.0.0"],
            cwd=str(PROJECT_ROOT),
        )
        procs.append(backend)

        # Start frontend
        print(f"Starting frontend on :{frontend_port} ...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(PROJECT_ROOT / "frontend"),
        )
        procs.append(frontend)

        # Open browser after backend is ready
        if _wait_for_port(backend_port, timeout=30.0):
            time.sleep(1)  # Let Vite start too
            _open_browser(f"http://localhost:{frontend_port}")
        else:
            print("Warning: backend did not start within 30s")

        # Wait for either process to exit
        print("Running. Press Ctrl+C to stop.")
        while True:
            for p in procs:
                if p.poll() is not None:
                    raise KeyboardInterrupt  # Trigger cleanup
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down ...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def cmd_serve(args):
    """Start backend only."""
    _ensure_env()
    python = _find_python()
    if _is_port_open(args.port):
        print(f"Port {args.port} in use — stopping old server ...")
        _kill_port(args.port)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _is_port_open(args.port):
            time.sleep(0.5)
        if _is_port_open(args.port):
            print(f"Error: port {args.port} still in use.")
            sys.exit(1)

    cmd = [
        python, str(PROJECT_ROOT / "api_server.py"),
        "--port", str(args.port), "--host", args.host,
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        pass


def cmd_cli(args):
    """Start the text CLI."""
    _ensure_env()
    python = _find_python()

    cmd = [python, str(PROJECT_ROOT / "main.py")]
    if args.verbose:
        cmd.append("--verbose")
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        pass


def cmd_mcp(args):
    """Start the MCP server."""
    python = _find_python()

    cmd = [python, str(PROJECT_ROOT / "mcp_server.py")]
    if args.verbose:
        cmd.append("--verbose")
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    except KeyboardInterrupt:
        pass


def cmd_purge():
    """Delete all sessions, memory, pipelines, and caches."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import get_data_dir
    data_dir = get_data_dir()
    print(f"This will delete ALL sessions, memory, pipelines, and caches in {data_dir}")
    try:
        answer = input("Type PURGE to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer != "PURGE":
        print("Aborted.")
        return
    # Kill running backend
    _kill_port(8000)
    # Purge data directory
    if data_dir.exists():
        shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Purged all data in {data_dir}")
    else:
        print(f"Nothing to clean — {data_dir} does not exist.")
    # Remove legacy directories
    for legacy in (".helion", ".helio-agent"):
        legacy_dir = Path.home() / legacy
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
            print(f"Removed legacy directory {legacy_dir}")
    print("Done. Refresh the browser to start fresh (stale state is cleared automatically).")


def main():
    parser = argparse.ArgumentParser(
        prog="xhelio",
        description="XHelio — AI-powered scientific data visualization",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Backend port (default: 8000)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # xhelio serve
    serve_parser = subparsers.add_parser("serve", help="Backend only (uvicorn)")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")

    # xhelio cli
    cli_parser = subparsers.add_parser("cli", help="Text CLI (interactive agent)")
    cli_parser.add_argument("--verbose", "-v", action="store_true")

    # xhelio mcp
    mcp_parser = subparsers.add_parser("mcp", help="MCP server over stdio")
    mcp_parser.add_argument("--verbose", "-v", action="store_true")

    # xhelio purge
    subparsers.add_parser("purge", help="Delete all sessions, memory, and caches")

    args = parser.parse_args()

    if args.subcommand is None:
        cmd_default(args)
    elif args.subcommand == "serve":
        cmd_serve(args)
    elif args.subcommand == "cli":
        cmd_cli(args)
    elif args.subcommand == "mcp":
        cmd_mcp(args)
    elif args.subcommand == "purge":
        cmd_purge()


if __name__ == "__main__":
    main()
