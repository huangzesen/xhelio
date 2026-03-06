"""
MCP client singleton for the SPICE ephemeris server.

Manages a persistent subprocess running `heliospice-mcp` (from the
heliospice package) and provides a synchronous call_tool() interface
for the agent core.

The subprocess is spawned lazily on first use and kept alive for the
agent's lifetime. A background daemon thread runs the async event loop.
"""

import atexit
import json
import os
import sys
import threading
from datetime import timedelta

from .event_bus import get_event_bus, DEBUG

# ---------------------------------------------------------------------------
# SpiceMCPClient
# ---------------------------------------------------------------------------

class SpiceMCPClient:
    """Manages a persistent MCP connection to the heliospice MCP server."""

    def __init__(self):
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None
        self._lock = threading.Lock()
        self._closed = False
        # Context manager stacks for cleanup
        self._stdio_cm = None
        self._session_cm = None
        # Discovered tools from MCP server
        self._discovered_tools: list[dict] = []

    def _start(self):
        """Spawn the background thread that runs the MCP session."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._error:
            raise RuntimeError(f"SPICE MCP server failed to start: {self._error}")

    def _run_loop(self):
        """Background thread: run the async event loop with the MCP session."""
        import asyncio

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_connect())
            # Keep the loop alive for call_tool requests
            loop.run_forever()
        except Exception as e:
            self._error = str(e)
            self._ready.set()
        finally:
            # Cleanup
            try:
                loop.run_until_complete(self._async_cleanup())
            except Exception:
                pass
            loop.close()

    async def _async_connect(self):
        """Establish the MCP stdio connection (runs in background thread)."""
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.session import ClientSession

        # Always use the current interpreter to launch the MCP server.
        # This avoids stale shebangs in venv entry-point scripts and
        # accidentally picking up a system/conda copy via PATH.
        command = sys.executable
        args = ["-m", "heliospice.server"]

        # Pass kernel dir from Helion config if available
        env = {**os.environ}
        if "HELIOSPICE_KERNEL_DIR" not in env:
            try:
                from config import get_data_dir
                env["HELIOSPICE_KERNEL_DIR"] = str(get_data_dir() / "spice_kernels")
            except Exception:
                pass

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        # Enter stdio_client context manager
        self._stdio_cm = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._stdio_cm.__aenter__()

        # Enter ClientSession context manager
        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()

        await self._session.initialize()

        # Discover available tools from the MCP server
        tools_result = await self._session.list_tools()
        self._discovered_tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
            }
            for t in tools_result.tools
        ]

        self._ready.set()

    async def _async_cleanup(self):
        """Clean up MCP session and stdio transport."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass

    def call_tool(self, name: str, args: dict, timeout: float = 120) -> dict:
        """Call an MCP tool synchronously.

        Args:
            name: Tool name (e.g., "get_spacecraft_position")
            args: Tool arguments dict
            timeout: Timeout in seconds

        Returns:
            Parsed dict from the tool's JSON response.

        Raises:
            RuntimeError: If the MCP session is not connected or tool call fails.
        """
        import asyncio

        if self._closed:
            raise RuntimeError("SPICE MCP client has been closed")

        if self._session is None or self._loop is None:
            raise RuntimeError("SPICE MCP client not connected")

        async def _call():
            result = await self._session.call_tool(
                name=name,
                arguments=args,
                read_timeout_seconds=timedelta(seconds=timeout),
            )
            if result.isError:
                error_text = result.content[0].text if result.content else "Unknown MCP error"
                return {"status": "error", "message": error_text}

            # The SPICE MCP tools return JSON-serializable dicts via FastMCP.
            # FastMCP serializes the return value as JSON text in a TextContent block.
            if result.content:
                for block in result.content:
                    if hasattr(block, "text"):
                        try:
                            return json.loads(block.text)
                        except (json.JSONDecodeError, TypeError):
                            return {"status": "success", "text": block.text}

            return {"status": "success", "text": ""}

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        return future.result(timeout=timeout)

    def close(self):
        """Shut down the MCP session and background thread."""
        if self._closed:
            return
        self._closed = True
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def get_tool_schemas(self) -> list[dict]:
        """Return discovered tools in the agent's schema format.

        Each dict has 'name', 'description', and 'parameters' keys
        matching the format used by agent/tools.py TOOLS list.
        """
        schemas = []
        for t in self._discovered_tools:
            schema = {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("inputSchema", {
                    "type": "object", "properties": {}, "required": []
                }),
            }
            schemas.append(schema)
        return schemas

    def get_tool_names(self) -> set[str]:
        """Return the set of discovered tool names for quick lookup."""
        return {t["name"] for t in self._discovered_tools}

    def is_connected(self) -> bool:
        """Check if the client has an active session."""
        return (
            self._session is not None
            and self._loop is not None
            and self._loop.is_running()
            and not self._closed
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: SpiceMCPClient | None = None
_client_lock = threading.Lock()


def get_spice_client() -> SpiceMCPClient:
    """Get or create the SPICE MCP client singleton.

    The client is lazily initialized on first call. The subprocess is kept
    alive for the lifetime of the agent process.
    """
    global _client
    if _client is not None and _client.is_connected():
        return _client

    with _client_lock:
        # Double-check inside lock
        if _client is not None and _client.is_connected():
            return _client

        # Clean up old client if it exists but is disconnected
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass

        get_event_bus().emit(DEBUG, agent="SpiceMCP", level="debug", msg="Starting SPICE MCP client subprocess...")
        _client = SpiceMCPClient()
        _client._start()
        get_event_bus().emit(DEBUG, agent="SpiceMCP", level="debug", msg="SPICE MCP client connected")
        return _client


def _cleanup_at_exit():
    """atexit handler to clean up the MCP client."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(_cleanup_at_exit)
