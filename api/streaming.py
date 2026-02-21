"""SSE bridge: sync agent thread → async event stream."""

import asyncio
from typing import AsyncIterator


class SSEBridge:
    """Bridge between synchronous agent callbacks and async SSE event stream.

    Usage:
        bridge = SSEBridge(loop)
        agent.subscribe_sse(bridge.callback)
        # In async endpoint:
        async for event in bridge.events():
            yield event
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def callback(self, event: dict) -> None:
        """Thread-safe callback invoked by the agent from a worker thread."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def finish(self, done_event: dict) -> None:
        """Signal the stream is complete by pushing the done event + sentinel."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, done_event)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def error(self, message: str) -> None:
        """Push an error event and terminate the stream."""
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, {"type": "error", "message": message}
        )
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    async def events(self) -> AsyncIterator[dict]:
        """Async generator yielding events until the stream ends."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
