"""SSE bridge: sync agent thread → async event stream."""

import asyncio
import threading
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


class SessionSSEBridge:
    """Persistent SSE bridge for a session's lifetime.

    Broadcasts events to all connected subscribers (multiple browser tabs,
    reconnects, etc.). Lives as long as the session's run_loop() is active.

    Thread-safe: callback() is called from the agent thread, while
    subscribe/unsubscribe are called from async request handlers.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._subscribers: list[asyncio.Queue[dict | None]] = []
        self._lock = threading.Lock()

    def callback(self, event: dict) -> None:
        """Thread-safe callback invoked by the agent from a worker thread."""
        with self._lock:
            for q in self._subscribers:
                self._loop.call_soon_threadsafe(q.put_nowait, event)

    def subscribe(self) -> asyncio.Queue[dict | None]:
        """Create a new subscriber queue. Returns a queue that yields events."""
        q: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def close(self) -> None:
        """Signal all subscribers to stop and clear the subscriber list."""
        with self._lock:
            for q in self._subscribers:
                self._loop.call_soon_threadsafe(q.put_nowait, None)
            self._subscribers.clear()
