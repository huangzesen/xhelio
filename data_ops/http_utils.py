"""HTTP utilities — shared request helpers for the data pipeline."""

import time as _time

import requests

from agent.event_bus import get_event_bus, DEBUG

DEFAULT_TIMEOUT = 5  # seconds per request
DEFAULT_RETRIES = 3


def request_with_retry(url, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES,
                       **kwargs):
    """GET request with retry on timeout/connection errors."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s backoff
                get_event_bus().emit(DEBUG, agent="data_ops", msg=f"[HTTP] Retry {attempt}/{retries} for {url} (wait {wait}s): {e}")
                _time.sleep(wait)
    raise last_exc
