"""
Time range types and parsing logic for the agent.

All datetimes are naive (no tzinfo) and implicitly UTC.
"""

import re
from datetime import datetime, timedelta, timezone


class TimeRangeError(Exception):
    """Raised when a time range string cannot be parsed.

    The message is user-facing and suggests valid formats.
    """


class TimeRange:
    """A UTC time range with naive (implicitly UTC) datetimes.

    Attributes:
        start: Naive start datetime, implicitly UTC (inclusive).
        end: Naive end datetime, implicitly UTC (exclusive).
    """

    def __init__(self, start: datetime, end: datetime):
        """Create a TimeRange.

        Args:
            start: Start datetime (naive, implicitly UTC).
            end: End datetime (naive, implicitly UTC).

        Raises:
            ValueError: If start >= end.
        """
        # Strip tzinfo if present (everything is implicitly UTC)
        if start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if end.tzinfo is not None:
            end = end.replace(tzinfo=None)
        if start >= end:
            raise ValueError(
                f"Start ({start.isoformat()}) must be before end ({end.isoformat()})"
            )
        self.start = start
        self.end = end

    def to_time_range_string(self) -> str:
        """Format as 'YYYY-MM-DD to YYYY-MM-DD' for CDAWeb queries.

        Omits the time component when both start and end are at midnight
        (day-precision). Includes T%H:%M:%S otherwise.
        """
        start_midnight = (self.start.hour == 0 and self.start.minute == 0
                          and self.start.second == 0)
        end_midnight = (self.end.hour == 0 and self.end.minute == 0
                        and self.end.second == 0)

        if start_midnight and end_midnight:
            return f"{self.start.strftime('%Y-%m-%d')} to {self.end.strftime('%Y-%m-%d')}"
        else:
            return f"{self.start.isoformat()} to {self.end.isoformat()}"

    def __repr__(self) -> str:
        return f"TimeRange({self.start.isoformat()}, {self.end.isoformat()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TimeRange):
            return NotImplemented
        return self.start == other.start and self.end == other.end


def _parse_single_datetime(s: str) -> datetime:
    """Try parsing a single datetime string in several formats.

    Tries: %Y-%m-%dT%H:%M:%S, %Y-%m-%dT%H:%M, %Y-%m-%d (in order).

    Returns:
        Naive datetime (implicitly UTC).

    Raises:
        ValueError: If none of the formats match.
    """
    s = s.strip().rstrip("Z")
    # Try fromisoformat first — handles fractional seconds and full ISO 8601
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: '{s}'")


# Month name -> number mapping
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_time_range(text: str) -> TimeRange:
    """Parse a time expression into a TimeRange.

    Supports:
      - Relative: "last week", "last 3 days", "last month", "last year"
      - Month+year: "January 2024", "Jan 2024"
      - Single date: "2024-01-15" (expands to full day)
      - Date range: "2024-01-15 to 2024-01-20"
      - Datetime range: "2024-01-15T06:00 to 2024-01-15T18:00"
      - Single datetime: "2024-01-15T06:00" (expands to 1-hour window)

    Args:
        text: Time expression string.

    Returns:
        TimeRange with naive (implicitly UTC) datetimes.

    Raises:
        TimeRangeError: If the input cannot be parsed. The message
            suggests valid formats so the agent can relay it to the user.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    text_lower = text.lower().strip()

    # --- Relative expressions ---

    # "last N days"
    if "last" in text_lower and "day" in text_lower:
        match = re.search(r"(\d+)\s*day", text_lower)
        if match:
            days = int(match.group(1))
            return TimeRange(today - timedelta(days=days), today)

    # "last week"
    if "last week" in text_lower:
        return TimeRange(today - timedelta(days=7), today)

    # "last month"
    if "last month" in text_lower:
        return TimeRange(today - timedelta(days=30), today)

    # "last year"
    if "last year" in text_lower:
        return TimeRange(today - timedelta(days=365), today)

    # --- Month + year (e.g., "January 2024", "Jan 2024") ---
    for month_name, month_num in _MONTHS.items():
        if month_name in text_lower:
            year_match = re.search(r"20\d{2}", text_lower)
            if year_match:
                year = int(year_match.group())
                start = datetime(year, month_num, 1)
                if month_num == 12:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, month_num + 1, 1)
                return TimeRange(start, end)

    # --- Range with "to" separator (date or datetime) ---
    if " to " in text:
        parts = text.split(" to ", 1)
        try:
            start_dt = _parse_single_datetime(parts[0])
            end_dt = _parse_single_datetime(parts[1])
        except ValueError as e:
            raise TimeRangeError(
                f"Could not parse time range '{text}': {e}. "
                "Try formats like '2024-01-15 to 2024-01-20' or "
                "'2024-01-15T06:00 to 2024-01-15T18:00'."
            ) from e
        return TimeRange(start_dt, end_dt)

    # --- Single datetime with time component (expand to 1-hour window) ---
    if "T" in text:
        try:
            dt = _parse_single_datetime(text.strip())
            return TimeRange(dt, dt + timedelta(hours=1))
        except ValueError:
            pass

    # --- Single date (expand to full day) ---
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text.strip()):
        try:
            dt = _parse_single_datetime(text.strip())
            return TimeRange(dt, dt + timedelta(days=1))
        except ValueError:
            pass

    # --- Nothing matched ---
    raise TimeRangeError(
        f"Could not parse time range '{text}'. Supported formats:\n"
        "  - Relative: 'last week', 'last 3 days', 'last month', 'last year'\n"
        "  - Month+year: 'January 2024', 'Jan 2024'\n"
        "  - Date: '2024-01-15' (single day)\n"
        "  - Date range: '2024-01-15 to 2024-01-20'\n"
        "  - Datetime range: '2024-01-15T06:00 to 2024-01-15T18:00'\n"
        "  - Single datetime: '2024-01-15T06:00' (1-hour window)"
    )
