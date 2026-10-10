"""Time utilities for candle alignment and cycle calculations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

_TIMEFRAME_HOURS: dict[str, int] = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "6h": 6,
    "8h": 8,
    "12h": 12,
    "1d": 24,
}


def timeframe_to_hours(timeframe: str) -> int:
    """Convert a timeframe string like '4h' to integer hours.

    Raises ``ValueError`` for unsupported timeframes.
    """
    if timeframe not in _TIMEFRAME_HOURS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Supported: {sorted(_TIMEFRAME_HOURS.keys())}"
        )
    return _TIMEFRAME_HOURS[timeframe]


def expected_closed_candle_time(
    cycle_time: datetime,
    timeframe: str = "4h",
) -> datetime:
    """Return the open-timestamp of the latest fully-closed candle.

    For a 4h timeframe and *cycle_time* = 04:05 UTC the latest closed
    candle is the one that opened at 00:00 UTC (closed at 04:00 UTC).
    For a 15m timeframe and *cycle_time* = 14:03 UTC the latest closed
    candle is the one that opened at 13:45 UTC (closed at 14:00 UTC).

    The returned datetime is always in UTC (timezone-aware).
    """
    utc = (
        cycle_time.replace(tzinfo=UTC) if cycle_time.tzinfo is None else cycle_time.astimezone(UTC)
    )

    if timeframe.endswith("m"):
        mins = int(timeframe[:-1])
        total_mins = utc.hour * 60 + utc.minute
        floored_mins = (total_mins // mins) * mins
        current_candle_start = utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=floored_mins)
        return current_candle_start - timedelta(minutes=mins)

    hours = timeframe_to_hours(timeframe)

    # Floor to the timeframe boundary
    floored_hour = (utc.hour // hours) * hours

    current_candle_start = utc.replace(hour=floored_hour, minute=0, second=0, microsecond=0)
    latest_closed = current_candle_start - timedelta(hours=hours)
    return latest_closed


def get_next_candle_boundary(dt: datetime, timeframe: str = "4h") -> datetime:
    """Calculate the exact next boundary timestamp when a candle of `timeframe` will close.

    For 15m at 14:03 UTC, returns 14:15:00 UTC.
    For 4h at 05:20 UTC, returns 08:00:00 UTC.
    """
    utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

    if timeframe.endswith("m"):
        mins = int(timeframe[:-1])
        total_mins = utc.hour * 60 + utc.minute
        next_boundary_mins = ((total_mins // mins) + 1) * mins
        return utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=next_boundary_mins)

    hours = timeframe_to_hours(timeframe)
    current_hour = utc.hour
    next_boundary_hour = ((current_hour // hours) + 1) * hours
    if next_boundary_hour >= 24:
        return (utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return utc.replace(hour=next_boundary_hour, minute=0, second=0, microsecond=0)


def start_of_utc_day(dt: datetime) -> datetime:
    """Return midnight UTC of the given datetime's date."""
    utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return utc.replace(hour=0, minute=0, second=0, microsecond=0)


def start_of_utc_week(dt: datetime) -> datetime:
    """Return Monday 00:00 UTC of the ISO week containing *dt*."""
    utc = start_of_utc_day(dt)
    return utc - timedelta(days=utc.weekday())


# Alias for backwards/alternate naming compatibility
get_latest_closed_candle_time = expected_closed_candle_time
