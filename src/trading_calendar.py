"""
[CN] 交易日历契约：判断/推算交易日、窗口对齐。
[Purpose] Calendar utilities.

Interfaces:
- is_trade_day(date: pd.Timestamp) -> bool
- next_trade_day(date: pd.Timestamp, n: int=1) -> pd.Timestamp
- prev_trade_day(date: pd.Timestamp, n: int=1) -> pd.Timestamp
- trade_days_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]

Assumptions:
- Uses get_trade_days under the hood; behavior consistent with exchange calendar.
"""
from __future__ import annotations

# file: src/trading_calendar.py
# -*- coding: utf-8 -*-
"""
Trading calendar utilities with local CSV cache and MyQuant integration.

Public APIs:
    - get_trading_days(start, end) -> list[pd.Timestamp]
    - is_trading_day(date) -> bool
    - prev_trading_day(date, n=1) -> pd.Timestamp
    - next_trading_day(date, n=1) -> pd.Timestamp
    - shift_trading_day(date, n) -> pd.Timestamp
    - last_n_trading_days(end, n) -> list[pd.Timestamp]
    - most_recent_trading_day(end=None) -> pd.Timestamp

Design notes:
    * Primary source: src/api/myquant_io.get_trade_days
    * CSV cache: data/ref/trade_calendar.csv with a single 'date' column
    * File lock (name-based): src/utils/state.with_file_lock(lock_name, timeout_sec=...)
    * Fallback: Monday–Friday (no holiday knowledge) with WARN logs

All code and comments in English per project convention.
"""


from pathlib import Path
from typing import List, Optional
import bisect
import datetime as dt

import pandas as pd

# ---------------------------------------------------------------------
# Import strategy (package first, script fallback)
# ---------------------------------------------------------------------
try:
    if __package__:
        try:
            from .api.myquant_io import get_trade_days as mq_get_trade_days  # type: ignore
        except Exception:
            mq_get_trade_days = None  # fallback later
        from .utils.fileio import ensure_dir, write_csv_atomic
        from .utils.state import with_file_lock
        from .utils import logging as log
    else:
        raise ImportError("Not in package mode")
except Exception:
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    try:
        from src.api.myquant_io import get_trade_days as mq_get_trade_days  # type: ignore
    except Exception:  # pragma: no cover
        mq_get_trade_days = None
    from src.utils.fileio import ensure_dir, write_csv_atomic
    from src.utils.state import with_file_lock
    from src.utils import logging as log

# ---- Constants ----
CACHE_PATH = Path("data/ref/trade_calendar.csv")
# with_file_lock in our codebase expects a *name* (string), not a path.
LOCK_NAME = "trade_calendar.lock"


# ---- Helpers ----
def _to_ts_date(x: pd.Timestamp | dt.date | dt.datetime | str) -> pd.Timestamp:
    """Convert to pd.Timestamp (date-normalized, tz-naive)."""
    if isinstance(x, pd.Timestamp):
        ts = x
    else:
        ts = pd.Timestamp(x)
    return pd.Timestamp(ts.date())


def _read_cache() -> pd.DatetimeIndex:
    """
    Load cached calendar as DatetimeIndex; empty if file missing.
    Use a dedicated name-based lock to avoid passing file paths into with_file_lock.
    """
    ensure_dir(CACHE_PATH.parent)
    with with_file_lock(LOCK_NAME, timeout_sec=5.0):
        if not CACHE_PATH.exists():
            return pd.DatetimeIndex([])
        try:
            df = pd.read_csv(CACHE_PATH)
        except Exception:
            # Corrupt cache -> treat as empty; next write will heal it
            return pd.DatetimeIndex([])
    if not isinstance(df, pd.DataFrame) or "date" not in df.columns:
        return pd.DatetimeIndex([])
    dates = pd.to_datetime(df["date"].astype(str), errors="coerce").dropna()
    dates = dates.map(lambda d: pd.Timestamp(pd.Timestamp(d).date()))
    return pd.DatetimeIndex(sorted(dates.unique()))


def _write_cache(didx: pd.DatetimeIndex) -> None:
    """Persist calendar cache atomically under a file lock."""
    ensure_dir(CACHE_PATH.parent)
    df = pd.DataFrame({"date": didx.astype("datetime64[ns]")})
    # Align to utils.state.with_file_lock(lock_name: str, timeout_sec: ...)
    with with_file_lock(LOCK_NAME, timeout_sec=10.0):
        write_csv_atomic(CACHE_PATH, df, index=False)


def _fetch_from_myquant(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """Fetch trading days from MyQuant; raise if not available."""
    if mq_get_trade_days is None:
        raise RuntimeError("MyQuant get_trade_days is unavailable")
    days = mq_get_trade_days(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    out = [pd.Timestamp(pd.Timestamp(d).date()) for d in days]
    return sorted(out)


def _fallback_weekdays(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """Fallback: Monday–Friday date range (no holidays)."""
    rng = pd.bdate_range(start=start, end=end, freq="C")
    return [pd.Timestamp(d.date()) for d in rng]


def _sync_cache_covering(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Ensure cache covers [start, end]. If cache misses the boundary span, fetch and merge.
    """
    start, end = _to_ts_date(start), _to_ts_date(end)
    cached = _read_cache()

    # Boundary-only coverage check to avoid weekend false positives.
    need_fetch = (
        len(cached) == 0
        or start < cached.min()
        or end > cached.max()
    )

    if not need_fetch:
        return cached

    log.step(f"[CAL] Sync trading calendar for {start.date()} → {end.date()} ...")
    try:
        fetched = _fetch_from_myquant(start, end)
        log.done(f"[CAL] MyQuant fetched {len(fetched)} days.")
    except Exception as e:
        log.warn(f"[CAL] MyQuant unavailable ({type(e).__name__}: {e}). Falling back to Mon–Fri weekdays (no holidays).")
        fetched = _fallback_weekdays(start, end)
        log.done(f"[CAL] Fallback generated {len(fetched)} weekdays.")

    merged = pd.DatetimeIndex(sorted(set(cached.tolist()) | set(fetched)))
    _write_cache(merged)
    log.done(f"[CAL] Cache updated. Total cached days: {len(merged)}")
    return merged


# ---- Public APIs ----
def get_trading_days(start: pd.Timestamp | str, end: pd.Timestamp | str) -> List[pd.Timestamp]:
    """Return trading days in [start, end] inclusive as a sorted list of pd.Timestamp (date-only)."""
    s, e = _to_ts_date(start), _to_ts_date(end)
    cal = _sync_cache_covering(s, e)
    mask = (cal >= s) & (cal <= e)
    return [pd.Timestamp(d.date()) for d in cal[mask]]


def is_trading_day(date: pd.Timestamp | str) -> bool:
    """True if `date` is a trading day (consults cache, syncing if necessary)."""
    d = _to_ts_date(date)
    cal = _sync_cache_covering(d, d)
    return d in set(cal)


def _locate_index(cal: pd.DatetimeIndex, d: pd.Timestamp) -> int:
    """Locate insertion point of d in calendar."""
    li = [ts.value for ts in cal]
    return bisect.bisect_left(li, d.value)


def prev_trading_day(date: pd.Timestamp | str, n: int = 1) -> pd.Timestamp:
    """The n-th previous trading day <= date (if date is trading day, count from it)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    d = _to_ts_date(date)
    cal = _sync_cache_covering(d - pd.Timedelta(days=365 * 5), d)
    i = _locate_index(cal, d)
    if i >= len(cal) or cal[i] != d:
        i -= 1
    target = i - (n - 1)
    if target < 0:
        raise IndexError("Calendar underflow while seeking previous trading day")
    return pd.Timestamp(cal[target].date())


def next_trading_day(date: pd.Timestamp | str, n: int = 1) -> pd.Timestamp:
    """The n-th next trading day >= date (if date is trading day, count from it)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    d = _to_ts_date(date)
    cal = _sync_cache_covering(d, d + pd.Timedelta(days=365 * 5))
    i = _locate_index(cal, d)
    if i >= len(cal):
        raise IndexError("Calendar overflow while seeking next trading day")
    target = i + (n - 1)
    if target >= len(cal):
        raise IndexError("Calendar overflow while seeking next trading day")
    return pd.Timestamp(cal[target].date())


def shift_trading_day(date: pd.Timestamp | str, n: int) -> pd.Timestamp:
    """Shift date by n trading days (positive = forward, negative = backward)."""
    if n == 0:
        return next_trading_day(date, 1)
    if n > 0:
        return next_trading_day(date, n)
    return prev_trading_day(date, -n)


def last_n_trading_days(end: pd.Timestamp | str, n: int) -> List[pd.Timestamp]:
    """Return the last n trading days ending at `end` (if `end` not a trading day, snap to previous)."""
    if n < 1:
        return []
    e = _to_ts_date(end)
    if not is_trading_day(e):
        e = prev_trading_day(e, 1)
    s_guess = e - pd.Timedelta(days=int(n * 2))
    cal = _sync_cache_covering(s_guess, e)
    days = cal[cal <= e]
    if len(days) < n:
        cal = _sync_cache_covering(e - pd.Timedelta(days=int(n * 5)), e)
        days = cal[cal <= e]
    if len(days) < n:
        raise RuntimeError(f"Insufficient calendar depth: need {n}, got {len(days)}")
    return [pd.Timestamp(x.date()) for x in days[-n:]]


def most_recent_trading_day(end: Optional[pd.Timestamp | str] = None) -> pd.Timestamp:
    """Return the most recent trading day ≤ `end` (default: today in local date)."""
    e = _to_ts_date(end or pd.Timestamp.today())
    if is_trading_day(e):
        return e
    return prev_trading_day(e, 1)


# ---- Optional: lightweight smoke when run directly ----
if __name__ == "__main__":
    log.set_verbosity("STEP")
    today = pd.Timestamp.today().normalize()
    s = today - pd.Timedelta(days=30)
    log.step("[SMOKE] Fetch last 30 calendar days ...")
    days = get_trading_days(s, today)
    log.done(f"[SMOKE] Got {len(days)} trading days. head={days[:3]}, tail={days[-3:] if days else []}")
    mr = most_recent_trading_day(today)
    log.done(f"[SMOKE] Most recent trading day: {mr.date()}")
    prev2 = prev_trading_day(mr, 2)
    next2 = next_trading_day(prev2, 2)
    log.done(f"[SMOKE] prev2={prev2.date()}, next2(back to)={next2.date()}")
