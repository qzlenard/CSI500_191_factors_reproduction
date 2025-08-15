"""
[CN] 取数与清洗契约：拉取一年日频 OHLCV，按股票 CSV 落地，必要字段兜底。
[Purpose] Collect & clean OHLCV; persist per-code CSV; build in-memory panel.

Interfaces:
- collect_and_store_ohlcv(codes: list[str], start: str, end: str, fq: str, out_dir: str) -> None
  Writes data/raw/ohlcv/{code}.csv (append/incremental).
- build_panel_from_csv(codes: list[str], start: str, end: str, raw_dir: str) -> pd.DataFrame
  Returns a panel-like long DataFrame indexed by date with 'code' column.

Cleaning Rules (Hard Constraints):
- Fill/derive paused, high_limit, low_limit if missing (use thresholds for non-ST ±9.8%, ST ±4.8%).
- Ensure monotonic dates; drop duplicates; clip inf/nan (utils/numerics.py).
- Log coverage and missing-field fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import time
import numpy as np
import pandas as pd

WARMUP_DAYS = 500  # 或者您可以在 config.py 中添加一个配置项，并从中读取


# ---------------- Project imports (package-first, script fallback) ----------------
try:
    if __package__:
        from .utils import logging as log
        from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic, append_with_rolloff
        from .utils.numerics import clip_inf_nan
        from .trading_calendar import get_trading_days
        from .api.myquant_io import get_ohlcv
        from . import universe
        # config aliases
        from config import (
            CFG,
            LOG_VERBOSITY,
            FQ,
            RAW_OHLCV_DIR, OUT_TS_DIR,
            COVERAGE_CSV,
            ROLLING_KEEP_DAYS,
        )
    else:
        raise ImportError
except Exception:
    # Script-mode fallback (tests / notebooks)
    from src.utils import logging as log
    from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic, append_with_rolloff
    from src.utils.numerics import clip_inf_nan
    from src.trading_calendar import get_trading_days
    from src.api.myquant_io import get_ohlcv
    from src import universe
    from config import (
        CFG,
        LOG_VERBOSITY,
        FQ,
        RAW_OHLCV_DIR, OUT_TS_DIR,
        COVERAGE_CSV,
        ROLLING_KEEP_DAYS,
    )

# ---------------- Config shorthands ----------------
BATCH_SIZE = int(getattr(CFG.fetch, "batch_size", 100))
MAX_RETRY = int(getattr(CFG.fetch, "max_retries", 3))
RETRY_BACKOFF = float(getattr(CFG.fetch, "retry_backoff_sec", 1.0))

RAW_DIR = RAW_OHLCV_DIR
TS_DIR = OUT_TS_DIR

# ---------------- Helpers ----------------
def _to_ts(x: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()

def _denorm_code(c: str) -> str:
    """
    Normalize code to '######.(SH|SZ)'.
    Accepts:
        'SHSE.600000' -> '600000.SH'
        'SZSE.000001' -> '000001.SZ'
        '600000.SH' (unchanged)
        '000001.SZ' (unchanged)
    """
    if not isinstance(c, str):
        c = str(c)
    u = c.strip().upper()
    if u.startswith("SHSE.") or u.startswith("SZSE."):
        ex, num = u.split(".", 1)
        suf = "SH" if ex.startswith("SH") else "SZ"
        return f"{num}.{suf}"
    if len(u) == 9 and u[:6].isdigit() and u[6:] in ("SH", "SZ"):
        return u
    if len(u) == 8 and u[:6].isdigit() and u[6:].isalpha():
        return f"{u[:6]}.{u[6:].upper()}"
    return u

def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out

def _retry_call(func, *args, **kwargs):
    last_err: Optional[Exception] = None
    for k in range(1, MAX_RETRY + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            log.warn(f"[PRE] fetch failed (attempt {k}/{MAX_RETRY}): {type(e).__name__}: {e}")
            time.sleep(RETRY_BACKOFF * (2 ** (k - 1)))
    if last_err:
        raise last_err
    raise RuntimeError("Unknown retry failure")

@dataclass
class CoverageRow:
    date: str
    codes_total: int
    codes_union_window: int  # New column for the union size
    days_total: int
    rows_raw: int
    rows_valid: int
    missing_paused_count: int
    missing_limit_count: int
    invalid_price_count: int
    invalid_volume_count: int
    files_written: int
    window_start: str
    window_end: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])

# ---------------- Core pipeline ----------------
def run_preprocess(start: pd.Timestamp | str,
                   end: pd.Timestamp | str,
                   warmup_days: Optional[int] = None) -> CoverageRow:
    """
    Main entry for Step-1 preprocessing.
    Returns a CoverageRow (also persisted with rolling keep_last).

    Parameters
    ----------
    start, end : str|Timestamp
        Desired main processing window. Coverage is keyed by end (or last trading day within).
    warmup_days : int|None
        Calendar days to back off before `start` to ensure indicators have runway.
        Defaults to CFG.fetch.calendar_buffer_days.
    """
    try:
        log.set_verbosity(LOG_VERBOSITY)
    except Exception:
        pass

    start = _to_ts(start)
    end = _to_ts(end)
    if warmup_days is None:
        warmup_days = int(getattr(CFG.fetch, "calendar_buffer_days", 10))

    # ---- calendar windows ----
    s_full = start - pd.Timedelta(days=int(warmup_days))
    days_full: List[pd.Timestamp] = get_trading_days(s_full, end)
    days_main: List[pd.Timestamp] = get_trading_days(start, end)
    if not days_full:
        raise RuntimeError("No trading days in the requested window (including warmup).")

    log.step(f"Preprocess OHLCV [{days_full[0].date()} → {days_full[-1].date()}] for CSI500 ...")

    # ---- build union universe across days_full ----
    sizes: List[int] = []
    union_codes: set[str] = set()
    t0 = time.time()

    # Use rolling union of all stocks for WARMUP_DAYS period
    for i, d in enumerate(days_full, 1):
        codes_t = universe.rolling_union_codes(start=d - pd.Timedelta(days=WARMUP_DAYS), end=d)  # Using rolling union
        union_codes.update(codes_t)
        sizes.append(len(codes_t))
        t0 = log.loop_progress("Build union universe (per-day)", i, len(days_full), start_time=t0, every=max(1, len(days_full)//10))

    avg_size = (sum(sizes) / len(sizes)) if sizes else float("nan")
    codes = sorted(union_codes)  # This holds the union of all stocks over the window
    log.done(f"[UNIV] average membership per day ≈ {avg_size:.1f}, union codes = {len(codes)}")

    # ---- batch fetch OHLCV (codes × full period) ----
    s_str = days_full[0].strftime("%Y-%m-%d")
    e_str = days_full[-1].strftime("%Y-%m-%d")
    frames: List[pd.DataFrame] = []

    log.step(f"Fetch OHLCV in batches (BATCH_SIZE={BATCH_SIZE}, fq='{FQ}')")
    N = len(codes)
    t1 = time.time()
    for i in range(0, N, BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        try:
            df_b = _retry_call(get_ohlcv, batch, s_str, e_str, FQ)
        except Exception as ex:
            log.warn(f"[PRE] Batch {i//BATCH_SIZE+1} failed and skipped: {type(ex).__name__}: {ex}")
            df_b = pd.DataFrame(columns=["date","code","open","high","low","close","volume"])
        frames.append(df_b)
        done_cnt = min(i + BATCH_SIZE, N)
        t1 = log.loop_progress("Fetch OHLCV", done_cnt, N, start_time=t1, every=max(1, BATCH_SIZE))

    panel_raw = (pd.concat(frames, axis=0, ignore_index=True) if frames else
                 pd.DataFrame(columns=["date","code","open","high","low","close","volume"]))
    rows_raw = int(len(panel_raw))

    # ---- normalize schema & light cleaning ----
    must = ["date","code","open","high","low","close","volume"]
    nice_to_have = ["amount","preclose","paused","high_limit","low_limit"]
    panel = panel_raw.copy()

    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    if "code" in panel.columns:
        panel["code"] = panel["code"].astype(str).map(_denorm_code)

    panel = _ensure_columns(panel, must + nice_to_have)

    panel = panel.drop_duplicates(subset=["code","date"]).sort_values(["code","date"], kind="mergesort")

    num_cols = ["open","high","low","close","volume","amount","preclose","high_limit","low_limit"]
    panel[num_cols] = clip_inf_nan(panel[num_cols])

    missing_paused = panel["paused"].isna()
    missing_limit = panel["high_limit"].isna() | panel["low_limit"].isna()
    missing_paused_count = int(missing_paused.sum())
    missing_limit_count = int(missing_limit.sum())

    invalid_price = pd.to_numeric(panel["close"], errors="coerce").fillna(np.nan) <= 0
    invalid_volume = pd.to_numeric(panel["volume"], errors="coerce").fillna(np.nan) < 0
    invalid_price_count = int(invalid_price.sum())
    invalid_volume_count = int(invalid_volume.sum())
    keep_mask = ~(invalid_price | invalid_volume)
    panel = panel.loc[keep_mask].copy()
    rows_valid = int(len(panel))

    log.done(f"[CLEAN] rows_raw={rows_raw}, rows_valid={rows_valid}, miss_paused={missing_paused_count}, miss_limit={missing_limit_count}, invalid_px={invalid_price_count}, invalid_vol={invalid_volume_count}")

    # ---- per-stock atomic write ----
    ensure_dir(RAW_DIR)
    uniq_codes = sorted(panel["code"].dropna().astype(str).unique().tolist())
    K = len(uniq_codes)

    t2 = time.time()
    files_written = 0
    out_cols = [c for c in (must + nice_to_have) if c in panel.columns]
    for idx, code in enumerate(uniq_codes, 1):
        df_c = panel.loc[panel["code"] == code, out_cols].sort_values("date").copy()

        fp = RAW_DIR / f"{code}.csv"
        old = read_csv_safe(fp, parse_dates=["date"])
        if not old.empty:
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()

        # --- FIX: only concat non-empty parts; if both empty, skip
        parts: List[pd.DataFrame] = []
        if not old.empty:
            parts.append(old[out_cols])
        if not df_c.empty:
            parts.append(df_c)

        if parts:
            merged = pd.concat(parts, axis=0, ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
            write_csv_atomic(fp, merged, index=False)
            files_written += 1
        # if both empty → skip writing & counting

        t2 = log.loop_progress("Write per-stock CSV", idx, K, start_time=t2, every=max(1, K//20 if K else 1))

    # ---- coverage row (rolling append) ----
    ensure_dir(TS_DIR)
    coverage_date = (days_main[-1] if days_main else days_full[-1]).strftime("%Y-%m-%d")
    cov = CoverageRow(
        date=coverage_date,
        codes_total=len(codes),
        codes_union_window=len(union_codes),  # New column for union size
        days_total=len(days_full),
        rows_raw=rows_raw,
        rows_valid=rows_valid,
        missing_paused_count=missing_paused_count,
        missing_limit_count=missing_limit_count,
        invalid_price_count=invalid_price_count,
        invalid_volume_count=invalid_volume_count,
        files_written=files_written,
        window_start=days_full[0].strftime("%Y-%m-%d"),
        window_end=days_full[-1].strftime("%Y-%m-%d"),
    )
    _ = append_with_rolloff(COVERAGE_CSV, cov.to_frame(), key="date", keep_last=ROLLING_KEEP_DAYS)

    log.done(f"Preprocess done: codes={len(codes)}, rows_raw={rows_raw}, rows_valid={rows_valid}, "
             f"miss_paused={missing_paused_count}, miss_limit={missing_limit_count}")

    return cov


# ---------------- Optional smoke test ----------------
if __name__ == "__main__":  # pragma: no cover
    log.set_verbosity(LOG_VERBOSITY)
    today = pd.Timestamp.today().normalize()
    days = get_trading_days(today - pd.Timedelta(days=40), today)
    if len(days) >= 20:
        s, e = days[-20], days[-1]
    else:
        s, e = (days[0], days[-1]) if days else (today - pd.Timedelta(days=20), today)
    cov = run_preprocess(s, e)
    log.done(f"[SMOKE] coverage appended for {cov.date}")
