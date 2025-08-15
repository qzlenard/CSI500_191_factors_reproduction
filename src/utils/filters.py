"""
[CN] 可交易性过滤契约：剔除停牌与涨/跌停。
[Purpose] Tradability filters for daily cross-sections.

Interfaces:
- tradable_codes(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str]) -> list[str]
  * Exclude paused==1
  * Exclude close >= high_limit or close <= low_limit
  * If limits missing, use thresholds (non-ST ±9.8%, ST ±4.8%) based on code or flags.

Notes:
- Must log counts: kept vs excluded, and reasons.
"""
from __future__ import annotations

# file: src/utils/filters.py
# -*- coding: utf-8 -*-
"""
Filtering utilities for tradability (suspension & price limits).

Contract:
- Input panel is a "long" DataFrame that at least contains columns:
  ['date','code','close','volume'] and *preferably* ['preclose','paused','high_limit','low_limit'].
  Optional columns: ['name','is_st'].
- Main API:
    tradable_codes(panel, date, codes) -> list[str]
  Helper APIs:
    trade_mask_with_reasons(panel, date, codes, ...) -> (pd.Series[bool], pd.DataFrame)
    filter_panel_by_tradability(panel, date, codes, ...) -> pd.DataFrame
- Rules (per project Master Prompt):
  * Exclude paused == 1 (or True). Fallback: volume == 0 implies suspension.
  * Prefer high_limit/low_limit to detect limit-up/limit-down hits by close.
  * If limit fields are missing, approximate using thresholds:
      non-ST: ±9.8%, ST: ±4.8%.
    ST detection: 'is_st'==True OR 'name' contains 'ST' / '*ST' (case-insensitive).
  * Clean, robust behavior under partial missing fields; never raise on missing columns.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd

# Use project logger if available; otherwise, create no-op shims.
try:
    from .logging import debug, warn
except Exception:  # pragma: no cover
    def debug(msg: str) -> None:  # type: ignore
        pass

    def warn(msg: str) -> None:  # type: ignore
        pass


@dataclass(frozen=True)
class LimitParams:
    normal_thresh: float = 0.098   # ±9.8% for non-ST
    st_thresh: float = 0.048       # ±4.8% for ST
    tick: float = 0.01             # A-share stock price tick (yuan)
    tol: float = 1e-4              # tolerance when comparing to limit prices


def _ensure_datetime(d) -> pd.Timestamp:
    """Coerce to pandas Timestamp (tz-naive)."""
    if isinstance(d, pd.Timestamp):
        return d.normalize()
    return pd.Timestamp(d).normalize()


def _slice_day_codes(panel: pd.DataFrame, date: pd.Timestamp, codes: Iterable[str]) -> pd.DataFrame:
    """Return a copy of rows for given date and codes, normalized column names (long format)."""
    if "date" not in panel.columns or "code" not in panel.columns:
        raise ValueError("panel must contain 'date' and 'code' columns (long format).")

    df = panel
    # Normalize date column to normalized Timestamps for comparison
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df_date = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        df_date = pd.to_datetime(df["date"]).dt.normalize()

    codes_set = set(codes)
    day_df = df.loc[(df_date == date) & (df["code"].isin(codes_set))].copy()
    day_df.sort_values(["code"], inplace=True, kind="mergesort")
    return day_df


def _is_st_like(name: Optional[str]) -> bool:
    """Heuristic ST detection from name text."""
    if not isinstance(name, str):
        return False
    s = name.upper()
    return ("ST" in s) or ("*ST" in s) or ("ＳＴ" in s)  # include full-width just in case


def _detect_st_flags(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series whether each row is ST."""
    if "is_st" in df.columns:
        col = df["is_st"]
        if np.issubdtype(col.dtype, np.bool_):
            return col.fillna(False)
        # Accept {1, '1', 'True', ...}
        return pd.to_numeric(col, errors="coerce").fillna(0).astype(int).astype(bool)
    if "name" in df.columns:
        return df["name"].map(_is_st_like).fillna(False)
    return pd.Series(False, index=df.index)


def _approx_limit_prices(df: pd.DataFrame, params: LimitParams) -> Tuple[pd.Series, pd.Series]:
    """
    Compute high/low limit prices per row, preferring existing columns,
    else approximate via preclose × (1 ± threshold(ST-aware)).
    """
    hi = pd.Series(np.nan, index=df.index, dtype=float)
    lo = pd.Series(np.nan, index=df.index, dtype=float)

    if "high_limit" in df.columns:
        hi = pd.to_numeric(df["high_limit"], errors="coerce")
    if "low_limit" in df.columns:
        lo = pd.to_numeric(df["low_limit"], errors="coerce")

    need_hi = hi.isna()
    need_lo = lo.isna()

    if need_hi.any() or need_lo.any():
        preclose = pd.to_numeric(df.get("preclose", np.nan), errors="coerce")
        st = _detect_st_flags(df)
        up_ratio = np.where(st, 1.0 + params.st_thresh, 1.0 + params.normal_thresh)
        dn_ratio = np.where(st, 1.0 - params.st_thresh, 1.0 - params.normal_thresh)

        approx_hi = preclose.values * up_ratio
        approx_lo = preclose.values * dn_ratio

        if params.tick and params.tick > 0:
            approx_hi = np.round(approx_hi / params.tick) * params.tick
            approx_lo = np.round(approx_lo / params.tick) * params.tick

        hi = hi.where(~need_hi, approx_hi)
        lo = lo.where(~need_lo, approx_lo)

    return hi, lo


def _is_paused(df: pd.DataFrame) -> pd.Series:
    """Paused if 'paused'==1 (preferred). Fallback: volume==0 or NaN close/open."""
    if "paused" in df.columns:
        paused = pd.to_numeric(df["paused"], errors="coerce").fillna(0).astype(int) == 1
    else:
        paused = pd.Series(False, index=df.index)

    if "volume" in df.columns:
        vol_zero = pd.to_numeric(df["volume"], errors="coerce").fillna(0) == 0
    else:
        vol_zero = pd.Series(False, index=df.index)

    nan_px = pd.Series(False, index=df.index)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            nan_px = nan_px | pd.to_numeric(df[col], errors="coerce").isna()
    return paused | vol_zero | nan_px


def _limit_hits_by_close(df: pd.DataFrame, params: LimitParams) -> Tuple[pd.Series, pd.Series]:
    """
    Detect whether 'close' is at/near high/low limit using tolerance.
    Returns (hit_up, hit_down).
    """
    if "close" not in df.columns:
        idx = df.index if "code" not in df.columns else df["code"].index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)

    close = pd.to_numeric(df["close"], errors="coerce")
    hi, lo = _approx_limit_prices(df, params)

    hit_up = close.ge(hi - params.tol) & hi.notna() & close.notna()
    hit_dn = close.le(lo + params.tol) & lo.notna() & close.notna()
    return hit_up, hit_dn


def trade_mask_with_reasons(
    panel: pd.DataFrame,
    date: pd.Timestamp | str,
    codes: Iterable[str],
    *,
    params: LimitParams = LimitParams(),
    return_reasons: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build a boolean mask for tradability with reasons (columns):
      - missing: missing essential OHLCV fields
      - paused: suspension (paused==1) or volume==0 or NaN prices
      - up_limit: close at/near upper limit
      - down_limit: close at/near lower limit
      - keep: valid (= not any reason)
    Index of returned Series/DataFrame is code.

    The input 'panel' may include extra codes; only 'codes' are considered.
    """
    day = _ensure_datetime(date)
    day_df = _slice_day_codes(panel, day, codes)

    # Reindex to include all requested codes (missing rows will appear for reporting)
    idx_codes = pd.Index(sorted(set(codes)), name="code")
    day_df = day_df.set_index("code").reindex(idx_codes)

    # Essential fields presence
    essential_cols = ["close", "volume"]
    missing_essentials = pd.Series(False, index=idx_codes)
    for col in essential_cols:
        if col not in day_df.columns:
            missing_essentials |= True
        else:
            missing_essentials |= pd.to_numeric(day_df[col], errors="coerce").isna()

    paused = _is_paused(day_df.reset_index()).values  # align to order
    up_hit, dn_hit = _limit_hits_by_close(day_df.reset_index(), params)
    up_hit = up_hit.values
    dn_hit = dn_hit.values

    reasons = pd.DataFrame(
        {
            "missing": missing_essentials.values,
            "paused": paused,
            "up_limit": up_hit,
            "down_limit": dn_hit,
        },
        index=idx_codes,
    )
    keep = ~(reasons.any(axis=1))

    # Logging summary — avoid f-string literal braces; print a dict instead
    try:
        total = len(idx_codes)
        kept = int(keep.sum())
        pct = 0.0 if total == 0 else kept / total * 100
        counts = reasons.sum(axis=0).astype(int).to_dict()
        summary = {
            "missing": int(counts.get("missing", 0)),
            "paused": int(counts.get("paused", 0)),
            "up_limit": int(counts.get("up_limit", 0)),
            "down_limit": int(counts.get("down_limit", 0)),
        }
        debug(f"[FILTER] date={day.date()} keep={kept}/{total} ({pct:.1f}%) reasons={summary}")
    except Exception as e:  # pragma: no cover
        warn(f"[FILTER] summary logging failed: {e}")

    if return_reasons:
        reasons["keep"] = keep
        return keep, reasons
    else:
        return keep, pd.DataFrame(index=idx_codes)


def tradable_codes(
    panel: pd.DataFrame,
    date: pd.Timestamp | str,
    codes: Iterable[str],
    *,
    params: LimitParams = LimitParams(),
) -> list[str]:
    """
    Public API: return sorted list of tradable codes on 'date' given 'codes'.
    This applies:
      - suspension filter (paused==1 or volume==0 / NaN px)
      - limit-up / limit-down close touch filter
      - robust handling of missing limit fields via approximation
    """
    mask, _ = trade_mask_with_reasons(panel, date, codes, params=params, return_reasons=True)
    # Ensure precise type: list[str]
    return mask.index[mask].astype(str).tolist()


def filter_panel_by_tradability(
    panel: pd.DataFrame,
    date: pd.Timestamp | str,
    codes: Iterable[str],
    *,
    params: LimitParams = LimitParams(),
    keep_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Convenience: return a DataFrame of rows (for 'date' & 'codes') that are tradable.
    keep_columns: optionally restrict output columns for efficiency.
    """
    day = _ensure_datetime(date)
    day_df = _slice_day_codes(panel, day, codes)
    mask, _ = trade_mask_with_reasons(day_df, day, day_df["code"].tolist(), params=params, return_reasons=True)
    out = day_df.set_index("code").loc[mask[mask].index]
    if keep_columns is not None:
        cols = [c for c in keep_columns if c in out.columns]
        out = out[cols]
    return out.reset_index()


# ---- Self-test (lightweight, can be removed/ignored in production) ----
if __name__ == "__main__":  # pragma: no cover
    # Minimal sanity check
    data = {
        "date": ["2025-08-13"] * 5,
        "code": ["A", "B", "C", "D", "E"],
        "close": [11.0, 10.98, 9.52, 8.0, np.nan],
        "preclose": [10.0, 10.0, 10.0, 8.0, 9.0],
        "volume": [1000, 0, 5000, 100, 100],
        "paused": [0, 0, 0, 1, 0],
        # Missing limit columns for A/B/C/E; D provided:
        "high_limit": [np.nan, np.nan, np.nan, 8.32, np.nan],  # 8*(1+4%)
        "low_limit": [np.nan, np.nan, np.nan, 7.68, np.nan],
        "name": ["Foo", "Bar", "*ST Baz", "Qux", "Quux"],
    }
    panel = pd.DataFrame(data)
    keep_list = tradable_codes(panel, "2025-08-13", ["A", "B", "C", "D", "E"])
    print("Tradable:", keep_list)
    # Expected:
    # A: approx +9.8% rule → treated as at limit => excluded
    # B: volume==0 => excluded
    # C: ST limit-down at 9.52 => excluded
    # D: paused==1 => excluded
    # E: close NaN => excluded
    # Result: []
