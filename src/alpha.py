# Cross-module constraints: see docs/contract_notes.md
"""
[CN] Alpha 生成契约：过去一年（lookback）两日因子收益率的均值作为下期预测，对当期残差截面加权聚合为 Alpha。
[Purpose] Turn factor return history into next-period alpha cross-section.

Signature:
def next_alpha_from_trailing_mean(resid_f_today: pd.DataFrame, factor_returns_path: str,
                                  lookback_days: int, codes: list[str]) -> pd.Series:
    '''
    Inputs:
      - resid_f_today: index=code, cols=f1..f191 (today’s residualized exposures)
      - factor_returns_path: CSV "out/ts/factor_returns.csv"
      - lookback_days: e.g., LOOKBACK_FOR_ALPHA (default 252)
      - codes: target codes to output
    Returns:
      - pd.Series alpha indexed by code (float), AFTER aligning columns and cleaning.
    Logging:
      - [ALPHA] saved K names, coverage %, lookback used, any NaNs dropped.
    Persistence:
      - Write out/alpha/YYYYMMDD_alpha.csv
    '''
"""
from __future__ import annotations

# file: src/alpha.py
# -*- coding: utf-8 -*-
"""
Alpha construction (Step 4/6): trailing-mean projection.

Contract
--------
def next_alpha_from_trailing_mean(resid_f_today: pd.DataFrame,
                                  factor_returns_path: str,
                                  lookback_days: int,
                                  codes: list[str]) -> pd.Series

Design
------
- Read last `lookback_days` rows from factor_returns.csv.
- Compute per-factor trailing mean μ_j over f1..f191 (skip NaNs).
- For each stock i: alpha_i = sum_j resid[i, f_j] * μ_j  (skip NaN terms).
- If a stock has all skipped terms -> alpha_i = NaN and WARN.
- Persist CSV: out/alpha/{yyyymmdd}_alpha.csv with schema [code, alpha].
- Logging:
  [STEP 4/6] Alpha projection (lookback=...) ...
  [ALPHA] active_factors={m}, saved={n} names
"""

from typing import List
from pathlib import Path
import time  # <- hotfix: ensure time is imported
import numpy as np
import pandas as pd

# -------------------- Imports (package-first; script fallback) ----------------
try:
    if __package__:
        from .utils import logging as log  # type: ignore
        from .utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe  # type: ignore
        from .utils.numerics import clip_inf_nan  # type: ignore
        from .trading_calendar import most_recent_trading_day  # type: ignore
        from config import (  # type: ignore
            ALPHA_CSV_PATTERN, LOG_VERBOSITY
        )
    else:
        raise ImportError
except Exception:  # pragma: no cover
    from src.utils import logging as log  # type: ignore
    from src.utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe  # type: ignore
    from src.utils.numerics import clip_inf_nan  # type: ignore
    from src.trading_calendar import most_recent_trading_day  # type: ignore
    # Minimal config shim (defaults)
    ALPHA_CSV_PATTERN = "out/alpha/{yyyymmdd}_alpha.csv"
    LOG_VERBOSITY = "STEP"

# ------------------------------ Constants -------------------------------------
F_COLS: List[str] = [f"f{i}" for i in range(1, 192)]  # f1..f191


# ------------------------------ Utilities -------------------------------------
def _infer_date(resid: pd.DataFrame) -> pd.Timestamp:
    """Infer cross-section date from resid.attrs['date']; fallback to most recent trading day."""
    d = resid.attrs.get("date", None)
    if d is None:
        return most_recent_trading_day()
    try:
        return pd.Timestamp(d).normalize()
    except Exception:
        return most_recent_trading_day()


def _to_series_nan(codes: list[str]) -> pd.Series:
    idx = pd.Index(list(codes), name="code")
    return pd.Series(np.nan, index=idx, dtype=float)


# ------------------------------ Public API ------------------------------------
def next_alpha_from_trailing_mean(
    resid_f_today: pd.DataFrame,
    factor_returns_path: str,
    lookback_days: int,
    codes: list[str],
) -> pd.Series:
    """
    Build alpha cross-section by dotting today's residual exposures with
    trailing-mean factor returns.

    Parameters
    ----------
    resid_f_today : DataFrame
        Index=code, columns=f1..f191 (may contain NaN).
    factor_returns_path : str
        Path to out/ts/factor_returns.csv.
    lookback_days : int
        Trailing window length (e.g., 252).
    codes : list[str]
        Output universe & row order of the result.

    Returns
    -------
    pd.Series
        Alpha scores indexed by `codes`. NaN if insufficient info.
    """
    t0 = time.time()
    try:
        log.set_verbosity(LOG_VERBOSITY)  # best-effort; ignore if not supported
    except Exception:
        pass

    log.step(f"[STEP 5/6] Alpha projection (lookback={int(lookback_days)}) ...")

    # 1) Load factor returns (rolling CSV)
    fr = read_csv_safe(str(factor_returns_path), parse_dates=["date"], default=pd.DataFrame())
    if fr is None or fr.empty or "date" not in fr.columns:
        log.warn(f"[ALPHA] no factor_returns at '{factor_returns_path}' or empty content; return all-NaN")
        # Persist a CSV with NaNs to keep pipeline consistent
        date_ts = _infer_date(resid_f_today)
        out_path = ALPHA_CSV_PATTERN.format(yyyymmdd=date_ts.strftime("%Y%m%d"))
        ensure_dir(out_path, is_file=True)
        alpha_nan = _to_series_nan(codes)
        df_out = pd.DataFrame({"code": alpha_nan.index, "alpha": alpha_nan.values})
        write_csv_atomic(out_path, df_out, index=False)
        log.done("[ALPHA] active_factors=0, saved=0 names")
        return alpha_nan

    # Ensure factor columns present (align to F_COLS)
    present_f = [c for c in F_COLS if c in fr.columns]
    if not present_f:
        log.warn(f"[ALPHA] factor_returns has no f1..f191 columns; return all-NaN")
        date_ts = _infer_date(resid_f_today)
        out_path = ALPHA_CSV_PATTERN.format(yyyymmdd=date_ts.strftime("%Y%m%d"))
        ensure_dir(out_path, is_file=True)
        alpha_nan = _to_series_nan(codes)
        write_csv_atomic(out_path, pd.DataFrame({"code": alpha_nan.index, "alpha": alpha_nan.values}), index=False)
        log.done("[ALPHA] active_factors=0, saved=0 names")
        return alpha_nan

    # Keep only the last `lookback_days` rows by date (ascending then tail)
    fr = fr.sort_values("date").tail(int(lookback_days))

    # 2) Trailing mean μ_j for each factor (skip NaNs)
    # Column-wise numeric coercion to avoid TypeError on DataFrame.
    fr_num = fr[present_f].apply(pd.to_numeric, errors="coerce")
    mu = fr_num.mean(axis=0, skipna=True)
    active_factors = mu.dropna().index.tolist()
    m = len(active_factors)

    if m == 0:
        log.warn("[ALPHA] trailing means are all-NaN; return all-NaN")
        date_ts = _infer_date(resid_f_today)
        out_path = ALPHA_CSV_PATTERN.format(yyyymmdd=date_ts.strftime("%Y%m%d"))
        ensure_dir(out_path, is_file=True)
        alpha_nan = _to_series_nan(codes)
        write_csv_atomic(out_path, pd.DataFrame({"code": alpha_nan.index, "alpha": alpha_nan.values}), index=False)
        log.done("[ALPHA] active_factors=0, saved=0 names")
        return alpha_nan

    # 3) Compute alpha = resid_f_today · mu  (skip NaN terms; if all skipped -> NaN)
    R = resid_f_today.copy()
    # Keep only factor columns, align order, coerce numeric
    R = R.reindex(columns=active_factors)
    for c in R.columns:
        R[c] = pd.to_numeric(R[c], errors="coerce")

    # Align rows to provided codes order
    R = R.reindex(index=pd.Index(codes, name="code"))

    # Per-row dot product with min_count=1 to return NaN if all terms are NaN
    alpha = (R * mu.reindex(active_factors)).sum(axis=1, min_count=1)
    alpha = clip_inf_nan(alpha)

    # 4) Warn for rows with all-NaN contributions
    all_nan_mask = R[active_factors].isna().all(axis=1)
    if all_nan_mask.any():
        bad_codes = list(alpha.index[all_nan_mask])
        head = bad_codes[:10]
        tail = f"...(+{len(bad_codes) - 10})" if len(bad_codes) > 10 else ""
        log.warn(f"[ALPHA] all-NaN residual exposures for {len(bad_codes)} codes: {head} {tail}")

    # 5) Persist daily alpha CSV
    date_ts = _infer_date(resid_f_today)
    out_path = ALPHA_CSV_PATTERN.format(yyyymmdd=date_ts.strftime("%Y%m%d"))
    ensure_dir(out_path, is_file=True)
    df_out = pd.DataFrame({"code": alpha.index, "alpha": alpha.values})
    write_csv_atomic(out_path, df_out, index=False)

    n_saved = int(pd.notna(alpha).sum())
    log.step(f"[ALPHA] active_factors={m}, saved={n_saved} names")
    elapsed = time.time() - t0
    log.done(f"[STEP 5/6] Alpha projection . done  time={elapsed:.2f}s")

    return alpha


# ------------------------------ Smoke Test ------------------------------------
if __name__ == "__main__":  # pragma: no cover
    """
    Smoke test:
    - Build a toy resid_f_today (3 stocks × 191 factors, only 5 factors filled; others NaN).
    - Build a small factor_returns.csv with 5 columns over several days.
    Assertions:
      * Output index order == input `codes`.
      * Columns whose μ_j is NaN are ignored.
      * Stock with all-NaN exposures -> NaN alpha and WARN.
      * Output CSV exists and is non-empty.
    """
    try:
        log.set_verbosity(LOG_VERBOSITY)
    except Exception:
        pass

    rng = np.random.default_rng(404)
    codes = ["600001.SH", "000002.SZ", "300003.SZ"]
    idx = pd.Index(codes, name="code")

    # Residuals: 191 columns; only f1..f5 filled with random, make 3rd name all NaN
    R = pd.DataFrame(np.nan, index=idx, columns=F_COLS, dtype=float)
    R.loc[:, ["f1", "f2", "f3", "f4", "f5"]] = rng.normal(size=(3, 5))
    R.loc["300003.SZ", ["f1", "f2", "f3", "f4", "f5"]] = np.nan  # force all-NaN row
    # Attach deterministic date for output path
    R.attrs["date"] = pd.Timestamp("2024-03-05")

    # Factor returns CSV (simulate 10 days; make f3 all-NaN so μ_f3 = NaN)
    dates = pd.bdate_range("2024-02-20", periods=10)
    FR = pd.DataFrame({
        "date": dates,
        "f1": rng.normal(size=10),
        "f2": rng.normal(size=10),
        "f3": [np.nan] * 10,  # will be ignored
        "f4": rng.normal(size=10),
        "f5": rng.normal(size=10),
    })
    # Place under canonical path
    fr_path = "out/ts/factor_returns.csv"
    ensure_dir(fr_path, is_file=True)
    write_csv_atomic(fr_path, FR, index=False)

    # Run
    alpha = next_alpha_from_trailing_mean(R, fr_path, lookback_days=5, codes=codes)

    # Checks
    assert list(alpha.index) == codes, "Index order mismatch"
    mu = FR.tail(5)[["f1", "f2", "f3", "f4", "f5"]].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True)
    assert pd.isna(mu["f3"]), "Expected μ_f3 to be NaN"

    out_path = ALPHA_CSV_PATTERN.format(yyyymmdd=pd.Timestamp("2024-03-05").strftime("%Y%m%d"))
    assert Path(out_path).exists() and Path(out_path).stat().st_size > 0, "Alpha CSV not written"

    assert pd.isna(alpha.loc["300003.SZ"]), "Row with all-NaN exposures should yield NaN"

    print("[SMOKE][ALPHA] OK")
