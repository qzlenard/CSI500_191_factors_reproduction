# Cross-module constraints: see docs/contract_notes.md
"""
[CN] 横截面回归契约：两日收益 ~ 行业 + 风格 + 残差因子 → 因子收益率（逐日）。
[Purpose] Estimate daily factor returns (betas) from cross-sectional regression.

Signatures:
def forward_return(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str], fwd_days: int=2) -> pd.Series:
    '''
    [CN] 对齐当日 t 暴露与未来 t→t+fwd_days 的简单收益：P_{t+f}/P_t - 1。
    Returns pd.Series indexed by code with forward simple returns.
    '''

def cs_factor_returns(y: pd.Series, styles: pd.DataFrame, inds: pd.DataFrame, resid_f: pd.DataFrame,
                      use_ridge: bool=True, ridge_alpha: float=1e-6) -> pd.Series:
    '''
    [CN] 逐日横截面：被解释变量 y=两日收益；解释变量 X=[inds, styles, resid_f]。
    Returns pd.Series of factor returns for columns resid_f (f1...f191) only.
    Regression:
      - Ridge with alpha=ridge_alpha (≈ OLS stabilization).
      - Skip if N < (p + 5); WARN and record metrics with NaN for that day.
    Logging:
      - Record R², N, effective columns used; write to out/ts/metrics.csv and out/ts/coverage.csv.
    Persistence:
      - Append factor returns row to out/ts/factor_returns.csv with rolling keep_last.
    '''

Hard Constraints:
- Cleaning: winsorize + z-score applied to X blocks (styles & residual factors) prior to fit.
- Sample: filter tradable universe (no paused, no limit-up/down).
"""
from __future__ import annotations
from __future__ import annotations

# file: src/regression.py
# -*- coding: utf-8 -*-
"""
Cross-sectional factor return estimation (Step 3/6).

Public API
----------
- forward_return(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str], fwd_days: int=2) -> pd.Series
    Compute r_{t→t+fwd_days} = P_{t+f}/P_t - 1 on close prices, with trading-day roll-forward.
- cs_factor_returns(y: pd.Series, styles: pd.DataFrame, inds: pd.DataFrame, resid_f: pd.DataFrame,
                    use_ridge: bool=True, ridge_alpha: float=1e-6) -> pd.Series
    Cross-sectional regression: y ~ styles + inds + resid_f, returning betas for f1..f191 only.

Design notes
------------
* Alignment: exposures at t explain forward returns over [t, t+f] with trading calendar roll-forward.
* Filtering: regression uses effective sample where y is finite and style exposures are available.
* Cleaning: safe double-guard — winsorize(1%) + zscore on styles and residual factors; clip inf/NaN.
* Regression: Ridge(alpha) with fit_intercept=True by default (do not penalize intercept).
* Storage: append-with-rolloff to out/ts/factor_returns.csv, metrics.csv, coverage.csv.
* Logging: uses utils.logging with STEP/LOOP/REG/DONE messages.

All code and comments are in English per project convention.
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import time

import numpy as np
import pandas as pd

# -------------------- Imports (package-first; script fallback) ----------------
try:
    if __package__:
        from .utils import logging as log  # type: ignore
        from .utils.fileio import ensure_dir, append_with_rolloff  # type: ignore
        from .utils.numerics import (
            winsorize, zscore, clip_inf_nan, ridge_fit, safe_lstsq, add_constant
        )  # type: ignore
        from .trading_calendar import shift_trading_day  # type: ignore
        from config import (  # type: ignore
            FACTOR_RETURNS_CSV, METRICS_CSV, COVERAGE_CSV,
            ROLLING_KEEP_DAYS, WINSOR_PCT, RIDGE_ALPHA, LOG_VERBOSITY
        )
        from .utils.state import DEFAULT_LOCKS_DIR  # type: ignore
    else:
        raise ImportError
except Exception:  # pragma: no cover
    from src.utils import logging as log  # type: ignore
    from src.utils.fileio import ensure_dir, append_with_rolloff  # type: ignore
    from src.utils.numerics import (
        winsorize, zscore, clip_inf_nan, ridge_fit, safe_lstsq, add_constant
    )  # type: ignore
    from src.trading_calendar import shift_trading_day  # type: ignore
    FACTOR_RETURNS_CSV = "out/ts/factor_returns.csv"
    METRICS_CSV = "out/ts/metrics.csv"
    COVERAGE_CSV = "out/ts/coverage.csv"
    ROLLING_KEEP_DAYS = 252
    WINSOR_PCT = 0.01
    RIDGE_ALPHA = 1e-6
    LOG_VERBOSITY = "STEP"
    from src.utils.state import DEFAULT_LOCKS_DIR  # type: ignore

F_COLS: List[str] = [f"f{i}" for i in range(1, 192)]  # f1..f191


# ------------------------------ Helpers -------------------------------------
def _to_ts(x: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def _std_cols(df: pd.DataFrame, winsor_pct: float) -> pd.DataFrame:
    """
    Column-wise: winsorize(two-sided) -> zscore. NaN-safe. Return a copy.
    """
    if df is None or df.shape[1] == 0:
        return pd.DataFrame(index=getattr(df, "index", None))
    out = df.copy()
    lo, hi = float(winsor_pct), float(1.0 - winsor_pct)
    for c in out.columns:
        s = pd.to_numeric(out[c], errors="coerce")
        s = winsorize(s, lower=lo, upper=hi)
        s = zscore(s)
        out[c] = s
    return out


def _clean_inds(inds: pd.DataFrame) -> pd.DataFrame:
    """
    Keep numeric columns, drop duplicate-named and all-zero dummy columns.
    """
    if inds is None or inds.shape[1] == 0:
        return pd.DataFrame(index=getattr(inds, "index", None))
    X = inds.copy()
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.select_dtypes(include=[np.number])
    zero_cols = [c for c in X.columns if (X[c].fillna(0) == 0).all()]
    if zero_cols:
        head = zero_cols[:8]
        tail = "..." if len(zero_cols) > 8 else ""
        log.warn(f"[REG][IND] drop all-zero dummies: {len(zero_cols)} -> {head}{tail}")
        X = X.drop(columns=zero_cols)
    return X


def _ensure_lock_subdir_for(path: str | Path) -> None:
    """
    with_file_lock(name) creates lock files like 'state/locks/<name>.lock'.
    If <name> has subdirs (e.g., 'out/ts/metrics.csv'), we pre-create 'state/locks/out/ts'.
    """
    lock_root = Path(DEFAULT_LOCKS_DIR)
    lock_subdir = lock_root / Path(path).parent
    ensure_dir(lock_subdir)


def _align_and_build_X(
    y: pd.Series,
    styles: pd.DataFrame,
    inds: pd.DataFrame,
    resid_f: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Align y with RHS (industry dummies + styles + residualized factors).
    Returns aligned y (attrs preserved) and X.
    """
    # 对齐索引
    idx = pd.Index(y.index, name="code")
    X_parts = []

    if inds is not None and not inds.empty:
        X_parts.append(inds.reindex(idx).fillna(0.0))

    if styles is not None and not styles.empty:
        X_parts.append(styles.reindex(idx))

    if resid_f is not None and not resid_f.empty:
        X_parts.append(resid_f.reindex(idx))

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=idx)

    # drop rows where y or all X are NaN
    mask = y.notna()
    if not X.empty:
        mask &= X.notna().any(axis=1)

    y_aligned = y[mask]
    X_aligned = X.loc[mask]

    # 保留 attrs（关键：保住 y.attrs['date']）
    try:
        y_aligned.attrs = dict(y.attrs)
    except Exception:
        pass

    return y_aligned, X_aligned


# ------------------------------ Public API -----------------------------------
def forward_return(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str], fwd_days: int = 2) -> pd.Series:
    """
    Compute forward return r_{t->t+fwd_days} = log(close_{t+fwd}) - log(close_t).
    Only t and t+fwd_days are required; if either side missing, return NaN series.
    """
    # required columns
    need_cols = ["date", "code", "close"]
    for c in need_cols:
        if c not in panel.columns:
            raise ValueError(f"panel missing required column: {c}")

    # normalize and slice
    df = panel[need_cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    codes = list(codes)

    # map t and t+fwd by trading calendar
    from .trading_calendar import shift_trading_day
    t = pd.to_datetime(date).normalize()
    t2 = shift_trading_day(t, n=int(fwd_days))

    # pick closes
    mask_t = (df["date"] == t) & df["code"].isin(codes)
    mask_t2 = (df["date"] == t2) & df["code"].isin(codes)

    s_t = (
        df.loc[mask_t, ["code", "close"]]
          .drop_duplicates("code")
          .set_index("code")["close"]
          .reindex(codes)
    )
    s_t2 = (
        df.loc[mask_t2, ["code", "close"]]
          .drop_duplicates("code")
          .set_index("code")["close"]
          .reindex(codes)
    )

    # if either side largely missing, return NaNs (上层据此跳过该 t)
    if s_t.isna().all() or s_t2.isna().all():
        y = pd.Series(index=codes, dtype=float, name="fwd_ret")
        y.attrs["date"] = t
        y.attrs["horizon_days"] = int(fwd_days)
        return y

    y = (np.log(s_t2) - np.log(s_t)).astype(float)
    y.name = "fwd_ret"
    y.attrs["date"] = t
    y.attrs["horizon_days"] = int(fwd_days)
    return y



def cs_factor_returns(
    y: pd.Series,
    styles: pd.DataFrame,
    inds: pd.DataFrame,
    resid_f: pd.DataFrame,
    use_ridge: bool = True,
    ridge_alpha: float = 1e-6,
) -> pd.Series:
    """
    Estimate factor returns via cross-sectional regression:
        y ~ styles + inds + resid_f
    """
    t0 = time.time()
    log.step("[STEP 3/6] Cross-sectional regression ...")

    # 1) 回归日必须由 forward_return 注入
    if not hasattr(y, "attrs") or "date" not in y.attrs:
        raise ValueError("y.attrs['date'] missing; forward_return must set it.")
    date_ts = _to_ts(y.attrs["date"])

    # 2) 数值化 y
    y = pd.to_numeric(y, errors="coerce")

    # 3) 构建设计矩阵 —— 用你项目里原来的签名/返回：只返回 (y_aligned, X)
    y_aligned, X = _align_and_build_X(y, styles, inds, resid_f)
    if not isinstance(X, pd.DataFrame) or not isinstance(y_aligned, pd.Series):
        raise TypeError(f"X/y must be pandas objects, got X={type(X)}, y={type(y_aligned)}")

    # 最小统计信息（为 _persist_outputs 兼容）
    N, p = int(y_aligned.shape[0]), int(X.shape[1])
    st = {"n_use_rows": N, "n_total_cols": p}

    # 4) 样本量守门
    safety_margin = 5
    if p == 0 or N < (p + safety_margin):
        log.warn(f"[REG] skip: insufficient sample (N={N}, p={p})")
        betas = pd.Series(index=F_COLS, dtype=float)
        _persist_outputs(date_ts, betas, r2=np.nan, rmse=np.nan, alpha=np.nan,
                         ridge_alpha=float(ridge_alpha), fail=1, reason="singular_or_lowN", coverage=st)
        log.done("[STEP 3/6] Cross-sectional regression ... done (skipped)")
        return betas

    # 5) 拟合
    try:
        from .utils.numerics import ridge_fit, safe_lstsq

        if use_ridge:
            coefs, intercept, info = ridge_fit(
                X, y_aligned,
                alpha=float(ridge_alpha),
                fit_intercept=True,
                penalize_intercept=False,
            )
            if not isinstance(coefs, pd.Series):
                coefs = pd.Series(np.asarray(coefs).reshape(-1), index=X.columns, dtype=float)
            alpha_hat = float(intercept) if intercept is not None else np.nan
        else:
            # 如果你工程没有 add_constant，可改成：
            # X_aug = X.copy(); X_aug.insert(0, "Intercept", 1.0)
            X_aug = add_constant(X, name="Intercept")
            coef_series, _rank = safe_lstsq(X_aug, y_aligned)
            if not isinstance(coef_series, pd.Series):
                coef_series = pd.Series(np.asarray(coef_series).reshape(-1), index=X_aug.columns, dtype=float)
            alpha_hat = float(coef_series.get("Intercept", np.nan))
            coefs = coef_series.drop(labels=["Intercept"], errors="ignore")
            info = {}

        # 6) 统一计算统计量
        y_hat = (X @ coefs.reindex(X.columns)).astype(float)
        if not np.isnan(alpha_hat):
            y_hat = y_hat + alpha_hat
        resid = (y_aligned - y_hat).astype(float)

        ss_res = float(np.square(resid).sum())
        ss_tot = float(np.square(y_aligned - y_aligned.mean()).sum())
        r2 = (np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot)
        dof = max(N - max(1, p), 1)
        rmse = float(np.sqrt(ss_res / dof))
    except Exception as e:
        log.error(f"[REG] regression failed: {type(e).__name__}: {e}")
        betas = pd.Series(index=F_COLS, dtype=float)
        _persist_outputs(date_ts, betas, r2=np.nan, rmse=np.nan, alpha=np.nan,
                         ridge_alpha=float(ridge_alpha), fail=1,
                         reason=f"exception:{type(e).__name__}", coverage=st)
        log.done("[STEP 3/6] Cross-sectional regression ... done (failed)")
        return betas

    # 7) 仅残差因子落盘
    betas = pd.Series({c: np.nan for c in F_COLS}, dtype=float)
    shared = [c for c in F_COLS if c in coefs.index]
    if shared:
        betas.loc[shared] = coefs.loc[shared].values
    else:
        log.warn(f"[REG] no residual factor overlap on {date_ts:%Y-%m-%d}: "
                 f"|F_COLS|={len(F_COLS)}, |coef_index|={len(coefs.index)}")

    log.step(f"[REG] R2={r2:.4f}, N={N}, p={p}")

    _persist_outputs(date_ts, betas, r2=r2, rmse=rmse, alpha=alpha_hat,
                     ridge_alpha=float(ridge_alpha), fail=0, reason=None, coverage=st)

    elapsed = time.time() - t0
    log.done(f"[STEP 3/6] Cross-sectional regression ... done  time={elapsed:.2f}s")
    return betas




# ------------------------------ Persistence ----------------------------------
def _persist_outputs(
    date_ts: pd.Timestamp,
    betas: pd.Series,
    *,
    r2: float,
    rmse: float,
    alpha: float,
    ridge_alpha: float,
    fail: int,
    reason: Optional[str],
    coverage: Dict[str, int],
) -> None:
    """Write factor_returns.csv, metrics.csv, coverage.csv with rolling retention."""
    date_str = date_ts.strftime("%Y-%m-%d")

    # Ensure lock subdirs exist for all targets (Windows-safe)
    _ensure_lock_subdir_for(FACTOR_RETURNS_CSV)
    _ensure_lock_subdir_for(METRICS_CSV)
    _ensure_lock_subdir_for(COVERAGE_CSV)

    # 1) factor_returns.csv
    fr = pd.DataFrame([betas.reindex(F_COLS)], columns=F_COLS)
    fr.insert(0, "date", date_str)
    append_with_rolloff(FACTOR_RETURNS_CSV, fr, key="date", keep_last=int(ROLLING_KEEP_DAYS))

    # 2) metrics.csv
    met_row = {
        "date": date_str,
        "factor_id": "ALL",
        "N": int(coverage.get("n_use_rows", np.nan)),
        "p": int(coverage.get("n_total_cols", np.nan)),
        "ridge_alpha": float(ridge_alpha),
        "R2": float(r2),
        "rmse": float(rmse),
        "alpha": float(alpha) if pd.notna(alpha) else np.nan,
        "fail": int(fail),
    }
    if reason:
        met_row["reason"] = str(reason)
    met = pd.DataFrame([met_row])
    met["date_factor"] = met["date"].astype(str) + "_" + met["factor_id"].astype(str)
    # Keep a large enough rolling window to coexist with per-factor metrics
    keep_last = int(ROLLING_KEEP_DAYS) * (len(F_COLS) + 1)
    append_with_rolloff(METRICS_CSV, met, key="date_factor", keep_last=keep_last)

    # 3) coverage.csv
    N_raw = int(coverage.get("n_raw_rows", 0))
    N_eff = int(coverage.get("n_use_rows", 0))
    filtered_ratio = 0.0 if N_raw == 0 else (1.0 - N_eff / max(N_raw, 1))
    cov_row = {
        "date": date_str,
        "N_raw": N_raw,
        "N_y_nonan": int(coverage.get("n_y_nonan", 0)),
        "N_styles_ok": int(coverage.get("n_styles_ok", 0)),
        "N_eff": N_eff,
        "filtered_ratio": float(filtered_ratio),
        "n_styles": int(coverage.get("n_styles", 0)),
        "n_inds": int(coverage.get("n_inds", 0)),
        "n_factors_present": int(coverage.get("n_factors_present", 0)),
    }
    cov = pd.DataFrame([cov_row])
    append_with_rolloff(COVERAGE_CSV, cov, key="date", keep_last=int(ROLLING_KEEP_DAYS))


# ------------------------------ Smoke Test -----------------------------------
if __name__ == "__main__":  # pragma: no cover
    """
    Smoke test (synthetic):
      - Build a 5-day panel with 40 stocks and random-walk closes.
      - Pick day t (3rd trading day), compute forward 2-day returns.
      - Create styles (3 cols), inds (2 dummies), residual factors (191 cols with only few non-NaNs).
      - Run cs_factor_returns(); expect:
          * factor_returns.csv appended with one row for date=t
          * metrics.csv appended with one row (factor_id=ALL)
          * coverage.csv appended with one row
          * betas index=F_COLS, not all-NaN
    """
    try:
        log.set_verbosity(LOG_VERBOSITY)
    except Exception:
        pass

    rng = np.random.default_rng(20250829)
    # Panel construction
    dates = pd.bdate_range("2024-01-02", periods=5)
    codes = [f"{i:06d}.SH" if i % 2 == 0 else f"{i:06d}.SZ" for i in range(1, 41)]
    rows = []
    for d in dates:
        for c in codes:
            base = 10.0 + (hash(c) % 100) / 10.0
            eps = rng.normal(0, 0.02)
            price = base * (1.0 + eps) ** (dates.get_loc(d) + 1)
            rows.append({"date": d, "code": c, "close": price, "volume": 1_000})
    panel = pd.DataFrame(rows)

    t = dates[2]  # day-3
    y = forward_return(panel, t, codes, fwd_days=2)
    # Exposures (index aligned to codes)
    idx = pd.Index(codes, name="code")
    styles = pd.DataFrame({
        "size": rng.normal(0, 1, len(idx)),
        "value": rng.normal(0, 1, len(idx)),
        "liq": rng.normal(0, 1, len(idx)),
    }, index=idx)
    inds = pd.get_dummies(pd.Series([0 if i % 3 else 1 for i in range(len(idx))], index=idx, name="ind"), prefix="IND")
    # Residual factors: only few filled; rest NaN to emulate real case
    resid_f = pd.DataFrame(index=idx, columns=F_COLS, dtype=float)
    for f in ["f2", "f5", "f17", "f88", "f131"]:
        s = rng.normal(0, 1, len(idx))
        s[rng.choice(len(idx), size=8, replace=False)] = np.nan
        resid_f[f] = s

    # Attach date into attrs for cs_factor_returns()
    y.attrs["date"] = t

    betas = cs_factor_returns(y, styles, inds, resid_f, use_ridge=True, ridge_alpha=RIDGE_ALPHA)
    assert set(betas.index) == set(F_COLS)
    present = ["f2", "f5", "f17", "f88", "f131"]
    assert np.isfinite(betas.loc[present]).any()

    print("[SMOKE][REG] OK")
