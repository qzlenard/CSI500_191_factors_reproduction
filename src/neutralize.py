# file: src/neutralize.py
# -*- coding: utf-8 -*-
"""
[CN] 正交化契约：逐日截面，将每个因子在“行业+风格”上回归，残差即正交化暴露。
[Design]
- y = factor (winsorize→zscore)；X = styles_std (winsorize→zscore) + inds_onehot；Ridge(α=RIDGE_ALPHA)。
- 样本量守卫：N < p+5 则跳过。
- 产物：返回残差矩阵；默认写 out/residuals/{YYYYMMDD}_residuals.csv。
- 历史补齐可设置：factors.attrs['skip_write_residual']=True → 跳过写残差CSV，仅记录 metrics。
- 文件命名日期来自 factors.attrs['date']；缺失时退化为最近交易日。

[接口稳定性] 已对照 docs/api_contract.csv：orthogonalize(factors, styles, inds) → DataFrame（不改签名）
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import time
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- Imports & Config (package-first; fallback defaults) ----------------
try:
    if __package__:
        from .utils import logging as log  # type: ignore
        from .utils.numerics import (  # type: ignore
            winsorize, zscore, ridge_fit, safe_lstsq, add_constant
        )
        from .utils.fileio import ensure_dir, write_csv_atomic, append_with_rolloff  # type: ignore
        from .trading_calendar import most_recent_trading_day  # type: ignore
        from config import (  # type: ignore
            WINSOR_PCT, RIDGE_ALPHA, ROLLING_KEEP_DAYS, LOG_VERBOSITY,
        )
        # optional path patterns
        RESIDUALS_CSV_PATTERN = "out/residuals/{yyyymmdd}_residuals.csv"
        METRICS_CSV = "out/ts/metrics.csv"
    else:
        raise ImportError
except Exception:  # pragma: no cover
    # Fallbacks for standalone smoke/test
    from src.utils import logging as log  # type: ignore
    from src.utils.numerics import (  # type: ignore
        winsorize, zscore, ridge_fit, safe_lstsq, add_constant
    )
    from src.utils.fileio import ensure_dir, write_csv_atomic, append_with_rolloff  # type: ignore
    from src.trading_calendar import most_recent_trading_day  # type: ignore
    WINSOR_PCT = 0.01
    RIDGE_ALPHA = 1e-6
    ROLLING_KEEP_DAYS = 252
    LOG_VERBOSITY = "STEP"
    RESIDUALS_CSV_PATTERN = "out/residuals/{yyyymmdd}_residuals.csv"
    METRICS_CSV = "out/ts/metrics.csv"

# ------------------------------ Constants -------------------------------------
F_COLS: List[str] = [f"f{i}" for i in range(1, 192)]  # f1..f191


# ------------------------------ Utilities -------------------------------------
def _infer_date(factors: pd.DataFrame) -> pd.Timestamp:
    """从 factors.attrs['date'] 解析截面日期；缺失则退化为最近交易日。"""
    try:
        dt = pd.Timestamp(getattr(getattr(factors, "attrs", {}), "get", lambda *_: None)("date"))
        if pd.isna(dt):
            raise ValueError("NaT")
        return dt.normalize()
    except Exception:
        return most_recent_trading_day(pd.Timestamp.today().normalize())


def _stdize_styles(df: pd.DataFrame, winsor_pct: float) -> pd.DataFrame:
    """Styles：逐列双侧 winsorize(lower=p, upper=1-p) 后 z-score。"""
    if df is None or df.empty:
        return pd.DataFrame(index=pd.Index([], name="code"))
    out = df.copy()
    lower = float(winsor_pct)
    upper = float(1.0 - lower)
    if not (0.0 <= lower < 0.5):
        lower, upper = 0.01, 0.99
    for c in out.columns:
        s = pd.to_numeric(out[c], errors="coerce")
        s = winsorize(s, lower=lower, upper=upper)
        s = zscore(s)
        out[c] = s
    return out


def _clean_inds(inds: pd.DataFrame) -> pd.DataFrame:
    """行业虚拟变量：去重名、去全零、仅保留数值列。"""
    X = inds.copy()
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    X = X.select_dtypes(include=[np.number])
    zero_cols = [c for c in X.columns if (pd.to_numeric(X[c], errors="coerce").fillna(0) == 0).all()]
    if zero_cols:
        log.warn(f"[NEU][IND] drop all-zero dummies: {len(zero_cols)}")
        X = X.drop(columns=zero_cols)
    return X


def _build_design(styles_std: pd.DataFrame, inds_ok: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    """X = styles_std (+ inds_ok)，按 idx 对齐。不加常数列（ridge_fit 走 fit_intercept）。"""
    blocks = [styles_std.reindex(idx)]
    if inds_ok is not None and not inds_ok.empty:
        blocks.append(inds_ok.reindex(idx))
    X = pd.concat(blocks, axis=1)
    return X


def _fit_and_residuals(X: pd.DataFrame, y: pd.Series, alpha: float) -> Tuple[pd.Series, Dict[str, float]]:
    """
    首选 Ridge（带截距、不惩罚截距）；失败则 OLS（对 X 添加常数列）。
    返回：(residual_series, {'r2','n','rmse'}).
    """
    # 对齐非空样本
    y = pd.to_numeric(y, errors="coerce")
    ok = y.notna()
    y = y[ok]
    X = X.loc[ok]
    n, p = X.shape
    if n == 0 or p == 0:
        return pd.Series(index=y.index, dtype=float), {"r2": np.nan, "n": int(n), "rmse": np.nan}

    # Ridge 路径（与 utils.numerics.ridge_fit 契约一致）
    try:
        coef, intercept, info = ridge_fit(
            X, y, alpha=float(alpha), fit_intercept=True, penalize_intercept=False
        )
        coef = coef.reindex(X.columns).fillna(0.0)
        y_hat = (X @ coef) + float(intercept)
        resid = y - y_hat
        sst = float(np.nansum((y - float(y.mean())) ** 2))
        sse = float(np.nansum((resid) ** 2))
        r2 = np.nan if sst <= 0 else float(1.0 - sse / sst)
        rmse = float(np.sqrt(np.nanmean((resid) ** 2))) if len(resid) else np.nan
        return pd.Series(resid, index=y.index), {"r2": r2, "n": int(len(y)), "rmse": rmse}
    except Exception:
        # 兜底：OLS（在 X 上加常数列）
        try:
            X_ = add_constant(X)  # 添加 'const' 列
            beta, _rank = safe_lstsq(X_, y)
            beta = beta.reindex(X_.columns).fillna(0.0)
            y_hat = X_ @ beta
            resid = y - y_hat
            sst = float(np.nansum((y - float(y.mean())) ** 2))
            sse = float(np.nansum((resid) ** 2))
            r2 = np.nan if sst <= 0 else float(1.0 - sse / sst)
            rmse = float(np.sqrt(np.nanmean((resid) ** 2))) if len(resid) else np.nan
            return pd.Series(resid, index=y.index), {"r2": r2, "n": int(len(y)), "rmse": rmse}
        except Exception:
            return pd.Series(index=y.index, dtype=float), {"r2": np.nan, "n": int(n), "rmse": np.nan}


# ------------------------------ Public API ------------------------------------
def orthogonalize(factors: pd.DataFrame, styles: pd.DataFrame, inds: pd.DataFrame) -> pd.DataFrame:
    """
    对每个因子做截面回归（行业+风格），返回残差矩阵；同时写 metrics.csv，并在需要时写 residuals/CSV。
    """
    try:
        log.set_verbosity(LOG_VERBOSITY)
    except Exception:
        pass

    t0 = time.time()
    log.step("[NEU] start orthogonalization ...")

    # 代码集对齐
    idx = pd.Index(sorted(set(factors.index) & set(styles.index) & set(inds.index)), name="code")
    if idx.empty:
        log.warn("[NEU] empty intersection of codes; return empty residuals.")
        return pd.DataFrame(index=pd.Index([], name="code"), columns=F_COLS)

    # 清洗 RHS
    styles_std = _stdize_styles(styles.loc[idx], winsor_pct=float(WINSOR_PCT))
    inds_ok = _clean_inds(inds.loc[idx])
    X0 = _build_design(styles_std, inds_ok, idx)

    # 结果框 & 指标
    resid_df = pd.DataFrame(index=idx, columns=F_COLS, dtype=float)
    metrics_rows: List[Dict] = []

    total = len(F_COLS)
    start = time.time()
    ridge_alpha = float(RIDGE_ALPHA)
    safety_margin = 5
    date_ts = _infer_date(factors)
    date_str = date_ts.strftime("%Y-%m-%d")

    # 主循环
    for i, fcol in enumerate(F_COLS, start=1):
        if i == 1 or i % 20 == 0 or i == total:
            log.loop_progress(task="[NEU]", current=i, total=total, start_time=start, every=1)

        y = pd.to_numeric(factors.get(fcol), errors="coerce")
        if y.dropna().empty:
            # log.warn(f"[NEU] {fcol} all-NaN today; skip.")
            metrics_rows.append({
                "date": date_str, "factor_id": fcol, "N": 0, "p": X0.shape[1],
                "ridge_alpha": ridge_alpha, "R2": np.nan, "rmse": np.nan, "fail": 1, "reason": "all-NaN",
            })
            continue
        if y is None:
            continue

        # LHS winsorize + zscore
        lower = float(WINSOR_PCT); upper = float(1.0 - float(WINSOR_PCT))
        y = winsorize(y, lower=lower, upper=upper)
        y = zscore(y)

        # 有效样本与维度
        ok = y.notna()
        X = X0.loc[ok]
        y = y[ok]
        N, p = X.shape
        if N < p + safety_margin:
            log.warn(f"[NEU] {fcol} insufficient sample: N={N} < p+{safety_margin}={p+safety_margin}; skip.")
            metrics_rows.append({
                "date": date_str, "factor_id": fcol, "N": int(N), "p": int(p),
                "ridge_alpha": ridge_alpha, "R2": np.nan, "rmse": np.nan, "fail": 1, "reason": "insufficient N",
            })
            continue

        # 回归与残差
        resid_s, info = _fit_and_residuals(X, y, alpha=ridge_alpha)
        if not resid_s.dropna().empty:
            resid_df.loc[resid_s.index, fcol] = resid_s.values
            metrics_rows.append({
                "date": date_str, "factor_id": fcol, "N": int(info.get("n", N)), "p": int(p),
                "ridge_alpha": ridge_alpha, "R2": float(info.get("r2", np.nan)),
                "rmse": float(info.get("rmse", np.nan)), "fail": 0,
            })
        else:
            metrics_rows.append({
                "date": date_str, "factor_id": fcol, "N": int(N), "p": int(p),
                "ridge_alpha": ridge_alpha, "R2": np.nan, "rmse": np.nan, "fail": 1, "reason": "fit_fail",
            })

    # 残差写盘（可跳过）
    skip_write = bool(getattr(factors, "attrs", {}).get("skip_write_residual", False))
    if not skip_write:
        yyyymmdd = date_ts.strftime("%Y%m%d")
        out_path = RESIDUALS_CSV_PATTERN.format(yyyymmdd=yyyymmdd)
        ensure_dir(out_path, is_file=True)
        out = resid_df.copy()
        out.insert(0, "code", out.index)
        write_csv_atomic(out_path, out.reset_index(drop=True), index=False)
    else:
        log.step(f"[NEU] skip residual write by attrs.skip_write_residual=True @ {date_str}")

    # 指标写盘（滚动窗口：keep_last = 252*191）
    met_df = pd.DataFrame(metrics_rows)
    met_df["date_factor"] = met_df["date"].astype(str) + "_" + met_df["factor_id"].astype(str)
    keep_last = int(ROLLING_KEEP_DAYS) * len(F_COLS)
    append_with_rolloff(METRICS_CSV, met_df, key="date_factor", keep_last=keep_last)

    elapsed = time.time() - t0
    ok_cnt = int((met_df["fail"] == 0).sum())
    fail_cnt = int((met_df["fail"] != 0).sum())
    log.done(f"[NEU] time={elapsed:.2f}s ok={ok_cnt} fail={fail_cnt}")
    return resid_df


# ------------------------------ Smoke Test ------------------------------------
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(42)
    n = 20
    idx = pd.Index([f"C{i:03d}.SZ" for i in range(n)], name="code")

    # 构造：仅 f1/f2 有暴露，f7 全空，其余默认 NaN
    F = pd.DataFrame(index=idx, columns=F_COLS, dtype=float)
    F["f1"] = rng.normal(size=n)
    F["f2"] = rng.normal(size=n)
    F["f7"] = np.nan
    F.attrs["date"] = pd.Timestamp("2024-03-05")

    S = pd.DataFrame({
        "size": rng.normal(size=n),
        "bp": rng.normal(size=n),
        "growth": rng.normal(size=n),
    }, index=idx)

    I = pd.DataFrame({
        "IND_A": [1 if i % 2 == 0 else 0 for i in range(n)],
        "IND_B": [0 if i % 2 == 0 else 1 for i in range(n)],
    }, index=idx)

    log.step("[SMOKE][NEU] run orthogonalization on synthetic data ...")
    R = orthogonalize(F, S, I)

    # Checks
    assert R.shape == (len(idx), 191), f"Unexpected residual shape: {R.shape}"
    mu = float(pd.to_numeric(R["f2"], errors="coerce").dropna().mean())
    assert np.isfinite(mu) and abs(mu) < 1e-2, f"Residual mean not near 0: {mu}"
    assert R["f7"].dropna().empty, "f7 should be all NaN due to insufficient N"

    out_path = RESIDUALS_CSV_PATTERN.format(yyyymmdd="20240305")
    ensure_dir(out_path, is_file=True)
    log.done("[SMOKE][NEU] OK")
