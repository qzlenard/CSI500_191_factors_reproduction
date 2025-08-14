"""
[CN] 数学与稳健回归契约：winsor、z-score、Ridge/OLS 安全封装。
[Purpose] Numerical helpers used by neutralization & regression.

Interfaces:
- winsorize(s: pd.Series, pct: float) -> pd.Series
- zscore(s: pd.Series) -> pd.Series
- clip_inf_nan(df: pd.DataFrame) -> pd.DataFrame
- ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict
  * Returns {'coef': np.ndarray, 'intercept': float, 'r2': float, 'n': int}
- safe_lstsq(X: np.ndarray, y: np.ndarray) -> dict
  * Fallback OLS with checks; same return keys.

Contracts:
- Center/standardize X if required by caller (policy in caller).
- Robust to rank deficiency; never crash—return NaNs and WARN instead.
"""
from __future__ import annotations

# file: src/utils/numerics.py
# -*- coding: utf-8 -*-
"""
Numerical utilities for factor research:
- Cleaning: clip_inf_nan
- Robust exposure processing: winsorize (with optional grouping), zscore (with optional grouping)
- Standardization pipeline: standardize_exposures (winsorize -> zscore)
- Linear algebra helpers: add_constant, safe_lstsq
- Ridge regression: ridge_fit (weighted, intercept optional; intercept not penalized by default)
- Stats helpers: weighted_mean, weighted_r2, sample_size_ok
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

# Optional lightweight project logger
try:
    from .logging import warn as _log_warn  # type: ignore
except Exception:  # pragma: no cover
    def _log_warn(msg: str) -> None:
        print(f"[WARN][numerics] {msg}")

ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray]


__all__ = [
    "clip_inf_nan",
    "winsorize",
    "zscore",
    "standardize_exposures",
    "add_constant",
    "safe_lstsq",
    "ridge_fit",
    "weighted_mean",
    "weighted_r2",
    "sample_size_ok",
]


# ----------------------------- Cleaning ------------------------------------- #
def clip_inf_nan(x: ArrayLike, fill_nan: Optional[float] = None) -> ArrayLike:
    """
    Replace +/- inf with NaN; optionally fill remaining NaN with a scalar.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        y = x.replace([np.inf, -np.inf], np.nan)
        if fill_nan is not None:
            y = y.fillna(fill_nan)
        return y
    arr = np.asarray(x, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    if fill_nan is not None:
        arr = np.where(np.isnan(arr), fill_nan, arr)
    return arr


# ----------------------------- Helper utils --------------------------------- #
def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected DataFrame")
    num = df.select_dtypes(include=[np.number])
    if len(num.columns) < len(df.columns):
        dropped = [c for c in df.columns if c not in num.columns]
        if dropped:
            _log_warn(f"Non-numeric columns dropped in numerics op: {dropped}")
    return num


def add_constant(X: pd.DataFrame, name: str = "Intercept") -> pd.DataFrame:
    """
    Add a constant column of ones if not already present.
    """
    if name in X.columns:
        return X
    const = pd.Series(1.0, index=X.index, name=name)
    return pd.concat([X, const], axis=1)


def sample_size_ok(n_obs: int, n_params: int, margin: int = 5) -> bool:
    """
    Check minimal sample size for regression: require n_obs >= n_params + margin.
    """
    try:
        n_obs = int(n_obs)
        n_params = int(n_params)
    except Exception:
        return False
    return n_obs >= (n_params + margin)


# ----------------------------- Winsorize/ZScore ------------------------------ #
def _winsorize_block(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """
    Column-wise winsorization using scalar bounds per column to satisfy type checkers.
    """
    df_num = _ensure_numeric_df(df).copy()
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("Invalid quantile bounds for winsorize")

    # Quantile per column
    q = df_num.quantile([lower, upper], interpolation="linear")
    lo = q.loc[lower]
    hi = q.loc[upper]

    # Clip each column with scalar bounds to avoid type checker warnings
    for c in df_num.columns:
        lb = None if not np.isfinite(lo[c]) else float(lo[c])
        ub = None if not np.isfinite(hi[c]) else float(hi[c])
        df_num[c] = df_num[c].clip(lower=lb, upper=ub)
    return df_num


def winsorize(
    x: Union[pd.Series, pd.DataFrame],
    lower: float = 0.01,
    upper: float = 0.99,
    by: Optional[pd.Series] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Quantile winsorization per column (or per group if `by` provided).
    """
    if isinstance(x, pd.Series):
        df = x.to_frame(name=x.name if x.name is not None else "value")
        res = winsorize(df, lower=lower, upper=upper, by=by)
        return res.iloc[:, 0].rename(x.name)

    df = x.copy()
    if by is None:
        df_num = _winsorize_block(df, lower, upper)
        # Reattach any dropped non-numeric columns unchanged
        return pd.concat([df_num, df.drop(columns=df_num.columns, errors="ignore")], axis=1)[df.columns]

    if not isinstance(by, pd.Series):
        by = pd.Series(by, index=df.index)
    parts = []
    for _, sub in df.groupby(by, sort=False):
        parts.append(_winsorize_block(sub, lower, upper).reindex(sub.index))
    out = pd.concat(parts, axis=0).reindex(df.index)
    non_num = df.drop(columns=df.select_dtypes(include=[np.number]).columns, errors="ignore")
    return pd.concat([out, non_num], axis=1)[df.columns]


def _zscore_block(
    df: pd.DataFrame,
    ddof: int = 0,
    clip: Optional[float] = None,
) -> pd.DataFrame:
    df_num = _ensure_numeric_df(df).copy()
    mean = df_num.mean(axis=0, skipna=True)
    std = df_num.std(axis=0, ddof=ddof, skipna=True)
    std_safe = std.replace(0.0, np.nan)
    z = (df_num - mean) / std_safe
    if clip is not None and clip > 0:
        z = z.clip(lower=-clip, upper=clip)
    return z


def zscore(
    x: Union[pd.Series, pd.DataFrame],
    by: Optional[pd.Series] = None,
    ddof: int = 0,
    clip: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Standardize to zero-mean, unit-variance (optionally per group).
    """
    if isinstance(x, pd.Series):
        df = x.to_frame(name=x.name if x.name is not None else "value")
        res = zscore(df, by=by, ddof=ddof, clip=clip)
        return res.iloc[:, 0].rename(x.name)

    df = x.copy()
    if by is None:
        z = _zscore_block(df, ddof=ddof, clip=clip)
        return z.reindex(columns=df.columns)

    if not isinstance(by, pd.Series):
        by = pd.Series(by, index=df.index)
    parts = []
    for _, sub in df.groupby(by, sort=False):
        parts.append(_zscore_block(sub, ddof=ddof, clip=clip).reindex(sub.index))
    out = pd.concat(parts, axis=0).reindex(df.index)
    return out.reindex(columns=df.columns)


def standardize_exposures(
    df: pd.DataFrame,
    winsor: Tuple[float, float] = (0.01, 0.99),
    by: Optional[pd.Series] = None,
    ddof: int = 0,
    clip_z: Optional[float] = None,
) -> pd.DataFrame:
    """
    Pipeline: winsorize -> zscore, optionally by group.
    """
    low, high = winsor
    w = winsorize(df, lower=low, upper=high, by=by)
    z = zscore(w, by=by, ddof=ddof, clip=clip_z)
    return z


# ----------------------------- Stats helpers -------------------------------- #
def weighted_mean(y: pd.Series, w: Optional[pd.Series]) -> float:
    """
    Weighted mean handling NaN in both y and weights.
    """
    if w is None:
        return float(pd.Series(y).mean(skipna=True))
    s = pd.concat([y, w], axis=1).dropna()
    if s.empty:
        return float("nan")
    yw = (s.iloc[:, 0] * s.iloc[:, 1]).sum()
    ww = s.iloc[:, 1].sum()
    return float(yw / ww) if ww != 0 else float("nan")


def weighted_r2(y: pd.Series, y_hat: pd.Series, w: Optional[pd.Series]) -> float:
    """
    Weighted R² = 1 - SSE/SST (with weighted mean baseline).
    """
    df = pd.concat([y, y_hat], axis=1)
    df.columns = ["y", "yhat"]
    if w is not None:
        df = pd.concat([df, w.rename("w")], axis=1)
    df = df.dropna()
    if df.empty:
        return float("nan")
    if w is None:
        ybar = df["y"].mean()
        sse = ((df["y"] - df["yhat"]) ** 2).sum()
        sst = ((df["y"] - ybar) ** 2).sum()
    else:
        ybar = (df["y"] * df["w"]).sum() / df["w"].sum() if df["w"].sum() != 0 else float("nan")
        sse = (df["w"] * (df["y"] - df["yhat"]) ** 2).sum()
        sst = (df["w"] * (df["y"] - ybar) ** 2).sum()
    if sst == 0 or not np.isfinite(sst):
        return float("nan")
    return float(1.0 - (sse / sst))


# ----------------------------- Linear solvers ------------------------------- #
def safe_lstsq(
    X: pd.DataFrame,
    y: pd.Series,
    rcond: float = 1e-12,
) -> Tuple[pd.Series, float]:
    """
    Safe least squares using numpy.linalg.lstsq with pandas inputs.
    """
    Xv = X.values.astype(float, copy=False)
    yv = y.values.astype(float, copy=False)
    beta, residuals, rank, s = np.linalg.lstsq(Xv, yv, rcond=rcond)
    coef = pd.Series(beta, index=X.columns, dtype=float)
    return coef, float(rank)


def ridge_fit(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1e-6,
    fit_intercept: bool = False,
    sample_weight: Optional[pd.Series] = None,
    penalize_intercept: bool = False,
) -> Tuple[pd.Series, Optional[float], dict]:
    """
    Ridge regression with optional weights; numerically stable and label-preserving.
    Solve: (X' W X + α * P) β = X' W y
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if df.empty:
        raise ValueError("No valid samples after dropping NaNs")
    y_clean = df.pop("__y__")
    X_clean = df

    intercept_name = "Intercept"
    if fit_intercept:
        X_aug = add_constant(X_clean, name=intercept_name)
    else:
        X_aug = X_clean

    if sample_weight is not None:
        w = pd.Series(sample_weight).reindex(X_aug.index)
        keep = (w.notna()) & (w >= 0)
        if not bool(keep.all()):
            X_aug = X_aug.loc[keep]
            y_clean = y_clean.loc[keep]
            w = w.loc[keep]
        sw = np.sqrt(w.values.astype(float))
        Xw = X_aug.values.astype(float) * sw[:, None]
        yw = y_clean.values.astype(float) * sw
    else:
        Xw = X_aug.values.astype(float)
        yw = y_clean.values.astype(float)

    n, p_aug = Xw.shape
    p = X_clean.shape[1]

    penalty = np.eye(p_aug, dtype=float)
    if fit_intercept and not penalize_intercept:
        if X_aug.columns[-1] != intercept_name:
            j = list(X_aug.columns).index(intercept_name)
        else:
            j = p_aug - 1
        penalty[j, j] = 0.0

    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw
    A = XtX + (alpha * penalty)

    try:
        beta_aug = np.linalg.solve(A, Xty)
        rank = float(np.linalg.matrix_rank(Xw))
    except np.linalg.LinAlgError:
        _log_warn("Ridge solve failed; falling back to lstsq on augmented system.")
        beta_aug, residuals, rank, s = np.linalg.lstsq(A, Xty, rcond=1e-12)

    if fit_intercept:
        if X_aug.columns[-1] == intercept_name:
            coef_vals = beta_aug[:-1]
            intercept = float(beta_aug[-1])
        else:
            j = list(X_aug.columns).index(intercept_name)
            intercept = float(beta_aug[j])
            mask = np.ones_like(beta_aug, dtype=bool)
            mask[j] = False
            coef_vals = beta_aug[mask]
            coef_vals = coef_vals[np.array([i for i in range(len(mask)) if mask[i]])]
    else:
        coef_vals = beta_aug
        intercept = None

    coef = pd.Series(coef_vals, index=X_clean.columns, dtype=float)

    y_hat_vec = (X_clean.values @ coef.values) + (intercept if intercept is not None else 0.0)
    y_hat = pd.Series(y_hat_vec, index=X_clean.index)
    r2 = weighted_r2(y_clean, y_hat, sample_weight.reindex(X_clean.index) if sample_weight is not None else None)

    info = {"n": int(n), "p": int(p), "rank": float(rank), "r2": float(r2)}
    return coef, intercept, info


# ----------------------------- Convenience ---------------------------------- #
def _demo() -> None:  # pragma: no cover
    """
    Quick smoke test to illustrate API.
    """
    rng = np.random.default_rng(123)
    n, p = 200, 5
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    beta_true = np.array([0.5, -0.2, 0.0, 0.3, 0.0])
    y = pd.Series(X.values @ beta_true + rng.normal(scale=0.2, size=n), name="y")

    X_clean = standardize_exposures(X, winsor=(0.01, 0.99), by=None, clip_z=5.0)
    coef, intercept, info = ridge_fit(X_clean, y, alpha=1e-6, fit_intercept=True)
    print("[demo] coef:\n", coef.round(3))
    print("[demo] intercept:", intercept)
    print("[demo] info:", info)


if __name__ == "__main__":  # pragma: no cover
    _demo()
