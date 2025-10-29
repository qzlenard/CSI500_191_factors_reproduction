# file: src/portfolio.py
# -*- coding: utf-8 -*-
"""
Portfolio construction: from alpha cross-section to today's orders.

本版要点（修复 linprog A_eq NaN/Inf）：
1) 在 _align_universe 中对 styles/inds/bench/prev 做强力净化：
   - to_numeric(errors='coerce') → replace(±inf, nan) → fillna(0) → clip负值为0（bench/prev） → 归一 sum=1（bench）。
   - 列名、索引统一，确保矩阵/向量全是有限数。
2) 在 _lp_bmark_neutral 中继续兜底：构造等式/不等式前 assert 全 finite；若发现非有限值，直接转为 0 并告警。
3) 其余逻辑保持：支持 mode="long_only_topk" 与 "optimizer_bmark_neutral"；允许做空（allow_short）。

对外签名：
build_orders_from_alpha(alpha, mode="long_only_topk", top_k=50, max_weight=0.05,
                        neutral=False, turnover_threshold=0.01,
                        prev_target_weights=None, allow_short=False)
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# -------------------- Imports & Config --------------------
try:
    if __package__:
        from .utils import logging as log  # type: ignore
        from .utils.fileio import ensure_dir, write_csv_atomic  # type: ignore
        from .utils.numerics import clip_inf_nan  # type: ignore
        from .trading_calendar import most_recent_trading_day  # type: ignore
        from config import (  # type: ignore
            ORDERS_CSV_PATTERN, LOG_VERBOSITY,
            PORTFOLIO_TOP_K, PORTFOLIO_MAX_WEIGHT, TURNOVER_THRESHOLD, CFG
        )
    else:
        raise ImportError
except Exception:  # pragma: no cover
    from src.utils import logging as log  # type: ignore
    from src.utils.fileio import ensure_dir, write_csv_atomic  # type: ignore
    from src.utils.numerics import clip_inf_nan  # type: ignore
    from src.trading_calendar import most_recent_trading_day  # type: ignore
    ORDERS_CSV_PATTERN = "out/orders/{yyyymmdd}_orders.csv"
    LOG_VERBOSITY = "STEP"
    PORTFOLIO_TOP_K = 50
    PORTFOLIO_MAX_WEIGHT = 0.05
    TURNOVER_THRESHOLD = 0.01
    class _CFG:
        class portfolio:
            mode = "long_only_topk"
            top_k = PORTFOLIO_TOP_K
            max_weight = PORTFOLIO_MAX_WEIGHT
            neutral = False
            turnover_threshold = TURNOVER_THRESHOLD
    CFG = _CFG()

try:
    from scipy.optimize import linprog
    _SCIPY_OK = True
except Exception:  # pragma: no cover
    _SCIPY_OK = False


# ------------------------------ Utilities ------------------------------
def _most_recent_trading_day_safe() -> pd.Timestamp:
    try:
        return most_recent_trading_day(None)  # type: ignore
    except TypeError:
        return most_recent_trading_day()      # type: ignore
    except Exception:
        return pd.Timestamp.today().normalize()

def _infer_date_from_alpha(alpha: pd.Series) -> pd.Timestamp:
    d = getattr(alpha, "attrs", {}).get("date", None)
    try:
        return pd.Timestamp(d).normalize() if d is not None else _most_recent_trading_day_safe()
    except Exception:
        return _most_recent_trading_day_safe()

def _orders_from_weights(weights: pd.Series, prev: Optional[pd.Series], date_t: pd.Timestamp) -> pd.DataFrame:
    idx = weights.index
    prev_w = (prev.reindex(idx).fillna(0.0) if prev is not None else pd.Series(0.0, index=idx))
    tw = weights.fillna(0.0).astype(float)
    delta = tw - prev_w
    side = np.where(delta > 1e-10, "BUY", np.where(delta < -1e-10, "SELL", "HOLD"))
    df = pd.DataFrame({
        "date": pd.Timestamp(date_t).strftime("%Y-%m-%d"),
        "code": idx.astype(str),
        "target_weight": tw.values,
        "side": side,
        "px_type": "MKT",
        "note": ""
    })
    return df.loc[df["side"] != "HOLD"].reset_index(drop=True)

def _apply_turnover_band(w_new: pd.Series, w_prev: pd.Series, band: float) -> pd.Series:
    out = w_new.copy()
    w_prev = w_prev.reindex(out.index).fillna(0.0)
    stick = (out - w_prev).abs() < float(band)
    if stick.any():
        out.loc[stick] = w_prev.loc[stick]
    return out

def _waterfill_cap_sum1(v: np.ndarray, cap: float) -> np.ndarray:
    """Given non-negative scores v, cap per-name at 'cap' and normalize to sum=1."""
    v = np.maximum(v, 0.0)
    if v.sum() <= 0:
        return np.full_like(v, 1.0 / len(v))
    v = v / v.sum()
    if cap <= 0:
        return v
    v = np.minimum(v, float(cap))
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / s

# ---------- NEW: robust sanitizers ----------
def _sanitize_series(s: Optional[pd.Series], name: str, nonneg: bool=False, normalize: bool=False, fallback_len: int=0) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(0.0, index=pd.Index([], name="code"))
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if nonneg:
        x = x.clip(lower=0.0)
    if normalize:
        tot = float(x.sum())
        if tot > 0:
            x = x / tot
        else:
            if fallback_len > 0:
                x = pd.Series(1.0 / fallback_len, index=x.index)
    bad = (~np.isfinite(x.values)).sum()
    if bad:
        log.warn(f"[OPT] {_short(name)} had {bad} non-finite -> set 0")
        x = pd.Series(np.nan_to_num(x.values, nan=0.0, posinf=0.0, neginf=0.0), index=x.index, dtype=float)
    return x

def _sanitize_matrix(M: Optional[pd.DataFrame], name: str, align_index: Optional[pd.Index]) -> Optional[pd.DataFrame]:
    if M is None or M is pd.NA or (isinstance(M, pd.DataFrame) and M.empty):
        return None
    D = M.copy()
    if align_index is not None:
        D = D.reindex(align_index)
    for c in D.columns:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    D.columns = [str(c) for c in D.columns]
    # 统计非有限修正（理论上此处已全部有限）
    if not np.isfinite(D.values).all():
        log.warn(f"[OPT] {_short(name)} had non-finite entries -> coerced to 0.")
        D = pd.DataFrame(np.nan_to_num(D.values, nan=0.0, posinf=0.0, neginf=0.0), index=D.index, columns=D.columns)
    return D

def _short(s: str) -> str:
    return (s or "").split("/")[-1][:24]

# ---------- Align & Clean ----------
def _align_universe(alpha: pd.Series,
                    styles: Optional[pd.DataFrame],
                    inds: Optional[pd.DataFrame],
                    bench: Optional[pd.Series],
                    prevw: Optional[pd.Series]) -> Tuple[pd.Index, pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame], pd.Series, pd.Series]:
    """Align all to common universe and sanitize to finite numeric arrays."""
    a = clip_inf_nan(pd.to_numeric(alpha, errors="coerce")).fillna(0.0)
    idx = pd.Index(sorted(a.index.astype(str)), name="code")
    uni = set(idx)
    if styles is not None and not styles.empty:
        uni &= set(styles.index.astype(str))
    if inds is not None and not inds.empty:
        uni &= set(inds.index.astype(str))
    if bench is not None and len(bench):
        uni &= set(bench.index.astype(str))
    if prevw is not None and len(prevw):
        uni &= set(prevw.index.astype(str))
    idx = pd.Index(sorted(uni), name="code")

    # Reindex & sanitize
    a = a.reindex(idx).fillna(0.0)

    S = _sanitize_matrix(styles, "styles", idx) if styles is not None else None
    I = _sanitize_matrix(inds, "inds", idx) if inds is not None else None

    # bench 非负 & 归一（若和为0，用等权）
    b_raw = (pd.Series(0.0, index=idx) if bench is None else bench.reindex(idx).fillna(0.0))
    b = _sanitize_series(b_raw, "bench", nonneg=True, normalize=True, fallback_len=len(idx))
    if abs(float(b.sum()) - 1.0) > 1e-8 and b.sum() > 0:
        b = b / float(b.sum())

    p_raw = (pd.Series(0.0, index=idx) if prevw is None else prevw.reindex(idx).fillna(0.0))
    p = _sanitize_series(p_raw, "prev", nonneg=False, normalize=False)

    return idx, a, S, I, b, p


# ------------------------------ LP: Benchmark-neutral, with/without short ------------------------------
def _lp_bmark_neutral(alpha: pd.Series,
                      styles: Optional[pd.DataFrame],
                      inds: Optional[pd.DataFrame],
                      bench: pd.Series,
                      prevw: pd.Series,
                      max_weight: float,
                      tc: float,
                      allow_short: bool) -> pd.Series:
    """
    minimize   -alpha^T w  +  (tc/2) * sum(u)
    s.t.       full-investment, style/industry equalities relative to bench,
               turnover linearization with u, per-name caps,
               allow_short -> use w = w+ - w-.
    """
    if not _SCIPY_OK:
        raise RuntimeError("scipy is required for optimizer_bmark_neutral")

    N = len(alpha)
    a = np.asarray(alpha.values, dtype=float)
    prev = np.asarray(prevw.values, dtype=float)
    max_w = float(max_weight)
    tc_eff = float(tc) * 0.5

    # 兜底：若仍非有限，置零并告警
    def _finite(name, arr):
        ok = np.isfinite(arr)
        if not ok.all():
            bad = int((~ok).sum())
            log.warn(f"[OPT] {name} had {bad} non-finite -> set 0.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    a = _finite("alpha", a)
    prev = _finite("prev", prev)

    if allow_short:
        # x = [w+, w-, u]
        c = np.concatenate([-a, a, np.full(N, tc_eff, dtype=float)])  # minimize
        bounds = ([(0.0, max_w if max_w > 0 else None)] * N +
                  [(0.0, max_w if max_w > 0 else None)] * N +
                  [(0.0, None)] * N)
        Aeq = []; beq = []
        # sum(w+ - w-) = 1
        row_sum = np.concatenate([np.ones(N), -np.ones(N), np.zeros(N)])
        Aeq.append(row_sum); beq.append(1.0)

        Aub = []; bub = []

        # Style neutrality
        if styles is not None and not styles.empty:
            S = np.asarray(styles.values, dtype=float)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            b_style = (np.asarray((styles.T * bench.values).sum(axis=1), dtype=float))
            b_style = np.nan_to_num(b_style, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(S.shape[1]):
                row = np.concatenate([S[:, j], -S[:, j], np.zeros(N)])
                Aeq.append(row); beq.append(b_style[j])

        # Industry neutrality
        if inds is not None and not inds.empty:
            I = np.asarray(inds.values, dtype=float)
            I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
            b_ind = np.asarray((inds.T @ bench).values, dtype=float)
            b_ind = np.nan_to_num(b_ind, nan=0.0, posinf=0.0, neginf=0.0)
            for k in range(I.shape[1]):
                row = np.concatenate([I[:, k], -I[:, k], np.zeros(N)])
                Aeq.append(row); beq.append(b_ind[k])

        # Turnover abs: u >= (w+ - w-) - prev  &  u >= -(w+ - w-) + prev
        for i in range(N):
            r1 = np.zeros(3*N); r1[i] = 1.0; r1[N+i] = -1.0; r1[2*N+i] = -1.0
            Aub.append(r1); bub.append(prev[i])
            r2 = np.zeros(3*N); r2[i] = -1.0; r2[N+i] = 1.0; r2[2*N+i] = -1.0
            Aub.append(r2); bub.append(-prev[i])

        Aeq = np.asarray(Aeq, dtype=float); beq = np.asarray(beq, dtype=float)
        Aub = np.asarray(Aub, dtype=float) if len(Aub) else None
        bub = np.asarray(bub, dtype=float) if Aub is not None else None

        # 终检：A_eq/A_ub 必须全有限
        if not np.isfinite(Aeq).all() or (Aub is not None and (not np.isfinite(Aub).all() or not np.isfinite(bub).all())):
            log.warn("[OPT] sanitize Aeq/Aub non-finite -> set to 0.")
            if not np.isfinite(Aeq).all():
                Aeq = np.nan_to_num(Aeq, nan=0.0, posinf=0.0, neginf=0.0)
                beq = np.nan_to_num(beq, nan=0.0, posinf=0.0, neginf=0.0)
            if Aub is not None and (not np.isfinite(Aub).all() or not np.isfinite(bub).all()):
                Aub = np.nan_to_num(Aub, nan=0.0, posinf=0.0, neginf=0.0)
                bub = np.nan_to_num(bub, nan=0.0, posinf=0.0, neginf=0.0)

        res = linprog(c=c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        x = res.x
        w = x[:N] - x[N:2*N]
        return pd.Series(w, index=alpha.index, dtype=float)

    else:
        # long-only: x = [w, u]
        c = np.concatenate([-a, np.full(N, tc_eff, dtype=float)])
        bounds = ([(0.0, max_w if max_w > 0 else None)] * N +
                  [(0.0, None)] * N)
        Aeq = []; beq = []
        # sum(w) = 1
        row_sum = np.concatenate([np.ones(N), np.zeros(N)])
        Aeq.append(row_sum); beq.append(1.0)

        Aub = []; bub = []

        if styles is not None and not styles.empty:
            S = np.asarray(styles.values, dtype=float)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            b_style = (np.asarray((styles.T * bench.values).sum(axis=1), dtype=float))
            b_style = np.nan_to_num(b_style, nan=0.0, posinf=0.0, neginf=0.0)
            for j in range(S.shape[1]):
                row = np.concatenate([S[:, j], np.zeros(N)])
                Aeq.append(row); beq.append(b_style[j])

        if inds is not None and not inds.empty:
            I = np.asarray(inds.values, dtype=float)
            I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
            b_ind = np.asarray((inds.T @ bench).values, dtype=float)
            b_ind = np.nan_to_num(b_ind, nan=0.0, posinf=0.0, neginf=0.0)
            for k in range(I.shape[1]):
                row = np.concatenate([I[:, k], np.zeros(N)])
                Aeq.append(row); beq.append(b_ind[k])

        for i in range(N):
            r1 = np.zeros(2*N); r1[i] = 1.0; r1[N+i] = -1.0
            Aub.append(r1); bub.append(prev[i])
            r2 = np.zeros(2*N); r2[i] = -1.0; r2[N+i] = -1.0
            Aub.append(r2); bub.append(-prev[i])

        Aeq = np.asarray(Aeq, dtype=float); beq = np.asarray(beq, dtype=float)
        Aub = np.asarray(Aub, dtype=float) if len(Aub) else None
        bub = np.asarray(bub, dtype=float) if Aub is not None else None

        if not np.isfinite(Aeq).all() or (Aub is not None and (not np.isfinite(Aub).all() or not np.isfinite(bub).all())):
            log.warn("[OPT] sanitize Aeq/Aub non-finite -> set to 0.")
            if not np.isfinite(Aeq).all():
                Aeq = np.nan_to_num(Aeq, nan=0.0, posinf=0.0, neginf=0.0)
                beq = np.nan_to_num(beq, nan=0.0, posinf=0.0, neginf=0.0)
            if Aub is not None and (not np.isfinite(Aub).all() or not np.isfinite(bub).all()):
                Aub = np.nan_to_num(Aub, nan=0.0, posinf=0.0, neginf=0.0)
                bub = np.nan_to_num(bub, nan=0.0, posinf=0.0, neginf=0.0)

        res = linprog(c=c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        x = res.x
        w = x[:N]
        return pd.Series(w, index=alpha.index, dtype=float)


# ------------------------------ Public API ------------------------------
def build_orders_from_alpha(alpha: pd.Series,
                            mode: str="long_only_topk", top_k: int=50, max_weight: float=0.05,
                            neutral: bool=False, turnover_threshold: float=0.01,
                            prev_target_weights: Optional[pd.Series]=None,
                            allow_short: bool=False) -> pd.DataFrame:
    """
    mode: "long_only_topk" | "optimizer_bmark_neutral" | "optimizer"
    allow_short: only used in "optimizer_bmark_neutral"
    """
    try:
        log.set_verbosity(LOG_VERBOSITY)
    except Exception:
        pass

    date_t = _infer_date_from_alpha(alpha)
    ensure_dir(ORDERS_CSV_PATTERN.format(yyyymmdd=date_t.strftime("%Y%m%d")), is_file=True)

    mode_lc = str(mode or CFG.portfolio.mode).lower()
    prevw = prev_target_weights

    if mode_lc == "long_only_topk":
        a = clip_inf_nan(pd.to_numeric(alpha, errors="coerce")).fillna(0.0)
        idx = a.index
        order = a.sort_values(ascending=False).index.tolist()
        capN = max(1, int(math.floor(1.0 / float(max_weight))) or 1)
        K = min(int(top_k), capN, len(order))
        pick = order[:K]
        scores = a.reindex(pick).clip(lower=0.0)
        if float(scores.sum()) <= 0:
            wvals = np.full(K, 1.0 / K, dtype=float)
        else:
            wvals = (scores.values.astype(float) / float(scores.sum()))
        w_new = pd.Series(0.0, index=idx, dtype=float)
        w_new.loc[pick] = _waterfill_cap_sum1(wvals, float(max_weight))
        w_band = _apply_turnover_band(
            w_new, prevw.reindex(idx).fillna(0.0) if prevw is not None else pd.Series(0.0, index=idx),
            float(turnover_threshold)
        )

    elif mode_lc == "optimizer_bmark_neutral":
        attrs: Dict[str, Any] = getattr(alpha, "attrs", {})
        styles = attrs.get("styles", None)
        inds = attrs.get("inds", None)
        bench = attrs.get("bench_weights", None)
        tc = float(attrs.get("tc", 0.0))

        if styles is None or inds is None or bench is None:
            raise ValueError("[OPT-bmark-neutral] require alpha.attrs['styles'], ['inds'], ['bench_weights'].")

        # Align + sanitize to finite numeric
        idx, a, S, I, b, p = _align_universe(alpha, styles, inds, bench, prevw)

        try:
            w_opt = _lp_bmark_neutral(
                a, S, I, b, p, max_weight=float(max_weight), tc=tc, allow_short=bool(allow_short)
            )
        except Exception as ex:
            log.warn(f"[OPT-bmark-neutral] LP failed: {ex}; fallback Top-K.")
            # 安全退化到Top-K（避免整条流水线中断）
            return build_orders_from_alpha(alpha, mode="long_only_topk",
                                           top_k=min(top_k, int(1.0/max_weight)),
                                           max_weight=max_weight, neutral=neutral,
                                           turnover_threshold=turnover_threshold,
                                           prev_target_weights=prevw)

        w_band = _apply_turnover_band(w_opt.reindex(idx).fillna(0.0), p.reindex(idx).fillna(0.0), float(turnover_threshold))

    else:
        # 保持兼容：旧 optimizer 分支退化为 Top-K
        a = clip_inf_nan(pd.to_numeric(alpha, errors="coerce")).fillna(0.0)
        idx = a.index
        pick = a.sort_values(ascending=False).index[:min(int(top_k), len(a))]
        scores = a.reindex(pick).clip(lower=0.0)
        w_new = pd.Series(0.0, index=idx, dtype=float)
        w_new.loc[pick] = _waterfill_cap_sum1(scores.values, float(max_weight))
        w_band = _apply_turnover_band(w_new, prevw if prevw is not None else pd.Series(0.0, index=idx), float(turnover_threshold))

    # ---------- Emit orders ----------
    orders = _orders_from_weights(w_band, prevw, date_t)
    try:
        out_path = ORDERS_CSV_PATTERN.format(yyyymmdd=date_t.strftime("%Y%m%d"))
        write_csv_atomic(out_path, orders, index=False)
        log.done(f"[ORDERS] K={len(orders)} mode={mode_lc} allow_short={allow_short}")
    except Exception as ex:
        log.warn(f"[ORDERS] write failed: {ex}")
    return orders


# ------------------------------ Smoke ------------------------------
if __name__ == "__main__":  # pragma: no cover
    np.random.seed(0)
    N = 60
    idx = pd.Index([f"{i:06d}.SZ" for i in range(N)], name="code")
    a = pd.Series(np.random.randn(N), index=idx)

    # Fake styles/industries
    S = pd.DataFrame({
        "size": np.random.randn(N),
        "value": np.random.randn(N),
        "mom": np.random.randn(N),
    }, index=idx)
    G = 10
    I = pd.DataFrame({f"IND_{g}": ((np.arange(N) % G) == g).astype(int) for g in range(G)}, index=idx)

    bench = pd.Series(1.0/N, index=idx)  # equal-weight benchmark
    prev = pd.Series(0.0, index=idx)

    a.attrs.update(dict(date=pd.Timestamp("2025-09-10"), styles=S, inds=I, bench_weights=bench, tc=0.003))
    o1 = build_orders_from_alpha(a, mode="optimizer_bmark_neutral", max_weight=0.06, prev_target_weights=prev, allow_short=False)
    print("long-only:", o1.head())
    o2 = build_orders_from_alpha(a, mode="optimizer_bmark_neutral", max_weight=0.06, prev_target_weights=prev, allow_short=True)
    print("long-short:", o2.head())
