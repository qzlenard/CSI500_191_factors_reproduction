"""
[CN] 数据接口契约：仅函数签名/输入输出说明，不实现。支持公告日感知与滞后。
[Purpose] Thin wrappers around MyQuant SDK (JQData-like) for reproducible I/O.

Signatures (do NOT implement here):
def get_trade_days(start: str, end: str) -> list[pd.Timestamp]:
    '''
    [CN] 返回交易日列表（闭区间）。
    Returns an ascending list of trading-day Timestamps in [start, end].
    '''

def get_index_members(index_code: str, date: pd.Timestamp) -> list[str]:
    '''
    [CN] 返回指定日期指数成份股代码列表（行情代码如 "000001.SZ"）。
    Returns member codes for the given index on `date`.
    '''

def get_ohlcv(codes: list[str], start: str, end: str, fq: str) -> pd.DataFrame:
    '''
    [CN] 批量拉取日频 OHLCV（建议前复权），返回长表：index=date, columns=[code, open, high, low, close, volume, amount?, preclose?, paused?, high_limit?, low_limit?]。
    Returns a long-format DataFrame with one row per (date, code). Must include `date` & `code`.
    '''

def get_fundamentals_snapshot(date: pd.Timestamp, codes: list[str], lag_days: int) -> pd.DataFrame:
    '''
    [CN] 公告日感知 + 滞后快照：返回在 (−∞, date − lag_days] 已可得的最新财报口径字段。
    Returns a per-code snapshot of fundamental metrics available as of (date - lag_days).
    Required columns (at least): float_mktcap, book_value, net_income_ttm, revenue_ttm,
                                total_assets, total_equity, ocf_ttm, roe_ttm, roa_ttm, etc.
    '''

Assumptions:
- Timezone-naive dates interpreted in exchange local timezone.
- Codes conform to platform convention (e.g., "000001.SZ", "600000.SH").
"""
from __future__ import annotations

# file: src/api/myquant_io.py
# -*- coding: utf-8 -*-
"""
Unified adapter to gm.api (掘金GM) for CSI500·191 pipeline.

Public contract:
    - get_trade_days(start: str, end: str) -> list[pd.Timestamp]
    - get_index_members(index_code: str, date: pd.Timestamp) -> list[str]
    - get_ohlcv(codes: list[str], start: str, end: str, fq: str) -> pd.DataFrame
    - get_fundamentals_snapshot(date: pd.Timestamp, codes: list[str], lag_days: int) -> pd.DataFrame

Design notes:
- Token comes from config.GM_TOKEN (or env GM_TOKEN/MYQUANT_TOKEN as fallback).
- Symbols normalized to "EXCHANGE.SECID".
- OHLCV schema: ['date','code','open','high','low','close','volume','amount','preclose','paused','high_limit','low_limit'].
- Fundamentals snapshot uses fixed official field lists + deterministic mapping:
    * stk_get_daily_valuation_pt:
        fields="pe_ttm,pb_mrq,pb_lyr,ps_ttm,pcf_ttm_oper,pcf_ttm_ncf,dy_ttm,dy_lfy"
        -> pe_ttm, pb (pb_mrq→pb_lyr), ps_ttm, pcf_ttm (oper→ncf)
    * stk_get_daily_mktvalue_pt:
        fields="tot_mv,a_mv_ex_ltd,b_mv_ex_ltd,a_mv,b_mv"
        -> market_cap=tot_mv, float_mktcap=(a_mv_ex_ltd+b_mv_ex_ltd) or (a_mv+b_mv)
    * stk_get_finance_prime_pt:
        fields=("roe,roe_weight_avg,roe_cut,roe_weight_avg_cut,"
                "net_prof_pcom,net_prof_pcom_cut,net_prof,"
                "ttl_inc_oper,inc_oper,net_cf_oper,ttl_ast,"
                "ttl_inc_oper_yoy,inc_oper_yoy,net_prof_pcom_yoy")
        -> roe_ttm (prefer weight_avg), roa_ttm≈(net_profit/total_assets), net_profit_ttm, revenue_ttm, oper_cf_ttm
        (yoy stored as fallback)
    * stk_get_finance_deriv_pt:
        fields="net_prof_yoy,ttl_inc_oper_yoy,inc_oper_yoy"
        -> growth_np_yoy, growth_rev_yoy
- Percentage fields converted to fractions (÷100).
"""


import os
from typing import Optional

import pandas as pd

# ---------------- gm.api imports ----------------
try:
    from gm.api import (
        set_token,
        ADJUST_PREV,
        ADJUST_POST,
        ADJUST_NONE,
        history,
        get_trading_dates,
    )
    from gm.api import (  # type: ignore
        stk_get_index_constituents,
        stk_get_finance_prime_pt,
        stk_get_finance_deriv_pt,
        stk_get_daily_valuation_pt,
        stk_get_daily_mktvalue_pt,
    )
    _GM_AVAILABLE = True
except Exception:  # pragma: no cover
    _GM_AVAILABLE = False

# ---------------- logging helpers ----------------
try:
    # loop_progress(task: str, current: int, total: int)
    from src.utils.logging import step, loop_progress, done, warn, error
except Exception:
    def step(msg: str) -> None: print(f"[STEP] {msg}")  # pragma: no cover
    def loop_progress(task: str, current: int, total: int) -> None:  # pragma: no cover
        pct = 0 if total <= 0 else int(current * 100 / max(total, 1))
        print(f"[LOOP] {task} ({current}/{total}, {pct}%)")
    def done(msg: str = "done") -> None: print(f"[DONE] {msg}")  # pragma: no cover
    def warn(msg: str) -> None: print(f"[WARN] {msg}")  # pragma: no cover
    def error(msg: str) -> None: print(f"[ERROR] {msg}")  # pragma: no cover


# ---------------- token bootstrap ----------------
def _ensure_gm_token() -> None:
    if not _GM_AVAILABLE:
        warn("gm.api is not available. Functions will raise if called.")
        return
    token: Optional[str] = None
    try:
        from config import GM_TOKEN  # type: ignore
        token = GM_TOKEN
    except Exception:
        token = os.environ.get("GM_TOKEN") or os.environ.get("MYQUANT_TOKEN")
    if not token:
        warn("No GM token found. Set config.GM_TOKEN or env GM_TOKEN/MYQUANT_TOKEN.")
    else:
        try:
            set_token(str(token))
        except Exception as ex:  # pragma: no cover
            warn(f"set_token failed: {ex}")


_ensure_gm_token()


# ---------------- symbol & time helpers ----------------
_EX_PREFIXES = ("SHSE", "SZSE")


def _is_gm_symbol(code: str) -> bool:
    return code.upper().startswith(_EX_PREFIXES)


def _normalize_equity(code: str) -> str:
    """Accept '600000.SH'/'000001.SZ'/'SHSE.600000'/'SZSE.000001' and normalize to 'EX.secid'."""
    c = code.strip().upper()
    if _is_gm_symbol(c):
        return c
    if c.endswith(".SH"):
        return f"SHSE.{c[:-3]}"
    if c.endswith(".SZ"):
        return f"SZSE.{c[:-3]}"
    if len(c) == 6 and c.isdigit():
        ex = "SHSE" if c.startswith("6") else "SZSE"
        return f"{ex}.{c}"
    return c


_INDEX_ALIASES = {
    "CSI500": "SHSE.000905",
    "000905.SH": "SHSE.000905",
    "SHSE.000905": "SHSE.000905",
    "CSI300": "SHSE.000300",
    "000300.SH": "SHSE.000300",
    "SHSE.000300": "SHSE.000300",
}


def _normalize_index(index_code: str) -> str:
    c = index_code.strip().upper()
    if c in _INDEX_ALIASES:
        return _INDEX_ALIASES[c]
    if _is_gm_symbol(c):
        return c
    if c.endswith(".SH") and len(c) == 9:
        return f"SHSE.{c[:6]}"
    return c


def _map_adjust(fq: str):
    f = (fq or "pre").lower()
    if f in ("pre", "qfq", "adjust_prev"):
        return ADJUST_PREV
    if f in ("post", "hfq", "adjust_post"):
        return ADJUST_POST
    if f in ("none", "raw", "no"):
        return ADJUST_NONE
    warn(f"Unknown fq '{fq}', fallback to 'pre'.")
    return ADJUST_PREV


def _to_naive_series(x):
    s = pd.to_datetime(x, errors="coerce")
    if hasattr(s, "dt"):
        try:
            tz = getattr(s.dt, "tz", None)
            if tz is not None:
                try:
                    s = s.dt.tz_convert("Asia/Shanghai")
                except Exception:
                    s = s.dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
                s = s.dt.tz_localize(None)
        except Exception:
            s = s.dt.tz_localize(None)
    return s


def _pct_to_frac(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    try:
        return s.astype(float) / 100.0
    except Exception:
        return pd.to_numeric(s, errors="coerce") / 100.0


def _coalesce(*cols: Optional[pd.Series]) -> Optional[pd.Series]:
    """Return the first non-all-NA series among inputs; combine_first across them."""
    cur: Optional[pd.Series] = None
    for s in cols:
        if s is None:
            continue
        s = pd.to_numeric(s, errors="coerce")
        cur = s if cur is None else cur.combine_first(s)
    return cur


def _safe_div(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[pd.Series]:
    if num is None or den is None:
        return None
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    with pd.option_context("mode.use_inf_as_na", True):
        out = num.divide(den.replace(0, pd.NA))
    return out


def _is_all_na(x) -> bool:
    """Type-safe 'all missing' check for Series or scalar."""
    if x is None:
        return True
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.isna(x).all()
    return bool(pd.isna(x))


# =====================================================================================
# Public contract
# =====================================================================================

def get_trade_days(start: str, end: str) -> list[pd.Timestamp]:
    """Query trading dates between [start, end], inclusive (SHSE calendar)."""
    if not _GM_AVAILABLE:
        raise RuntimeError("gm.api not available")

    step(f"Query trading days [{start} -> {end}]")
    dates = get_trading_dates(exchange="SHSE", start_date=start, end_date=end)
    out: list[pd.Timestamp] = []
    for d in dates:
        ts = pd.Timestamp(d)
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        out.append(pd.Timestamp(ts.date()))
    done(f"Trading days = {len(out)}")
    return out


def get_index_members(index_code: str, date: pd.Timestamp) -> list[str]:
    """Index constituents on/ before `date`."""
    if not _GM_AVAILABLE:
        raise RuntimeError("gm.api not available")

    idx = _normalize_index(index_code)
    trade_date = pd.Timestamp(date).strftime("%Y-%m-%d")
    step(f"Fetch index members {idx} @ {trade_date}")
    try:
        df = stk_get_index_constituents(index=idx, trade_date=trade_date)
    except Exception as ex:
        error(f"stk_get_index_constituents failed: {ex}")
        raise

    if df is None or len(df) == 0:
        warn(f"No constituents for {idx} @ {trade_date}")
        return []

    syms = sorted(set(df["symbol"].astype(str)))
    done(f"Members = {len(syms)}")
    return syms


def get_ohlcv(codes: list[str], start: str, end: str, fq: str) -> pd.DataFrame:
    """Daily OHLCV via gm.api.history()."""
    if not _GM_AVAILABLE:
        raise RuntimeError("gm.api not available")
    if not codes:
        return pd.DataFrame(columns=[
            "date", "code", "open", "high", "low", "close", "volume", "amount",
            "preclose", "paused", "high_limit", "low_limit"
        ])

    adj = _map_adjust(fq)
    start_ts = f"{start} 09:00:00"
    end_ts = f"{end} 16:00:00"

    step(f"Fetch OHLCV {len(codes)} symbols [{start} -> {end}], fq={fq}")

    frames: list[pd.DataFrame] = []
    n = len(codes)
    for i, raw in enumerate(codes, 1):
        sym = _normalize_equity(raw)
        loop_progress("Fetch OHLCV", i, n)

        try:
            df = history(
                symbol=sym,
                frequency="1d",
                start_time=start_ts,
                end_time=end_ts,
                fields="eob,open,high,low,close,volume,amount",  # include eob explicitly
                adjust=adj,
                adjust_end_time=end,
                df=True,
            )
        except Exception as ex:
            warn(f"history failed for {sym}: {ex}")
            continue

        if df is None or len(df) == 0:
            continue
        if "date" not in df.columns:
            df = df.rename_axis("date").reset_index()
        df["date"] = _to_naive_series(df["date"]).dt.normalize()

        df = df.copy()
        if "eob" in df.columns:
            dt = _to_naive_series(df["eob"])
        elif "bob" in df.columns:
            dt = _to_naive_series(df["bob"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dt = _to_naive_series(df.index); dt.index = df.index
        else:
            # if first column is numeric, skip this symbol to avoid 1970 dates
            first_col = df.iloc[:, 0]
            if pd.api.types.is_numeric_dtype(first_col):
                warn(f"{sym}: missing datetime column; skip to avoid bad dates.")
                continue
            dt = _to_naive_series(first_col)

        df["date"] = _to_naive_series(df["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        df["code"] = sym

        keep = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
        df = df[[c for c in keep if c in df.columns]].sort_values("date")

        df["preclose"] = df["close"].shift(1)
        df["paused"] = (df["volume"] <= 0).astype(int)
        df["high_limit"] = pd.NA
        df["low_limit"] = pd.NA

        frames.append(df)

    if not frames:
        warn("No OHLCV fetched.")
    out = (pd.concat(frames, axis=0, ignore_index=True)
           if frames else pd.DataFrame(columns=[
               "date", "code", "open", "high", "low", "close", "volume", "amount",
               "preclose", "paused", "high_limit", "low_limit"]))
    done(f"OHLCV rows = {len(out)}")
    return out


def get_fundamentals_snapshot(date: pd.Timestamp, codes: list[str], lag_days: int) -> pd.DataFrame:
    """
    Publication-aware fundamentals snapshot.

    - Daily valuation/mktvalue use trade_date = date.
    - Finance (prime/deriv) use date = date - lag_days (publication-aware).
    """
    if not _GM_AVAILABLE:
        raise RuntimeError("gm.api not available")
    if not codes:
        return pd.DataFrame()

    trade_date = pd.Timestamp(date).strftime("%Y-%m-%d")
    lag_date = (pd.Timestamp(date) - pd.Timedelta(days=int(lag_days))).strftime("%Y-%m-%d")

    step(f"Fundamentals snapshot @ {trade_date} with LAG={lag_days}d (pub_date {lag_date})")
    symbols = [_normalize_equity(c) for c in codes]
    syms_arg = ",".join(symbols)

    pieces: list[pd.DataFrame] = []

    # --------- Valuation (trade date) ---------
    try:
        val = stk_get_daily_valuation_pt(
            symbols=syms_arg,
            trade_date=trade_date,
            fields="pe_ttm,pb_mrq,pb_lyr,ps_ttm,pcf_ttm_oper,pcf_ttm_ncf,dy_ttm,dy_lfy",
            df=True,
        )
    except Exception as ex:
        warn(f"stk_get_daily_valuation_pt failed: {ex}")
        val = None

    if val is not None and len(val) > 0:
        val = val.copy()
        pe_ttm = val["pe_ttm"] if "pe_ttm" in val.columns else None
        ps_ttm = val["ps_ttm"] if "ps_ttm" in val.columns else None
        pb = _coalesce(val.get("pb_mrq"), val.get("pb_lyr"))
        pcf = _coalesce(val.get("pcf_ttm_oper"), val.get("pcf_ttm_ncf"))
        out_val = pd.DataFrame({
            "symbol": val.get("symbol", val.get("sec_code", pd.NA)),
            "pe_ttm": pe_ttm,
            "pb": pb,
            "ps_ttm": ps_ttm,
            "pcf_ttm": pcf,
        })
        if "symbol" not in out_val.columns or out_val["symbol"].isna().all():
            out_val["symbol"] = val.get("symbol", pd.NA)
        pieces.append(out_val)

    # --------- Market value (trade date) ---------
    try:
        mv = stk_get_daily_mktvalue_pt(
            symbols=syms_arg,
            trade_date=trade_date,
            fields="tot_mv,a_mv_ex_ltd,b_mv_ex_ltd,a_mv,b_mv",
            df=True,
        )
    except Exception as ex:
        warn(f"stk_get_daily_mktvalue_pt failed: {ex}")
        mv = None

    if mv is not None and len(mv) > 0:
        mv = mv.copy()
        market_cap = mv.get("tot_mv")
        a_ex = mv.get("a_mv_ex_ltd"); b_ex = mv.get("b_mv_ex_ltd")
        a = mv.get("a_mv"); b = mv.get("b_mv")
        float_ex = None
        if a_ex is not None or b_ex is not None:
            float_ex = (a_ex.fillna(0) if a_ex is not None else 0) + (b_ex.fillna(0) if b_ex is not None else 0)
        float_inc = None
        if a is not None or b is not None:
            float_inc = (a.fillna(0) if a is not None else 0) + (b.fillna(0) if b is not None else 0)
        float_mktcap = float_ex if (float_ex is not None and not _is_all_na(float_ex)) else float_inc
        out_mv = pd.DataFrame({
            "symbol": mv.get("symbol", mv.get("sec_code", pd.NA)),
            "market_cap": market_cap,
            "float_mktcap": float_mktcap,
        })
        pieces.append(out_mv)

    # --------- Finance prime (lag date) ---------
    try:
        prime = stk_get_finance_prime_pt(
            symbols=syms_arg,
            date=lag_date,
            fields=("roe,roe_weight_avg,roe_cut,roe_weight_avg_cut,"
                    "net_prof_pcom,net_prof_pcom_cut,net_prof,"
                    "ttl_inc_oper,inc_oper,net_cf_oper,ttl_ast,"
                    "ttl_inc_oper_yoy,inc_oper_yoy,net_prof_pcom_yoy"),
            df=True,
        )
    except Exception as ex:
        warn(f"stk_get_finance_prime_pt failed: {ex}")
        prime = None

    fallback_growth_rev = None
    fallback_growth_np = None

    if prime is not None and len(prime) > 0:
        prime = prime.copy()
        # ROE (% → fraction)
        roe_ttm = _coalesce(prime.get("roe_weight_avg"), prime.get("roe"),
                            prime.get("roe_weight_avg_cut"), prime.get("roe_cut"))
        roe_ttm = _pct_to_frac(roe_ttm)

        # ROA ≈ net_profit / total_assets
        net_profit_any = _coalesce(prime.get("net_prof_pcom"), prime.get("net_prof"))
        roa_ttm = _safe_div(net_profit_any, prime.get("ttl_ast"))

        # Net profit & revenue & OCF
        net_profit_ttm = net_profit_any
        revenue_ttm = _coalesce(prime.get("ttl_inc_oper"), prime.get("inc_oper"))
        oper_cf_ttm = prime.get("net_cf_oper")

        out_prime = pd.DataFrame({
            "symbol": prime.get("symbol", prime.get("sec_code", pd.NA)),
            "roe_ttm": roe_ttm,
            "roa_ttm": roa_ttm,
            "net_profit_ttm": net_profit_ttm,
            "revenue_ttm": revenue_ttm,
            "oper_cf_ttm": oper_cf_ttm,
        })
        pieces.append(out_prime)

        # yoy fallback from prime
        fallback_growth_rev = _pct_to_frac(_coalesce(prime.get("ttl_inc_oper_yoy"), prime.get("inc_oper_yoy")))
        fallback_growth_np = _pct_to_frac(prime.get("net_prof_pcom_yoy"))

    # --------- Finance derived (lag date) ---------
    try:
        deriv = stk_get_finance_deriv_pt(
            symbols=syms_arg,
            date=lag_date,
            fields="net_prof_yoy,ttl_inc_oper_yoy,inc_oper_yoy",
            df=True,
        )
    except Exception as ex:
        warn(f"stk_get_finance_deriv_pt failed: {ex}")
        deriv = None

    if deriv is not None and len(deriv) > 0:
        deriv = deriv.copy()
        growth_np_yoy = _pct_to_frac(deriv.get("net_prof_yoy"))
        growth_rev_yoy = _pct_to_frac(_coalesce(deriv.get("ttl_inc_oper_yoy"), deriv.get("inc_oper_yoy")))
        out_deriv = pd.DataFrame({
            "symbol": deriv.get("symbol", deriv.get("sec_code", pd.NA)),
            "growth_np_yoy": growth_np_yoy,
            "growth_rev_yoy": growth_rev_yoy,
        })
        pieces.append(out_deriv)
    else:
        if fallback_growth_rev is not None or fallback_growth_np is not None:
            out_yoy = pd.DataFrame({
                "symbol": (prime.get("symbol", prime.get("sec_code", pd.NA)) if prime is not None else pd.NA),
                "growth_np_yoy": fallback_growth_np,
                "growth_rev_yoy": fallback_growth_rev,
            })
            pieces.append(out_yoy)

    # --------- merge & finalize ---------
    if not pieces:
        warn("No fundamentals fields returned; snapshot is empty.")
        return pd.DataFrame(index=pd.Index(symbols, name="code"))

    base = None
    for seg in pieces:
        base = seg if base is None else pd.merge(base, seg, on="symbol", how="outer")

    base = base.rename(columns={"symbol": "code"})
    base["code"] = base["code"].astype(str)
    base = base.drop_duplicates(subset=["code"]).set_index("code")

    preferred = [
        "market_cap", "float_mktcap",
        "pb", "pe_ttm", "ps_ttm", "pcf_ttm",
        "roe_ttm", "roa_ttm", "net_profit_ttm", "revenue_ttm", "oper_cf_ttm",
        "growth_np_yoy", "growth_rev_yoy",
    ]
    ordered = [c for c in preferred if c in base.columns] + [c for c in base.columns if c not in preferred]
    base = base[ordered]

    done(f"Snapshot fields={list(base.columns)}, names={len(base)}")
    return base

# file: src/api/myquant_io.py  （追加在文件末尾）

def get_index_weights_ffmc_monthly(index_code: str,
                                   date: pd.Timestamp,
                                   backfill_months: int = 12) -> pd.Series:
    """
    用“自由流通市值”在**月初第一个交易日**计算并缓存月度基准权重。

    缓存位置：
        data/ref/index_weights/{<规范化指数代码>}/{YYYYMM}.csv
    CSV列： code, weight  （code形如 '600519.SH'；同月内多次调用命中缓存）

    行为：
    - 当月文件存在：直接读缓存并返回；
    - 不存在：计算当月；若之前月份也缺，最多**回补 backfill_months 个月**；
    - 权重 = float_mktcap / sum ；若 float_mktcap 缺失则用 market_cap；再不行退化为等权。

    参数
    ----
    index_code : 'SHSE.000905' / '000905.SH' / 'CSI500' 等
    date       : 属于目标月份的任意日期
    backfill_months : 缺月回补的最大月数（默认12）

    返回
    ----
    pd.Series(index='600519.SH', values=权重，和为1)
    """
    from config import load_config  # 延迟导入，避免循环依赖
    from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic

    CFG = load_config()
    P = CFG.paths

    # GM→行情代码（'SHSE.600519' -> '600519.SH'）
    def _gm_to_market(s: str) -> str:
        s = str(s).upper()
        if s.startswith("SHSE."):
            return s.split(".", 1)[1] + ".SH"
        if s.startswith("SZSE."):
            return s.split(".", 1)[1] + ".SZ"
        return s

    idx_norm = _normalize_index(index_code)
    t = pd.Timestamp(date).normalize()
    ym = t.strftime("%Y%m")

    base_dir = P.ref_dir / "index_weights" / idx_norm
    ensure_dir(base_dir)

    def _month_path(ym_: str):
        return base_dir / f"{ym_}.csv"

    # 1) 缓存命中
    cur_path = _month_path(ym)
    cached = read_csv_safe(str(cur_path), parse_dates=None, default=None)
    if cached is not None and not cached.empty and {"code", "weight"} <= set(cached.columns):
        s = pd.to_numeric(cached.set_index("code")["weight"], errors="coerce").fillna(0.0)
        s = s[s > 0]
        s = s / s.sum() if s.sum() > 0 else s
        done(f"[BENCH-FFMC] cache hit {idx_norm} {ym}: names={len(s)}, sum={float(s.sum()):.4f}")
        return s

    # 2) 需要计算的月份（含当月；遇到已有文件就停止回溯）
    months_to_compute: list[str] = []
    cur = pd.Timestamp(t.year, t.month, 1)
    for _ in range(max(1, int(backfill_months))):
        ym_i = cur.strftime("%Y%m")
        if not _month_path(ym_i).exists():
            months_to_compute.append(ym_i)
        else:
            break
        cur = (cur - pd.offsets.MonthBegin(1)).normalize()
    months_to_compute = list(reversed(months_to_compute))  # 旧→新

    for ym_i in months_to_compute:
        first_day = pd.Timestamp(ym_i + "01")
        last_day = (first_day + pd.offsets.MonthEnd(0)).normalize()

        # 当月第一个交易日
        tdays = get_trade_days(first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d"))
        if not tdays:
            warn(f"[BENCH-FFMC] no trading day for {idx_norm} month {ym_i}; skip.")
            continue
        m0 = tdays[0]

        # 成分（按月初那天）
        try:
            members = get_index_members(idx_norm, m0)
        except Exception as ex:
            warn(f"[BENCH-FFMC] get_index_members failed {idx_norm} @ {m0}: {ex}")
            members = []

        if not members:
            warn(f"[BENCH-FFMC] empty members {idx_norm} @ {m0}; skip month {ym_i}.")
            continue

        # 市值快照（lag=0，用日级市值）
        try:
            snap = get_fundamentals_snapshot(m0, codes=members, lag_days=0)
        except Exception as ex:
            warn(f"[BENCH-FFMC] fundamentals snapshot failed @ {m0}: {ex}")
            snap = pd.DataFrame(index=pd.Index([_gm_to_market(c) for c in members], name="code"))

        if snap is None or snap.empty:
            # 等权兜底
            codes = pd.Index([_gm_to_market(c) for c in members], name="code")
            ser = pd.Series(1.0 / len(codes), index=codes, name="weight")
        else:
            df = snap.copy()
            # 索引转为 '600519.SH'
            if df.index.name is None:
                df.index = pd.Index([_gm_to_market(x) for x in df.index], name="code")
            else:
                df.index = df.index.map(_gm_to_market)

            ff = df.get("float_mktcap")
            if ff is None or ff.isna().all():
                ff = df.get("market_cap")
            if ff is None or ff.isna().all():
                codes = df.index
                ser = pd.Series(1.0 / len(codes), index=codes, name="weight")
            else:
                w = pd.to_numeric(ff, errors="coerce").fillna(0.0).clip(lower=0.0)
                ser = (w / float(w.sum())) if float(w.sum()) > 0 else pd.Series(1.0 / len(w), index=df.index)
                ser = ser.rename("weight")
                ser.index.name = "code"

        # 写入月度缓存
        write_csv_atomic(str(_month_path(ym_i)), ser.reset_index(), index=False)
        done(f"[BENCH-FFMC] cache write {idx_norm} {ym_i}: names={len(ser)}, sum={float(ser.sum()):.4f}")

    # 3) 返回当月
    cached = read_csv_safe(str(cur_path), parse_dates=None, default=None)
    if cached is None or cached.empty:
        warn(f"[BENCH-FFMC] cache miss after compute {idx_norm} {ym}; returning empty.")
        return pd.Series(dtype=float)
    s = pd.to_numeric(cached.set_index("code")["weight"], errors="coerce").fillna(0.0)
    s = s[s > 0]
    s = s / s.sum() if s.sum() > 0 else s
    return s


def get_index_weights(index_code: str, date: pd.Timestamp) -> pd.Series:
    """
    对外包装：返回`date`所在月份的**自由流通市值加权**基准权重（带缓存）。
    以后若接入官方“历史成分权重”，可以在此先尝试官方数据，再回退到本函数。
    """
    return get_index_weights_ffmc_monthly(index_code=index_code, date=date, backfill_months=12)
