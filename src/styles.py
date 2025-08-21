"""
[CN] 风格因子契约：用财报快照 + 价格代理生成稳定的风格暴露，公告日感知 + 滞后。
[Purpose] Style exposures for cross-section regression RHS.

Signature:
def style_exposures(date: pd.Timestamp, codes: list[str],
                    panel: pd.DataFrame, funda_snapshot: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code with stable column set, e.g.:
    ["size","bp","ep_ttm","sp_ttm","growth_rev_yoy","growth_np_yoy",
     "leverage","roe_ttm","roa_ttm","cf_yield", ...]
    Cleaning: winsorize(WINSOR_PCT) + z-score if STANDARDIZE.
    Missing: small gaps may fall back to price-based proxies; must WARN in logs.
    '''

Notes:
- size = ln(float_mktcap); bp = book_value/market_cap; etc.
- Forward-fill daily between announcement snapshots; lag controlled by STYLE_LAG_DAYS.
"""
from __future__ import annotations
from __future__ import annotations

# file: src/styles.py
# -*- coding: utf-8 -*-
"""
Styles exposures for CSI500·191 pipeline.

本文件职责：
- 在交易日 t、股票集合 codes 上，计算风格小类截面，并按 groups.yaml 聚合到大类；
- 统一执行 winsorize(1%) + z-score，可选按行业内标准化；
- 支持热插拔 YAML（subfactors.yaml / groups.yaml / weights.yaml）；
- 引入外部一致预期 EPS（FORECAST_EPS）到 EPIBS：机构内≤t最新、机构间等权；缺失回退到模型估计；
- 输出覆盖率/关键日志，SMOKE 自检。

对外 API（保持稳定）：
style_exposures(date, codes, panel, funda_snapshot,
                *, by_industry: bool=False, industry: pd.Series|None=None,
                include_subfactors: bool=False,
                beta_market: str="000300.SH", beta_lookback: int=250, beta_halflife: int=60)
    -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 项目内依赖（若相对导入不可用，则回退到绝对导入）
# ---------------------------------------------------------------------
try:
    if __package__:
        from .utils import logging as log
        from .utils.fileio import ensure_dir, write_csv_atomic
        from .utils.numerics import clip_inf_nan
        from .trading_calendar import last_n_trading_days
        from .api.myquant_io import get_ohlcv, get_fundamentals_snapshot
        from config import CFG, REF_DIR, FQ, LOG_VERBOSITY
    else:
        raise ImportError
except Exception:
    from src.utils import logging as log
    from src.utils.fileio import ensure_dir, write_csv_atomic
    from src.utils.numerics import clip_inf_nan
    from src.trading_calendar import last_n_trading_days
    from src.api.myquant_io import get_ohlcv, get_fundamentals_snapshot
    from config import CFG, REF_DIR, FQ, LOG_VERBOSITY

# ==========================
# 基础配置与通用小工具
# ==========================
WINSOR_PCT = 0.01
ZSCORE_DDOF = 0

DEFAULT_GROUPS: Dict[str, List[str]] = {
    "momentum": ["rstr_500_skip21_hl120"],
    "volatility": ["dastd_250_hl40", "cmra_12m", "hsigma_250", "beta_252_hl60"],
    "size": ["lncap"],
    "earnings_yield": ["etop", "cetop"],  # epibs 可同时并入或单独作为 earnings 的一员
    "value": ["btop", "sp_ttm", "cf_yield", "etop"],  # 拓展口径：sp_ttm=revenue/mktcap，cf_yield=cetop
    "growth": ["sgro", "egro", "egib", "egib_s", "growth_rev_yoy", "growth_np_yoy"],
    "leverage": ["mlev", "dtoa", "blev"],
    "liquidity": ["stom_21", "stoq_3m", "stoa_12m"],
    "beta": ["beta_252_hl60"]  # CAPM β
}

# ---------- Logging 便利 ----------
try:
    log.set_verbosity(LOG_VERBOSITY)
except Exception:
    pass

def _now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- YAML 读取 ----------
def _read_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return d if isinstance(d, dict) else {}
    except Exception as e:
        log.warn(f"[STYLE] read YAML fail {path}: {e}")
        return {}

def _load_configs() -> Tuple[dict, dict, dict]:
    """返回 (subfactors_cfg, groups_cfg, weights_cfg)；缺失返回空 dict"""
    style_dir = Path(REF_DIR) / "style"
    return (
        _read_yaml(style_dir / "subfactors.yaml"),
        _read_yaml(style_dir / "groups.yaml"),
        _read_yaml(style_dir / "weights.yaml"),
    )

# ---------- DataFrame 透视 ----------
def _pivot_wide(panel: pd.DataFrame, value: str, safe: bool = False) -> pd.DataFrame:
    """长表→宽表：index=date, columns=code, values=value"""
    need = {"date", "code", value}
    if not need.issubset(panel.columns):
        if safe:
            return pd.DataFrame()
        missing = need - set(panel.columns)
        raise KeyError(f"_pivot_wide: panel missing columns {missing}")
    df = panel[["date", "code", value]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "code"])
    wide = df.pivot(index="date", columns="code", values=value).sort_index()
    return wide

def _safe_div(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
        if isinstance(out, (pd.Series, pd.DataFrame)):
            out = out.replace([np.inf, -np.inf], np.nan)
        return out

# ---------- winsor + zscore ----------
def _winsorize(s: pd.Series, lower: float = WINSOR_PCT, upper: float = 1 - WINSOR_PCT) -> pd.Series:
    if s.dropna().empty:
        return s
    ql, qu = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def _zscore(s: pd.Series, ddof: int = ZSCORE_DDOF) -> pd.Series:
    if s.dropna().empty:
        return s
    mu = s.mean()
    sd = s.std(ddof=ddof)
    return (s - mu) / sd if sd and not np.isclose(sd, 0.0) else s - mu

def standardize_exposures(df: pd.DataFrame, winsor: Tuple[float, float] = (WINSOR_PCT, 1 - WINSOR_PCT),
                          by: Optional[pd.Series] = None, ddof: int = ZSCORE_DDOF) -> pd.DataFrame:
    """对截面做 winsor+zscore；若 by 给出行业，则行业内标准化"""
    if df.empty:
        return df
    if by is None:
        out = df.apply(lambda col: _zscore(_winsorize(col, winsor[0], winsor[1]), ddof=ddof))
        return out
    # 行业内
    out_list = []
    for g, idx in by.dropna().groupby(by).groups.items():
        sub = df.reindex(index=idx)
        sub = sub.apply(lambda col: _zscore(_winsorize(col, winsor[0], winsor[1]), ddof=ddof))
        out_list.append(sub)
    rest_idx = df.index.difference(pd.Index.union_many([x.index for x in out_list]) if out_list else df.index[:0])
    if len(rest_idx):
        rest = df.reindex(index=rest_idx).apply(lambda col: _zscore(_winsorize(col, winsor[0], winsor[1]), ddof=ddof))
        out_list.append(rest)
    out = pd.concat(out_list).reindex(df.index)
    return out

# ---------- 覆盖率打印 ----------
def _print_coverage(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        log.warn(f"[STYLE] {name}: empty")
        return
    cov = df.notna().mean()
    cov_str = ", ".join([f"{k}={v:.2f}" for k, v in cov.items()])
    log.step(f"[STYLE] coverage {name}: {cov_str}")

# ---------- 代码规范化 ----------
def _norm_code(x: str) -> str:
    if not isinstance(x, str):
        return x
    s = x.strip().upper().replace("_", ".")
    if s.endswith(".SH") or s.endswith(".SZ"):
        a, b = s.split(".")
        return f"{a.zfill(6)}.{b}"
    if s.startswith("SH") and len(s) == 8 and s[2:].isdigit():
        return f"{s[2:]}.SH"
    if s.startswith("SZ") and len(s) == 8 and s[2:].isdigit():
        return f"{s[2:]}.SZ"
    if s.isdigit() and len(s) == 6:
        return f"{s}.SH" if s[0] == "6" else f"{s}.SZ"
    return s

# ==========================
# SubfactorContext
# ==========================
@dataclass
class SubfactorContext:
    date: pd.Timestamp
    codes: List[str]
    panel: pd.DataFrame
    funda: pd.DataFrame

# ==========================
# 外部一致预期 EPS 接入（EPIBS）
# ==========================
def _resolve_forecast_cols(df: pd.DataFrame, user_map: Optional[dict] = None) -> dict:
    cand = {c.lower(): c for c in df.columns}
    def pick(*alts):
        for a in alts:
            if a in cand:
                return cand[a]
        return None
    m = {"code": None, "name": None, "date": None, "agency": None, "eps": None}
    if user_map:
        for k, v in user_map.items():
            if k in m and v in df.columns:
                m[k] = v
    m["code"]   = m["code"]   or pick("stock_code", "code", "ticker", "symbol")
    m["name"]   = m["name"]   or pick("stock_name", "name")
    m["date"]   = m["date"]   or pick("entrytime", "forecast_date", "updatetime", "update_time", "time", "tmstamp", "ts")
    m["agency"] = m["agency"] or pick("researchinstitute", "research_institute", "agency", "organ_name", "broker")
    m["eps"]    = m["eps"]    or pick("forecast_eps", "est_eps", "eps", "forecast_ep")
    return m

def _discover_forecast_csv() -> Optional[str]:
    """ENV -> CFG -> data/raw/forcast/最新 -> 会话中硬路径（若存在）"""
    p = os.getenv("STYLE_FORECAST_CSV")
    if p and Path(p).exists():
        return p
    try:
        cfg_p = getattr(getattr(CFG, "paths", object()), "forecast_csv", None)
        if isinstance(cfg_p, str) and Path(cfg_p).exists():
            return cfg_p
    except Exception:
        pass
    base = Path(REF_DIR).parent / "raw" / "forcast"  # data/raw/forcast/
    if base.exists():
        cands = sorted([x for x in base.iterdir() if x.suffix.lower() == ".csv"],
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if cands:
            return str(cands[0])
    # 会话中你给的硬路径（若存在则用）
    hard = r"D:\Code\R\quant-leiying\task5\191multi-factor reproduce\data\raw\forcast\RPT_FORECAST_STK_202508191216.csv"
    return hard if Path(hard).exists() else None

def _forecast_eps_for_t(codes: List[str], t: pd.Timestamp,
                        csv_path: Optional[str], cols_map: Optional[dict] = None) -> pd.Series:
    """对 t 日形成每个 code 的 forecast_eps_t（机构内≤t最新、机构间等权）；取不到→NaN"""
    idx = pd.Index([_norm_code(c) for c in codes], name="code")
    if not csv_path or not Path(csv_path).exists():
        return pd.Series(np.nan, index=idx, name="forecast_eps")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        log.warn(f"[STYLE][EPIBS] read csv failed: {csv_path} :: {e}")
        return pd.Series(np.nan, index=idx, name="forecast_eps")
    cols = _resolve_forecast_cols(df, cols_map)
    need = ("code", "date", "eps")
    if any(cols[k] is None for k in need):
        log.warn(f"[STYLE][EPIBS] missing columns; need code/date/eps; map={cols}")
        return pd.Series(np.nan, index=idx, name="forecast_eps")
    use_cols = [cols["code"], cols["date"], cols["eps"]] + ([cols["agency"]] if cols["agency"] else [])
    sub = df[use_cols].rename(columns={
        cols["code"]: "code", cols["date"]: "dt", cols["eps"]: "eps",
        **({cols["agency"]: "agency"} if cols["agency"] else {})
    })
    sub["code"] = sub["code"].astype(str).map(_norm_code)
    sub = sub[sub["code"].isin(idx)]
    sub["dt"] = pd.to_datetime(sub["dt"], errors="coerce")
    sub = sub[(sub["dt"].notna()) & (sub["dt"] <= pd.Timestamp(t).normalize())]  # 严禁前视
    sub["eps"] = pd.to_numeric(sub["eps"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=["eps"])
    if sub.empty:
        return pd.Series(np.nan, index=idx, name="forecast_eps")
    if "agency" not in sub.columns:
        sub["agency"] = "__one__"
    latest = sub.sort_values(["code", "agency", "dt"]).groupby(["code", "agency"], as_index=False).tail(1)
    mean_eps = latest.groupby("code")["eps"].mean()
    return mean_eps.reindex(idx).rename("forecast_eps")

def _compute_epibs(ctx: SubfactorContext, industry: Optional[pd.Series] = None) -> pd.Series:
    """
    EPIBS：外部 forecast_eps/price_t 优先；缺失→回退模型估算（ttm eps × 增长 proxy）。
    打印覆盖率：used_forecast / used_fallback / none。
    """
    codes = list(ctx.codes)
    t = pd.Timestamp(ctx.date).normalize()

    # 当日收盘（若缺列返回空）
    px_w = _pivot_wide(ctx.panel, "close", safe=True).reindex(columns=codes).ffill()
    px_t = px_w.iloc[-1] if not px_w.empty else pd.Series(index=codes, dtype=float)

    # 1) 外部 forecast_eps
    csv_path = _discover_forecast_csv()
    if csv_path:
        log.step(f"[STYLE][EPIBS] CSV resolved: {csv_path}")
    s_forecast = _forecast_eps_for_t(codes, t, csv_path, cols_map=None)
    epibs_forecast = s_forecast.divide(pd.Series(px_t).reindex(s_forecast.index))

    # 2) 回退模型：eps_ttm × 增长 proxy
    f = ctx.funda
    net_profit_ttm = pd.to_numeric(f.get("net_profit_ttm"), errors="coerce").reindex(codes) if "net_profit_ttm" in f.columns else pd.Series(index=codes, dtype=float)
    total_shares   = pd.to_numeric(f.get("total_shares"),   errors="coerce").reindex(codes) if "total_shares"   in f.columns else pd.Series(index=codes, dtype=float)
    float_shares   = pd.to_numeric(f.get("float_shares"),   errors="coerce").reindex(codes) if "float_shares"   in f.columns else pd.Series(index=codes, dtype=float)
    market_cap     = pd.to_numeric(f.get("market_cap"),     errors="coerce").reindex(codes) if "market_cap"     in f.columns else pd.Series(index=codes, dtype=float)
    px_vec         = pd.Series(px_t).reindex(codes)

    shares_hat = total_shares.where(total_shares.notna(), float_shares)
    shares_hat = shares_hat.where(shares_hat.notna(), _safe_div(market_cap, px_vec))
    eps_ttm    = _safe_div(net_profit_ttm, shares_hat)

    g_rev = pd.to_numeric(f.get("growth_rev_yoy"), errors="coerce").reindex(codes) if "growth_rev_yoy" in f.columns else pd.Series(index=codes, dtype=float)
    g_np  = pd.to_numeric(f.get("growth_np_yoy"),  errors="coerce").reindex(codes) if "growth_np_yoy"  in f.columns else pd.Series(index=codes, dtype=float)
    g_hat = pd.concat([g_rev, g_np], axis=1).mean(axis=1).clip(-0.4, 0.4)

    est_eps_hat    = eps_ttm * (1.0 + g_hat)
    epibs_fallback = _safe_div(est_eps_hat, px_vec).rename("epibs_fallback")

    # 组合
    use_forecast = epibs_forecast.notna()
    epibs = epibs_forecast.where(use_forecast, epibs_fallback).rename("epibs")

    used_forecast = float(use_forecast.mean())
    used_fallback = float((~use_forecast & epibs_fallback.notna()).mean())
    used_none     = float(epibs.isna().mean())
    log.step(f"[STYLE][EPIBS] used_forecast={used_forecast:.2f}, used_fallback={used_fallback:.2f}, none={used_none:.2f}")

    return epibs

# ==========================
# 价量与回归类小因子
# ==========================
def _halflife_weights(n: int, halflife: int) -> np.ndarray:
    """指数半衰期权重，长度 n，最后一个点权重最大"""
    if n <= 0:
        return np.zeros(0)
    lam = math.log(2.0) / max(1, halflife)
    t = np.arange(n, dtype=float)
    w = np.exp(lam * (t - (n - 1)))
    w /= w.sum() if w.sum() != 0 else 1.0
    return w

def _daily_return(wide_close: pd.DataFrame) -> pd.DataFrame:
    return wide_close.pct_change().replace([np.inf, -np.inf], np.nan)

def _compute_beta_hsigma(ctx: SubfactorContext, beta_market: str, lookback: int, halflife: int) -> Tuple[pd.Series, pd.Series]:
    """CAPM β（带截距，指数加权），残差加权 std 作为 hsigma"""
    codes = list(ctx.codes)
    close_w = _pivot_wide(ctx.panel, "close", safe=True).reindex(columns=codes).ffill()
    if close_w.empty or len(close_w) < lookback:
        return (pd.Series(np.nan, index=codes, name="beta_252_hl60"),
                pd.Series(np.nan, index=codes, name="hsigma_250"))

    # 市场指数：若取不到，就用等权组合近似
    try:
        mkt = _pivot_wide(ctx.panel, f"close_{beta_market}", safe=True)
        if mkt.empty:
            raise KeyError
        mkt_close = mkt.iloc[:, 0].ffill()
    except Exception:
        mkt_close = close_w.mean(axis=1)

    ret = _daily_return(close_w).iloc[-lookback:]
    mret = _daily_return(mkt_close.to_frame("m")).iloc[-lookback:, 0]
    w = _halflife_weights(len(ret), halflife)
    X = np.vstack([np.ones_like(mret.values), mret.values]).T  # 截距 + 市场收益
    WX = X * w[:, None]
    betas = []
    hs = []
    for c in ret.columns:
        y = ret[c].values
        Wy = y * w
        # Ridge ~ 近似 OLS，稳定求解
        XtX = WX.T @ WX + 1e-6 * np.eye(2)
        Xty = WX.T @ Wy
        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coef = np.array([np.nan, np.nan])
        beta = coef[1]
        resid = y - X @ coef
        hsigma = np.sqrt(np.nanmean((resid ** 2) * w))
        betas.append(beta)
        hs.append(hsigma)
    beta_s = pd.Series(betas, index=ret.columns, name="beta_252_hl60")
    hsigma_s = pd.Series(hs, index=ret.columns, name="hsigma_250")
    return beta_s, hsigma_s

def _compute_rstr(ctx: SubfactorContext, lookback: int, skip: int, halflife: int) -> pd.Series:
    """RSTR: 跳过近期 skip 天，窗口 lookback，半衰 halflife，对 ln(1+r) 加权求和"""
    codes = ctx.codes
    close_w = _pivot_wide(ctx.panel, "close", safe=True).reindex(columns=codes).ffill()
    ret = _daily_return(close_w)
    if ret.empty or len(ret) < (skip + 10):
        return pd.Series(np.nan, index=codes, name="rstr_500_skip21_hl120")
    r = np.log1p(ret)
    r = r.iloc[-lookback:-skip] if lookback > skip else r.iloc[:-skip]
    w = _halflife_weights(len(r), halflife)
    val = np.nansum(r.values * w[:, None], axis=0)
    out = pd.Series(val, index=r.columns, name="rstr_500_skip21_hl120")
    return out.reindex(codes)

def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    m = np.nansum(x * w) / (np.nansum(w) if np.nansum(w) else 1.0)
    v = np.nansum(w * (x - m) ** 2) / (np.nansum(w) if np.nansum(w) else 1.0)
    return float(np.sqrt(v))

def _compute_dastd(ctx: SubfactorContext, lookback: int, halflife: int) -> pd.Series:
    """DASTD：加权日收益标准差"""
    codes = ctx.codes
    close_w = _pivot_wide(ctx.panel, "close", safe=True).reindex(columns=codes).ffill()
    ret = _daily_return(close_w).iloc[-lookback:]
    if ret.empty:
        return pd.Series(np.nan, index=codes, name="dastd_250_hl40")
    w = _halflife_weights(len(ret), halflife)
    vals = []
    for c in ret.columns:
        vals.append(_weighted_std(ret[c].values, w))
    return pd.Series(vals, index=ret.columns, name="dastd_250_hl40").reindex(codes)

def _compute_cmra(ctx: SubfactorContext) -> pd.Series:
    """CMRA：过去12个月的 ln(1+月收益) 路径的 max-min"""
    codes = ctx.codes
    close_w = _pivot_wide(ctx.panel, "close", safe=True).reindex(columns=codes).ffill()
    if close_w.empty or len(close_w) < 252 * 0.7:
        return pd.Series(np.nan, index=codes, name="cmra_12m")
    # 月度价格：用月末收盘
    mclose = close_w.resample("M").last()
    mret = mclose.pct_change().dropna(how="all")
    z = np.log1p(mret).cumsum()
    cmra = z.rolling(12).apply(lambda a: np.nanmax(a) - np.nanmin(a), raw=True)
    out = cmra.iloc[-1].rename("cmra_12m")
    return out.reindex(codes)

# ==========================
# 财报/快照类小因子
# ==========================
def _hydrate_funda(date: pd.Timestamp, codes: List[str], funda_snapshot: pd.DataFrame) -> pd.DataFrame:
    """将 myquant_io.get_fundamentals_snapshot(<=t-LAG) 规范到 code 索引，补必要列"""
    df = funda_snapshot.copy()
    if "code" in df.columns:
        df = df.set_index("code")
    df.index = df.index.astype(str)
    # 补必要列
    need = ["market_cap","float_mktcap","pb","net_profit_ttm","revenue_ttm","oper_cf_ttm",
            "total_assets","total_debt","long_debt","roe_ttm","roa_ttm",
            "growth_rev_yoy","growth_np_yoy","float_shares","total_shares","book_equity"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    # book_equity 兜底：market_cap/pb
    if df["book_equity"].isna().all() and "pb" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["book_equity"] = pd.to_numeric(df["market_cap"], errors="coerce") / \
                                pd.to_numeric(df["pb"], errors="coerce").replace(0, np.nan)
    out = df.reindex(codes)
    cov = out.notna().mean()
    top = ", ".join([f"{k}={v:.2f}" for k, v in cov.sort_values(ascending=False).head(10).items()])
    log.step(f"[STYLE] funda coverage {top}")
    return out

def _compute_lncap(ctx: SubfactorContext) -> pd.Series:
    f = ctx.funda
    s = pd.to_numeric(f.get("float_mktcap"), errors="coerce")
    s = s.where(s.notna(), pd.to_numeric(f.get("market_cap"), errors="coerce"))
    out = np.log(s.replace(0, np.nan))
    return out.rename("lncap").reindex(ctx.codes)

def _compute_value_ratios(ctx: SubfactorContext) -> Dict[str, pd.Series]:
    f = ctx.funda.reindex(ctx.codes)
    mktcap = pd.to_numeric(f["market_cap"], errors="coerce")
    netinc  = pd.to_numeric(f.get("net_profit_ttm"), errors="coerce")
    ocf     = pd.to_numeric(f.get("oper_cf_ttm"), errors="coerce")
    revenue = pd.to_numeric(f.get("revenue_ttm"), errors="coerce")
    pb      = pd.to_numeric(f.get("pb"), errors="coerce")
    be      = pd.to_numeric(f.get("book_equity"), errors="coerce")

    etop = _safe_div(netinc, mktcap).rename("etop")
    cetop = _safe_div(ocf, mktcap).rename("cetop")
    sp_ttm = _safe_div(revenue, mktcap).rename("sp_ttm")
    btop = _safe_div(be, mktcap)
    btop = btop.where(btop.notna(), (1.0 / pb.replace(0, np.nan)))
    btop = btop.rename("btop")

    out = {"etop": etop, "cetop": cetop, "btop": btop, "sp_ttm": sp_ttm, "cf_yield": cetop}
    return {k: v.reindex(ctx.codes) for k, v in out.items()}

def _compute_growth_leverage_profitability(ctx: SubfactorContext) -> Dict[str, pd.Series]:
    f = ctx.funda.reindex(ctx.codes)
    # growth proxies
    sgro = pd.to_numeric(f.get("growth_rev_yoy"), errors="coerce").rename("sgro")
    egro = pd.to_numeric(f.get("growth_np_yoy"), errors="coerce").rename("egro")
    egib = pd.Series(np.nan, index=f.index, name="egib")   # 外部一致预期三年/一年增长，如无置 NaN
    egib_s = pd.Series(np.nan, index=f.index, name="egib_s")

    # leverage
    mlev = _safe_div(pd.to_numeric(f.get("market_cap"), errors="coerce")
                     + pd.to_numeric(f.get("long_debt"), errors="coerce"),
                     pd.to_numeric(f.get("market_cap"), errors="coerce")).rename("mlev")
    dtoa = _safe_div(pd.to_numeric(f.get("total_debt"), errors="coerce"),
                     pd.to_numeric(f.get("total_assets"), errors="coerce")).rename("dtoa")
    blev = _safe_div(pd.to_numeric(f.get("book_equity"), errors="coerce")
                     + pd.to_numeric(f.get("long_debt"), errors="coerce"),
                     pd.to_numeric(f.get("book_equity"), errors="coerce")).rename("blev")


    out = {
        "sgro": sgro, "egro": egro, "egib": egib, "egib_s": egib_s,
        "mlev": mlev, "dtoa": dtoa, "blev": blev
    }
    return {k: v.reindex(ctx.codes) for k, v in out.items()}

def _compute_stom_stoq_stoa(ctx: SubfactorContext) -> Dict[str, pd.Series]:
    """STOM/STOQ/STOA：基于 volume/float_shares；若无 float_shares 返回 NaN"""
    codes = ctx.codes
    vol_w = _pivot_wide(ctx.panel, "volume", safe=True).reindex(columns=codes).ffill()
    if vol_w.empty:
        return {"stom_21": pd.Series(np.nan, index=codes),
                "stoq_3m": pd.Series(np.nan, index=codes),
                "stoa_12m": pd.Series(np.nan, index=codes)}
    fshares = pd.to_numeric(ctx.funda.get("float_shares"), errors="coerce").reindex(codes)
    fshares = fshares.replace(0, np.nan)

    # 21D
    stom = np.log((vol_w.iloc[-21:].sum() / fshares).replace(0, np.nan)).rename("stom_21")

    # 月度 STOM
    stom_daily = np.log((vol_w / fshares).replace(0, np.nan))
    stom_m = stom_daily.resample("M").sum()
    stoq = np.log(np.exp(stom_m).rolling(3).mean()).iloc[-1].rename("stoq_3m")
    stoa = np.log(np.exp(stom_m).rolling(12).mean()).iloc[-1].rename("stoa_12m")

    return {
        "stom_21": stom.reindex(codes),
        "stoq_3m": stoq.reindex(codes),
        "stoa_12m": stoa.reindex(codes)
    }

# ==========================
# 分组聚合
# ==========================
def _aggregate_groups(sub_z: pd.DataFrame, groups_cfg: dict, weights_cfg: dict) -> pd.DataFrame:
    """按 groups.yaml 聚合（默认等权）；weights.yaml 可指定权重"""
    # 解析 groups_cfg：{group: {"members":[...], "method":"weighted"/"mean", "weights":{...}} 或
    #                  {group: ["a","b",...]} 的简化形式
    groups_map: Dict[str, List[str]] = {}
    methods: Dict[str, str] = {}
    wmap: Dict[str, Dict[str, float]] = {}

    if not groups_cfg:
        groups_map = DEFAULT_GROUPS
        methods = {g: "mean" for g in groups_map}
    else:
        for g, spec in groups_cfg.items():
            if isinstance(spec, dict):
                members = spec.get("members") or spec.get("items") or []
                groups_map[g] = list(members)
                methods[g] = str(spec.get("method", "mean"))
                # weights: 来自两处，优先 spec，再回退 weights_cfg
                w1 = spec.get("weights") or {}
                w2 = weights_cfg.get(g, {}) if isinstance(weights_cfg, dict) else {}
                wm = {**w2, **w1} if isinstance(w2, dict) else w1
                wmap[g] = {k: float(v) for k, v in wm.items()} if isinstance(wm, dict) else {}
            elif isinstance(spec, list):
                groups_map[g] = list(spec)
                methods[g] = "mean"
            else:
                continue

    out = {}
    for g, members in groups_map.items():
        cols = [c for c in members if c in sub_z.columns]
        if not cols:
            out[g] = pd.Series(np.nan, index=sub_z.index)
            continue
        mat = sub_z[cols]
        if methods.get(g, "mean") == "weighted":
            w = np.array([wmap.get(g, {}).get(c, 1.0) for c in cols], dtype=float)
            if np.all(np.isnan(w)) or np.isclose(w.sum(), 0):
                val = mat.mean(axis=1)
            else:
                w = w / w.sum()
                val = np.nansum(mat.values * w[None, :], axis=1)
                val = pd.Series(val, index=mat.index)
        else:
            val = mat.mean(axis=1)
        out[g] = val
    return pd.DataFrame(out).reindex(index=sub_z.index)

# ==========================
# 主入口（保持对外签名）
# ==========================
def style_exposures(date: pd.Timestamp,
                    codes: List[str],
                    panel: pd.DataFrame,
                    funda_snapshot: pd.DataFrame,
                    *,
                    by_industry: bool = False,
                    industry: Optional[pd.Series] = None,
                    include_subfactors: bool = False,
                    beta_market: str = "000300.SH",
                    beta_lookback: int = 250,
                    beta_halflife: int = 60
                    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算当日风格大类（默认）或（大类, 小类）矩阵。
    - 输入：date, codes, panel(≥500D), funda_snapshot(≤t-LAG)
    - 清洗：winsor(1%) + z-score；可选 by_industry 标准化
    """
    t = pd.Timestamp(date).normalize()
    codes = list(dict.fromkeys([_norm_code(c) for c in codes]))
    log.step(f"[STEP] Styles @ t={t.strftime('%Y%m%d')}: codes={len(codes)}")

    # 配置加载
    sub_cfg, grp_cfg, w_cfg = _load_configs()

    # 快照水化
    funda = _hydrate_funda(t, codes, funda_snapshot)
    ctx = SubfactorContext(t, codes, panel, funda)

    # 价量/回归
    beta_s, hsigma_s = _compute_beta_hsigma(ctx, beta_market, beta_lookback, beta_halflife)
    rstr_s = _compute_rstr(ctx, 500, 21, 120)
    dastd_s = _compute_dastd(ctx, 250, 40)
    cmra_s = _compute_cmra(ctx)

    # 财报/流动性/epibs
    lncap_s = _compute_lncap(ctx)
    val_map = _compute_value_ratios(ctx)
    glp_map = _compute_growth_leverage_profitability(ctx)
    epibs_s = _compute_epibs(ctx, industry)
    sto_map = _compute_stom_stoq_stoa(ctx)

    # 组装小类
    sub_df = pd.DataFrame({
        "beta_252_hl60": beta_s,
        "hsigma_250": hsigma_s,
        "rstr_500_skip21_hl120": rstr_s,
        "dastd_250_hl40": dastd_s,
        "cmra_12m": cmra_s,
        "lncap": lncap_s,
        "epibs": epibs_s,
        **val_map,
        **glp_map,
        **sto_map,
    }).reindex(index=codes)
    sub_df = clip_inf_nan(sub_df)

    # 若 subfactors.yaml 中还有额外表达式，这里可按需扩展（默认关闭，避免误用）
    # if isinstance(sub_cfg, dict) and sub_cfg:
    #     env = {**{k: funda[k] for k in funda.columns if k not in sub_df.columns},
    #            **{k: sub_df[k] for k in sub_df.columns}}
    #     for name, expr in sub_cfg.items():
    #         if name in sub_df.columns or not isinstance(expr, str): continue
    #         try:
    #             with np.errstate(all="ignore"):
    #                 series = pd.eval(expr, local_dict=env, engine="python")
    #             if isinstance(series, pd.Series):
    #                 sub_df[name] = pd.to_numeric(series, errors="coerce").reindex(sub_df.index)
    #         except Exception as e:
    #             log.warn(f"[STYLE] subfactors.yaml expr failed '{name}': {e}")

    # 清洗+标准化
    group_by = industry.reindex(codes) if (by_industry and isinstance(industry, pd.Series)) else None
    sub_z = standardize_exposures(sub_df, winsor=(WINSOR_PCT, 1 - WINSOR_PCT), by=group_by, ddof=ZSCORE_DDOF)
    _print_coverage("STYLE_SUB", sub_z)

    # 聚合到大类
    groups_df = _aggregate_groups(sub_z, grp_cfg if grp_cfg else DEFAULT_GROUPS,
                                  w_cfg if isinstance(w_cfg, dict) else {})
    groups_df = standardize_exposures(groups_df, winsor=(WINSOR_PCT, 1 - WINSOR_PCT), by=group_by, ddof=ZSCORE_DDOF)
    _print_coverage("STYLE_GRP", groups_df)

    groups_df = groups_df.reindex(columns=sorted(groups_df.columns))
    sub_z = sub_z.reindex(columns=sorted(sub_z.columns))
    log.done(f"[DONE] Styles ready: groups={groups_df.shape[1]}, subfactors={sub_z.shape[1]}")
    return (groups_df, sub_z) if include_subfactors else groups_df

# ==========================
# SMOKE（自检）
# ==========================
if __name__ == "__main__":  # pragma: no cover
    try:
        log.set_verbosity("STEP")
    except Exception:
        pass
    rng = np.random.default_rng(2024)
    # 生成 260 交易日×N股票 的伪面板
    dates = pd.bdate_range("2024-07-01", periods=260)
    codes = [f"{600000+i:06d}.SH" if i % 2 == 0 else f"{300000+i:06d}.SZ" for i in range(1, 201)]
    panel = []
    for c in codes:
        px = 10 + rng.normal(0, 0.02, size=len(dates)).cumsum()
        vol = rng.integers(100_000, 500_000, size=len(dates))
        df = pd.DataFrame({"date": dates, "code": c, "close": px, "volume": vol})
        panel.append(df)
    panel = pd.concat(panel, ignore_index=True)

    # 伪财报快照（支持回退估计）
    funda = pd.DataFrame({
        "code": codes,
        "market_cap": rng.uniform(1e9, 5e10, size=len(codes)),
        "float_mktcap": rng.uniform(5e8, 3e10, size=len(codes)),
        "pb": rng.uniform(0.8, 5.0, size=len(codes)),
        "book_equity": rng.uniform(5e8, 3e10, size=len(codes)),
        "net_profit_ttm": rng.normal(5e8, 2e8, size=len(codes)),
        "revenue_ttm": rng.normal(2e10, 5e9, size=len(codes)),
        "oper_cf_ttm": rng.normal(1e9, 3e8, size=len(codes)),
        "total_assets": rng.uniform(1e10, 1e11, size=len(codes)),
        "total_debt": rng.uniform(1e9, 5e10, size=len(codes)),
        "long_debt": rng.uniform(1e7, 5e9, size=len(codes)),
        "roe_ttm": rng.uniform(0.05, 0.2, size=len(codes)),
        "roa_ttm": rng.uniform(0.02, 0.08, size=len(codes)),
        "growth_rev_yoy": rng.normal(0.1, 0.05, size=len(codes)),
        "growth_np_yoy": rng.normal(0.08, 0.06, size=len(codes)),
        "float_shares": rng.uniform(1e7, 1e9, size=len(codes)),
        "total_shares": rng.uniform(1e7, 1e9, size=len(codes)),
    })

    t = dates[-1]
    # 预检：外部 CSV 可用率（仅统计 csv 是否能提供每股预测）
    csv_path = _discover_forecast_csv()
    if csv_path:
        s_fore = _forecast_eps_for_t(codes, t, csv_path, cols_map=None)
        ratio_fore = float(s_fore.notna().mean())
        log.step(f"[SMOKE][EPIBS] external_forecast_available_ratio={ratio_fore:.2f}")
    else:
        log.step("[SMOKE][EPIBS] no external CSV discovered; fallback-only expected")

    g, s = style_exposures(t, codes, panel, funda, by_industry=False, include_subfactors=True)
    ep_cov = float(s["epibs"].notna().mean()) if "epibs" in s.columns else float("nan")
    log.step(f"[SMOKE] groups: {g.shape}, subs: {s.shape}, epibs_coverage={ep_cov:.2f}")
