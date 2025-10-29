# """
# 变更摘要（最小化）：
# 1) 两日换仓：每日照常计算 residuals@t、回归收益@d=t-2、alpha@t；但仅在“距上次换仓已满 fwd_days（默认2）个交易日”的 t 才生成 orders。
# 2) 仓位表：新增 out/ts/positions.csv（schema: date, code, weight, date_code）。换仓日前读“上次仓位”→传入 portfolio.build_orders_from_alpha；换仓后把目标权重落盘，供下次复用。
# 3) 命名口径：alpha / orders 一律按下单日 t 命名；通过 attrs['date']=t 传递，外部写盘也强制 t。
#
# 不改任一对外接口签名。
# """
#
# from __future__ import annotations
#
# from typing import Optional, List, Dict
# from pathlib import Path
# import os
# import time
# import pandas as pd
# import numpy as np
#
# # ---------------- Project imports ----------------
# try:
#     if __package__:
#         from src.utils import logging as log  # type: ignore
#         from src.utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe, append_with_rolloff  # type: ignore
#         from src.utils.state import load_manifest, save_manifest  # type: ignore
#         from src.trading_calendar import (  # type: ignore
#             get_trading_days, last_n_trading_days, shift_trading_day, is_trading_day
#         )
#         from src.api.myquant_io import get_index_members, get_fundamentals_snapshot  # type: ignore
#         from src.industry import industry_dummies  # type: ignore
#         from src.styles import style_exposures  # type: ignore
#         from src.factors_191 import factor_exposures_191  # type: ignore
#         from src.neutralize import orthogonalize  # type: ignore
#         from src.regression import forward_return, cs_factor_returns  # type: ignore
#         from src.alpha import next_alpha_from_trailing_mean  # type: ignore
#         from src.portfolio import build_orders_from_alpha  # type: ignore
#         from src.preprocess import run_preprocess  # type: ignore
#         from config import load_config  # type: ignore
#     else:
#         raise ImportError
# except Exception:  # pragma: no cover
#     from src.utils import logging as log  # type: ignore
#     from src.utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe, append_with_rolloff  # type: ignore
#     from src.utils.state import load_manifest, save_manifest  # type: ignore
#     from src.trading_calendar import (  # type: ignore
#         get_trading_days, last_n_trading_days, shift_trading_day, is_trading_day
#     )
#     from src.api.myquant_io import get_index_members, get_fundamentals_snapshot  # type: ignore
#     from src.industry import industry_dummies  # type: ignore
#     from src.styles import style_exposures  # type: ignore
#     from src.factors_191 import factor_exposures_191  # type: ignore
#     from src.neutralize import orthogonalize  # type: ignore
#     from src.regression import forward_return, cs_factor_returns  # type: ignore
#     from src.alpha import next_alpha_from_trailing_mean  # type: ignore
#     from src.portfolio import build_orders_from_alpha  # type: ignore
#     from src.preprocess import run_preprocess  # type: ignore
#     from config import load_config  # type: ignore
#
# CFG = load_config()
# P = CFG.paths
#
# # ---------------- Helpers ----------------
# def _yyyymmdd(ts: pd.Timestamp) -> str:
#     return pd.Timestamp(ts).strftime("%Y%m%d")
#
# def _date_str(ts: pd.Timestamp) -> str:
#     return pd.Timestamp(ts).strftime("%Y-%m-%d")
#
# def _bind_logfile(tag: str) -> None:
#     if not CFG.logging.log_to_file:
#         return
#     ensure_dir(P.out_logs_dir)
#     stamp = time.strftime("%Y%m%d_%H%M%S")
#     logfile = str(P.out_logs_dir / f"{stamp}_{tag}.log")
#     try:
#         log.bind_logfile(logfile)
#     except Exception:
#         pass
#
# # --- Code normalizers: "SHSE.600000" -> "600000.SH"; "SZSE.000001" -> "000001.SZ" ---
# def _norm_code(code: str) -> str:
#     if code is None:
#         return code
#     s = str(code).strip().upper()
#     if s.startswith("SHSE."):
#         return s.split(".", 1)[1] + ".SH"
#     if s.startswith("SZSE."):
#         return s.split(".", 1)[1] + ".SZ"
#     return s
#
# def _norm_codes(codes: List[str]) -> List[str]:
#     return [_norm_code(c) for c in codes]
#
# def _panel_from_raw(codes_src: List[str], code_map: Dict[str, str], start: str, end: str) -> pd.DataFrame:
#     """Delegate loading/cleaning to preprocess.build_panel_from_csv so that
#     the file naming (normalized codes) stays consistent with preprocess stage."""
#     # 1) source codes -> normalized codes (CSV files are named by normalized codes)
#     codes_norm: List[str] = [code_map.get(c, c) for c in codes_src]
#
#     # 2) reuse trusted builder (read, concat, clip to window, sort, de-dup, clean)
#     from src.preprocess import build_panel_from_csv  # delayed import
#     from config import RAW_OHLCV_DIR
#
#     panel = build_panel_from_csv(
#         codes=codes_norm,
#         start=start,
#         end=end,
#         raw_dir=str(RAW_OHLCV_DIR),
#     )
#
#     # 3) stabilize schema
#     if not panel.empty:
#         panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
#         panel["code"] = panel["code"].astype(str)
#         panel = panel.sort_values(["date", "code"])
#
#     return panel
#
# # ---------------- Positions I/O ----------------
# def _positions_path() -> Path:
#     return P.out_ts_dir / "positions.csv"
#
# def _load_last_positions() -> Optional[pd.Series]:
#     """
#     Load last rebalance positions as Series(index=code, values=weight).
#     Returns None if no positions yet.
#     """
#     path = str(_positions_path())
#     df = read_csv_safe(path, parse_dates=["date"], default=pd.DataFrame())
#     if df is None or df.empty:
#         return None
#     df["date"] = pd.to_datetime(df["date"]).dt.normalize()
#     last_d = df["date"].max()
#     last = df[df["date"] == last_d]
#     if last.empty:
#         return None
#     s = last.set_index("code")["weight"].astype(float)
#     s.name = "weight"
#     return s
#
# def _save_positions(date_t: pd.Timestamp, weights: pd.Series, keep_last_days: int = 252) -> None:
#     """
#     Append today's target weights to positions.csv with rolling retention.
#     Schema: date, code, weight, date_code (key)
#     """
#     ensure_dir(P.out_ts_dir)
#     df = pd.DataFrame({
#         "date": pd.Timestamp(date_t).normalize(),
#         "code": weights.index.astype(str),
#         "weight": pd.to_numeric(weights.values, errors="coerce"),
#     })
#     df["date_code"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["code"].astype(str)
#     keep_last = int(keep_last_days) * max(100, len(df))  # 252 trading days × holdings upper bound
#     append_with_rolloff(str(_positions_path()), df, key="date_code", keep_last=keep_last)
#
# # ---------------- Rebalance schedule ----------------
# def _is_rebalance_day(t: pd.Timestamp, last_reb: Optional[pd.Timestamp], fwd_days: int) -> bool:
#     """
#     Rebalance iff no previous rebalance OR trading-day distance from last_reb >= fwd_days.
#     """
#     if last_reb is None:
#         return True
#     if t <= last_reb:
#         return False
#     # trading-day distance = (#trading days from last_reb (exclusive) to t (inclusive))
#     dist = len(get_trading_days(_date_str(last_reb), _date_str(t))) - 1
#     return dist >= int(fwd_days)
#
# # ---------------- Core Orchestration ----------------
# def run_daily(start: str, end: str, fq: str = "none") -> Optional[pd.DataFrame]:
#     """Run pipeline between start and end (inclusive). If `end` is a trading day, return that day's orders (only if rebalance day)."""
#     _bind_logfile("run")
#     log.set_verbosity(CFG.logging.verbosity)
#     log.step("[ENTRY] run_daily ...")
#
#     days = get_trading_days(start, end)
#     if not days:
#         log.done("[DONE] No trading days in range.")
#         return None
#
#     manifest = load_manifest(
#         str(P.manifest_path),
#         create_if_missing=True,
#         defaults={
#             "last_processed": None,
#             "window": int(CFG.universe.lookback_trading_days),
#             "last_rebalance": None,  # NEW: persist last rebalance date
#         },
#     )
#     last_processed = manifest.get("last_processed", None)
#     if last_processed:
#         last_processed = pd.to_datetime(str(last_processed)).normalize()
#         days = [d for d in days if d > last_processed]
#     if not days:
#         log.done("[DONE] No new trading days.")
#         return None
#
#     # ---------------- STEP 1/6: Preprocess once (incremental) ----------------
#     log.step("[STEP 1/6] Preprocess (incremental, once) ...")
#     first_day = days[0]
#     try:
#         win = last_n_trading_days(first_day, n=max(int(CFG.universe.lookback_trading_days), 252) + 148)
#         s_panel = win[0]
#     except Exception:
#         s_panel = pd.Timestamp(first_day) - pd.Timedelta(days=600)
#     try:
#         _ = run_preprocess(start=_date_str(s_panel), end=_date_str(days[-1]), warmup_days=None)
#     except Exception as ex:
#         log.warn(f"[DATA] run_preprocess failed for window {s_panel}→{days[-1]}: {ex}")
#     log.done("[STEP 1/6] Preprocess done")
#
#     orders_today: Optional[pd.DataFrame] = None
#     t0 = time.time()
#     fwd = int(getattr(CFG.universe, "forward_days", 2))
#     last_reb = manifest.get("last_rebalance", None)
#     last_reb = None if last_reb in (None, "", "None") else pd.to_datetime(str(last_reb)).normalize()
#
#     # ---------------- Daily loop ----------------
#     for i, t in enumerate(days, start=1):
#         print(t)
#         log.loop_progress(task=f"t={_date_str(t)}", current=i, total=len(days), start_time=t0, every=1)
#
#         # Universe（index 原始代码 -> 规范化代码）
#         try:
#             codes_src: List[str] = get_index_members(CFG.universe.index_code, t)
#         except Exception as ex:
#             log.warn(f"[UNIV] get_index_members failed at {t}: {ex}")
#             codes_src = []
#         codes_norm: List[str] = _norm_codes(codes_src)
#         code_map: Dict[str, str] = {s: n for s, n in zip(codes_src, codes_norm)}
#
#         # OHLCV 面板（含 t+2 以便计算 r_{t→t+2}）
#         try:
#             win = last_n_trading_days(t, n=max(int(CFG.universe.lookback_trading_days), 252) + 148)
#             s_win = win[0]
#             t_end = shift_trading_day(t, fwd)
#         except Exception:
#             s_win = pd.Timestamp(t) - pd.Timedelta(days=600)
#             t_end = t
#         panel = _panel_from_raw(codes_src=codes_src, code_map=code_map,
#                                 start=_date_str(s_win), end=_date_str(t_end))
#         log.step(f"[DATA] panel rows={len(panel)}, codes={len(set(panel['code']) if 'code' in panel.columns else [])}")
#
#         # ---------------- STEP 2/6: Residuals for TODAY (date = t) ----------------
#         # Industry dummies @ t
#         try:
#             inds_t = industry_dummies(t, codes_norm)
#         except Exception as ex:
#             log.warn(f"[IND] industry_dummies failed @ {t}: {ex}")
#             inds_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#
#         # Styles @ t（财报快照 → 风格）
#         try:
#             funda_t = get_fundamentals_snapshot(t, codes_src, lag_days=CFG.styles.lag_trading_days)
#             if isinstance(funda_t, pd.DataFrame) and not funda_t.empty:
#                 if funda_t.index.name is None:
#                     funda_t.index = pd.Index([_norm_code(x) for x in funda_t.index], name="code")
#                 else:
#                     funda_t.index = funda_t.index.map(_norm_code)
#         except Exception as ex:
#             log.warn(f"[STYLE] fundamentals snapshot failed @ {t}: {ex}")
#             funda_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#         try:
#             style_df_t = style_exposures(t, codes=codes_norm, panel=panel, funda_snapshot=funda_t)
#         except Exception as ex:
#             log.warn(f"[STYLE] style_exposures failed @ {t}: {ex}")
#             style_df_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#
#         # 191 因子 & 正交化（写 out/residuals/{t}_residuals.csv）
#         try:
#             factors_t = factor_exposures_191(t, panel=panel, codes=codes_norm)
#             factors_t.attrs["date"] = pd.Timestamp(t)
#         except Exception as ex:
#             log.warn(f"[F191] factor_exposures_191 failed @ {t}: {ex}")
#             factors_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#             factors_t.attrs["date"] = pd.Timestamp(t)
#         try:
#             resid_today = orthogonalize(factors=factors_t, styles=style_df_t, inds=inds_t)
#             try:
#                 resid_today.attrs["date"] = pd.Timestamp(t)   # ensure alpha uses order-day t
#             except Exception:
#                 pass
#         except Exception as ex:
#             log.warn(f"[NEU] orthogonalize failed @ {t}: {ex}")
#             resid_today = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#
#         # ---------------- STEP 3–4/6: Factor return regression BACKFILL (dates ≤ t-2) ----------------
#         hist_end = shift_trading_day(t, -fwd)   # d = t-2
#         hist_days = last_n_trading_days(hist_end, n=int(getattr(CFG.alpha, "trailing_days", 252)))
#         fr_path = str(P.out_ts_dir / "factor_returns.csv")
#         fr = read_csv_safe(fr_path, parse_dates=["date"], default=pd.DataFrame())
#         have_dates = set() if fr is None or fr.empty else set(pd.to_datetime(fr["date"]).dt.normalize())
#         need_days = [d for d in hist_days if d not in have_dates]
#         log.step(f"[REG] backfill need_days={len(need_days)} within lookback (up to {len(hist_days)} total)")
#
#         for d in need_days:
#             # inds/styles/factors @ d
#             try:
#                 inds_d = industry_dummies(d, codes_norm)
#             except Exception as ex:
#                 log.warn(f"[IND] industry_dummies failed @ {d}: {ex}")
#                 inds_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#             try:
#                 funda_d = get_fundamentals_snapshot(d, codes_src, lag_days=CFG.styles.lag_trading_days)
#                 if isinstance(funda_d, pd.DataFrame) and not funda_d.empty:
#                     if funda_d.index.name is None:
#                         funda_d.index = pd.Index([_norm_code(x) for x in funda_d.index], name="code")
#                     else:
#                         funda_d.index = funda_d.index.map(_norm_code)
#             except Exception as ex:
#                 log.warn(f"[STYLE] fundamentals snapshot failed @ {d}: {ex}")
#                 funda_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#             try:
#                 factors_d = factor_exposures_191(d, panel=panel, codes=codes_norm)
#                 factors_d.attrs["date"] = pd.Timestamp(d)
#                 # backfill days: skip residual CSV write to avoid file storm
#                 factors_d.attrs["skip_write_residual"] = True
#             except Exception as ex:
#                 log.warn(f"[F191] factor_exposures_191 failed @ {d}: {ex}")
#                 factors_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#                 factors_d.attrs["date"] = pd.Timestamp(d)
#                 factors_d.attrs["skip_write_residual"] = True
#             try:
#                 resid_d = orthogonalize(factors=factors_d, styles=style_exposures(d, codes=codes_norm, panel=panel, funda_snapshot=funda_d), inds=inds_d)
#             except Exception as ex:
#                 log.warn(f"[NEU] orthogonalize failed @ {d}: {ex}")
#                 resid_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
#             # Regression for day d (writes factor_returns & metrics)
#             try:
#                 y = forward_return(panel=panel, date=d, codes=codes_norm, fwd_days=fwd)
#                 resid_use = resid_d.loc[:, resid_d.notna().any(axis=0)] if isinstance(resid_d, pd.DataFrame) else resid_d
#                 _ = cs_factor_returns(
#                     y=y, styles=style_exposures(d, codes=codes_norm, panel=panel, funda_snapshot=funda_d),
#                     inds=inds_d, resid_f=resid_use, use_ridge=True, ridge_alpha=float(CFG.regression.ridge_alpha)
#                 )
#             except Exception as ex:
#                 log.warn(f"[REG] cs_factor_returns failed @ {d}: {ex}")
#
#         # ---------------- STEP 5/6: Alpha (use resid @ t; trailing mean from factor_returns) ----------------
#         _alpha_lb = int(getattr(CFG.alpha, "trailing_days", 252))
#         log.step(f"[STEP 5/6] Alpha projection (lookback={_alpha_lb}) ...")
#         if resid_today is None or resid_today.empty:
#             log.warn("[ALPHA] resid_today is empty; skip alpha.")
#             alpha = pd.Series(dtype=float)
#         else:
#             alpha = next_alpha_from_trailing_mean(
#                 resid_f_today=resid_today, factor_returns_path=fr_path,
#                 lookback_days=_alpha_lb, codes=codes_norm
#             )
#             try:
#                 alpha.attrs["date"] = pd.Timestamp(t)
#             except Exception:
#                 pass
#
#         # ---------------- STEP 6/6: Orders — rebalance only every fwd days ----------------
#         do_rebalance = _is_rebalance_day(t, last_reb, fwd)
#         if not do_rebalance:
#             log.step(f"[ORDERS] non-rebalance day @ {t:%Y-%m-%d} (fwd={fwd}); skip orders.")
#             orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])
#         else:
#             log.step(f"[STEP 6/6] Build orders from alpha (rebalance day, fwd={fwd}) ...")
#             if isinstance(alpha, pd.Series) and not alpha.empty and alpha.notna().any():
#                 try:
#                     prev_w = _load_last_positions()
#                     orders = build_orders_from_alpha(
#                         alpha=alpha,
#                         mode=str(CFG.portfolio.mode),
#                         top_k=int(CFG.portfolio.top_k),
#                         max_weight=float(CFG.portfolio.max_weight),
#                         neutral=bool(CFG.portfolio.neutral),
#                         turnover_threshold=float(CFG.portfolio.turnover_threshold),
#                         prev_target_weights=prev_w,
#                     )
#                 except Exception as ex:
#                     log.warn(f"[ORDERS] build_orders_from_alpha failed: {ex}")
#                     orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])
#             else:
#                 log.warn("[ORDERS] skip: alpha empty/invalid; no order generated today")
#                 orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])
#
#             # 外部强制写一遍（同 portfolio 内部输出），确保按下单日 t 命名
#             if orders is not None and not orders.empty:
#                 try:
#                     ensure_dir(P.out_orders_dir)
#                     out_path = P.out_orders_dir / f"{_yyyymmdd(t)}_orders.csv"
#                     write_csv_atomic(str(out_path), orders, index=False)
#                     log.done(f"[ORDERS] K={len(orders)} mode={CFG.portfolio.mode}")
#                 except Exception as ex:
#                     log.warn(f"[ORDERS] write failed: {ex}")
#
#                 # 同步更新 positions.csv（用目标权重）
#                 try:
#                     pos_weights = orders.set_index("code")["target_weight"].astype(float)
#                     _save_positions(t, pos_weights, keep_last_days=252)
#                     log.done(f"[POS] positions updated @ {t:%Y-%m-%d}, names={len(pos_weights)}")
#                 except Exception as ex:
#                     log.warn(f"[POS] write positions failed: {ex}")
#
#                 # 更新上次换仓日
#                 last_reb = t
#                 manifest["last_rebalance"] = _date_str(last_reb)
#
#         # 最终返回（仅当 end 是交易日且为换仓日时返回当日 orders）
#         if t == days[-1] and is_trading_day(t) and do_rebalance:
#             orders_today = orders
#
#         # Persist manifest（每日推进 last_processed）
#         try:
#             manifest["last_processed"] = _date_str(t)
#             save_manifest(manifest, str(P.manifest_path))
#         except Exception as ex:
#             log.warn(f"[STATE] save_manifest failed @ {t}: {ex}")
#
#     log.done("[DONE] run_daily finished.")
#     return orders_today
#
#
# if __name__ == "__main__":  # pragma: no cover
#     s, e = os.environ.get("RUN_START", "2024-01-02"), os.environ.get("RUN_END", "2024-01-07")
#     run_daily(s, e, fq=os.environ.get("RUN_FQ", "pre"))
# file: run_daily.py
# -*- coding: utf-8 -*-
"""
本文件职责：串起日常 1→6 步，落地滚动增量；当日(t)生成 residuals/alpha；回归日(d=t-2)追加因子收益；
执行“2日一换仓”并可调用优化器（多空/中性约束）。
外部依赖：trading_calendar / api.myquant_io / industry / styles / factors_191 / neutralize / regression / alpha / portfolio / utils.*

关键改动（定位用）：
- [A] Alpha 后：alpha.attrs.update({...}) → 给 optimizer 传 styles/inds/bench 与约束（多空、行业±5%、风格±0.1）。
- [B] Orders 前：两日一换仓判定；读写 positions.csv；把 prev_target_weights 传给 portfolio。
- [C] 顶部 helpers：_load_last_positions/_save_positions/_is_rebalance_day。
"""

from __future__ import annotations

from typing import Optional, List, Dict
from pathlib import Path
import os
import time
import pandas as pd
import numpy as np

# ---------------- Project imports ----------------
try:
    if __package__:
        from src.utils import logging as log  # type: ignore
        from src.utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe, append_with_rolloff  # type: ignore
        from src.utils.state import load_manifest, save_manifest  # type: ignore
        from src.trading_calendar import (  # type: ignore
            get_trading_days, last_n_trading_days, shift_trading_day, is_trading_day
        )
        from src.api.myquant_io import get_index_members, get_fundamentals_snapshot, get_index_weights # type: ignore
        from src.industry import industry_dummies  # type: ignore
        from src.styles import style_exposures  # type: ignore
        from src.factors_191 import factor_exposures_191  # type: ignore
        from src.neutralize import orthogonalize  # type: ignore
        from src.regression import forward_return, cs_factor_returns  # type: ignore
        from src.alpha import next_alpha_from_trailing_mean  # type: ignore
        from src.portfolio import build_orders_from_alpha  # type: ignore
        from src.preprocess import run_preprocess  # type: ignore
        from config import load_config  # type: ignore
    else:
        raise ImportError
except Exception:  # pragma: no cove
    from src.utils import logging as log  # type: ignore
    from src.utils.fileio import ensure_dir, write_csv_atomic, read_csv_safe, append_with_rolloff  # type: ignore
    from src.utils.state import load_manifest, save_manifest  # type: ignore
    from src.trading_calendar import (  # type: ignore
        get_trading_days, last_n_trading_days, shift_trading_day, is_trading_day
    )
    from src.api.myquant_io import get_index_members, get_fundamentals_snapshot,get_index_weights  # type: ignore
    from src.industry import industry_dummies  # type: ignore
    from src.styles import style_exposures  # type: ignore
    from src.factors_191 import factor_exposures_191  # type: ignore
    from src.neutralize import orthogonalize  # type: ignore
    from src.regression import forward_return, cs_factor_returns  # type: ignore
    from src.alpha import next_alpha_from_trailing_mean  # type: ignore
    from src.portfolio import build_orders_from_alpha  # type: ignore
    from src.preprocess import run_preprocess  # type: ignore
    from config import load_config  # type: ignore

CFG = load_config()
P = CFG.paths

# ---------------- Helpers ----------------
def _yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")

def _date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

def _bind_logfile(tag: str) -> None:
    if not CFG.logging.log_to_file:
        return
    ensure_dir(P.out_logs_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logfile = str(P.out_logs_dir / f"{stamp}_{tag}.log")
    try:
        log.bind_logfile(logfile)
    except Exception:
        pass

# --- Code normalizers: "SHSE.600000" -> "600000.SH"; "SZSE.000001" -> "000001.SZ" ---
def _norm_code(code: str) -> str:
    if code is None:
        return code
    s = str(code).strip().upper()
    if s.startswith("SHSE."):
        return s.split(".", 1)[1] + ".SH"
    if s.startswith("SZSE."):
        return s.split(".", 1)[1] + ".SZ"
    return s

def _norm_codes(codes: List[str]) -> List[str]:
    return [_norm_code(c) for c in codes]

def _panel_from_raw(codes_src: List[str], code_map: Dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Reuse preprocess builder to get cleaned panel aligned to normalized codes."""
    from src.preprocess import build_panel_from_csv  # delayed import
    from config import RAW_OHLCV_DIR
    codes_norm: List[str] = [code_map.get(c, c) for c in codes_src]
    panel = build_panel_from_csv(codes=codes_norm, start=start, end=end, raw_dir=str(RAW_OHLCV_DIR))
    if not panel.empty:
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel["code"] = panel["code"].astype(str)
        panel = panel.sort_values(["date", "code"])
    return panel

# ---------------- Positions I/O  [C] ----------------
def _positions_path() -> Path:
    return P.out_ts_dir / "positions.csv"

def _load_last_positions() -> Optional[pd.Series]:
    """Load last rebalance positions as Series(index=code, values=weight)."""
    path = str(_positions_path())
    df = read_csv_safe(path, parse_dates=["date"], default=pd.DataFrame())
    if df is None or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    last_d = df["date"].max()
    last = df[df["date"] == last_d]
    if last.empty:
        return None
    s = last.set_index("code")["weight"].astype(float)
    s.name = "weight"
    return s

def _save_positions(date_t: pd.Timestamp, weights: pd.Series, keep_last_days: int = 252) -> None:
    """Append today's target weights to positions.csv with rolling retention."""
    ensure_dir(P.out_ts_dir)
    df = pd.DataFrame({
        "date": pd.Timestamp(date_t).normalize(),
        "code": weights.index.astype(str),
        "weight": pd.to_numeric(weights.values, errors="coerce"),
    })
    df["date_code"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["code"].astype(str)
    keep_last = int(keep_last_days) * max(100, len(df))  # 粗上界
    append_with_rolloff(str(_positions_path()), df, key="date_code", keep_last=keep_last)

# ---------------- Rebalance schedule  [C] ----------------
def _is_rebalance_day(t: pd.Timestamp, last_reb: Optional[pd.Timestamp], fwd_days: int) -> bool:
    """Rebalance iff no previous rebalance OR trading-day distance from last_reb >= fwd_days."""
    if last_reb is None:
        return True
    if t <= last_reb:
        return False
    dist = len(get_trading_days(_date_str(last_reb), _date_str(t))) - 1
    return dist >= int(fwd_days)

# ---------------- Core Orchestration ----------------
def run_daily(start: str, end: str, fq: str = "none") -> Optional[pd.DataFrame]:
    """Run pipeline between start and end (inclusive). If `end` is a trading day, return that day's orders (only if rebalance day)."""
    _bind_logfile("run")
    log.set_verbosity(CFG.logging.verbosity)
    log.step("[ENTRY] run_daily ...")

    days = get_trading_days(start, end)
    if not days:
        log.done("[DONE] No trading days in range.")
        return None

    manifest = load_manifest(
        str(P.manifest_path),
        create_if_missing=True,
        defaults={"last_processed": None, "window": int(CFG.universe.lookback_trading_days), "last_rebalance": None},
    )
    last_processed = manifest.get("last_processed", None)
    if last_processed:
        last_processed = pd.to_datetime(str(last_processed)).normalize()
        days = [d for d in days if d > last_processed]
    if not days:
        log.done("[DONE] No new trading days.")
        return None

    # ---------------- STEP 1/6: Preprocess once (incremental) ----------------
    log.step("[STEP 1/6] Preprocess (incremental, once) ...")
    first_day = days[0]
    try:
        win = last_n_trading_days(first_day, n=max(int(CFG.universe.lookback_trading_days), 252) + 148)
        s_panel = win[0]
    except Exception:
        s_panel = pd.Timestamp(first_day) - pd.Timedelta(days=600)
    try:
        _ = run_preprocess(start=_date_str(s_panel), end=_date_str(days[-1]), warmup_days=None)
    except Exception as ex:
        log.warn(f"[DATA] run_preprocess failed for window {s_panel}→{days[-1]}: {ex}")
    log.done("[STEP 1/6] Preprocess done")

    orders_today: Optional[pd.DataFrame] = None
    t0 = time.time()
    fwd = int(getattr(CFG.universe, "forward_days", 2))
    last_reb = manifest.get("last_rebalance", None)
    last_reb = None if last_reb in (None, "", "None") else pd.to_datetime(str(last_reb)).normalize()

    # ---------------- Daily loop ----------------
    for i, t in enumerate(days, start=1):
        print(t)
        log.loop_progress(task=f"t={_date_str(t)}", current=i, total=len(days), start_time=t0, every=1)

        # Universe（index 原始代码 -> 规范化代码）
        try:
            codes_src: List[str] = get_index_members(CFG.universe.index_code, t)
        except Exception as ex:
            log.warn(f"[UNIV] get_index_members failed at {t}: {ex}")
            codes_src = []
        codes_norm: List[str] = [_norm_code(c) for c in codes_src]
        code_map: Dict[str, str] = {s: n for s, n in zip(codes_src, codes_norm)}

        # OHLCV 面板（含 t+2 以便计算 r_{t→t+2}）
        try:
            win = last_n_trading_days(t, n=max(int(CFG.universe.lookback_trading_days), 252) + 148)
            s_win = win[0]
            t_end = shift_trading_day(t, fwd)
        except Exception:
            s_win = pd.Timestamp(t) - pd.Timedelta(days=600)
            t_end = t
        panel = _panel_from_raw(codes_src=codes_src, code_map=code_map, start=_date_str(s_win), end=_date_str(t_end))
        log.step(f"[DATA] panel rows={len(panel)}, codes={len(set(panel['code']) if 'code' in panel.columns else [])}")

        # ---------------- STEP 2/6: Residuals for TODAY (date = t) ----------------
        # Industry @ t
        try:
            inds_t = industry_dummies(t, codes_norm)
        except Exception as ex:
            log.warn(f"[IND] industry_dummies failed @ {t}: {ex}")
            inds_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))

        # Styles @ t（财报快照 → 风格）
        try:
            funda_t = get_fundamentals_snapshot(t, codes_src, lag_days=CFG.styles.lag_trading_days)
            if isinstance(funda_t, pd.DataFrame) and not funda_t.empty:
                if funda_t.index.name is None:
                    funda_t.index = pd.Index([_norm_code(x) for x in funda_t.index], name="code")
                else:
                    funda_t.index = funda_t.index.map(_norm_code)
        except Exception as ex:
            log.warn(f"[STYLE] fundamentals snapshot failed @ {t}: {ex}")
            funda_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
        try:
            style_df_t = style_exposures(t, codes=codes_norm, panel=panel, funda_snapshot=funda_t)
        except Exception as ex:
            log.warn(f"[STYLE] style_exposures failed @ {t}: {ex}")
            style_df_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))

        # 191 因子 & 正交化（写 out/residuals/{t}_residuals.csv）
        try:
            factors_t = factor_exposures_191(t, panel=panel, codes=codes_norm)
            factors_t.attrs["date"] = pd.Timestamp(t)
        except Exception as ex:
            log.warn(f"[F191] factor_exposures_191 failed @ {t}: {ex}")
            factors_t = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
            factors_t.attrs["date"] = pd.Timestamp(t)
        try:
            resid_today = orthogonalize(factors=factors_t, styles=style_df_t, inds=inds_t)
            try:
                resid_today.attrs["date"] = pd.Timestamp(t)   # ensure alpha uses order-day t
            except Exception:
                pass
        except Exception as ex:
            log.warn(f"[NEU] orthogonalize failed @ {t}: {ex}")
            resid_today = pd.DataFrame(index=pd.Index(codes_norm, name="code"))

        # ---------------- STEP 3–4/6: Factor return regression BACKFILL (dates ≤ t-2) ----------------
        hist_end = shift_trading_day(t, -fwd)   # d = t-2
        hist_days = last_n_trading_days(hist_end, n=int(getattr(CFG.alpha, "trailing_days", 252)))
        fr_path = str(P.out_ts_dir / "factor_returns.csv")
        fr = read_csv_safe(fr_path, parse_dates=["date"], default=pd.DataFrame())
        have_dates = set() if fr is None or fr.empty else set(pd.to_datetime(fr["date"]).dt.normalize())
        need_days = [d for d in hist_days if d not in have_dates]
        log.step(f"[REG] backfill need_days={len(need_days)} within lookback (up to {len(hist_days)} total)")

        for d in need_days:
            # inds/styles/factors @ d
            try:
                inds_d = industry_dummies(d, codes_norm)
            except Exception as ex:
                log.warn(f"[IND] industry_dummies failed @ {d}: {ex}")
                inds_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
            try:
                funda_d = get_fundamentals_snapshot(d, codes_src, lag_days=CFG.styles.lag_trading_days)
                if isinstance(funda_d, pd.DataFrame) and not funda_d.empty:
                    if funda_d.index.name is None:
                        funda_d.index = pd.Index([_norm_code(x) for x in funda_d.index], name="code")
                    else:
                        funda_d.index = funda_d.index.map(_norm_code)
            except Exception as ex:
                log.warn(f"[STYLE] fundamentals snapshot failed @ {d}: {ex}")
                funda_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
            try:
                factors_d = factor_exposures_191(d, panel=panel, codes=codes_norm)
                factors_d.attrs["date"] = pd.Timestamp(d)
                factors_d.attrs["skip_write_residual"] = True  # backfill不写残差文件
            except Exception as ex:
                log.warn(f"[F191] factor_exposures_191 failed @ {d}: {ex}")
                factors_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
                factors_d.attrs["date"] = pd.Timestamp(d)
                factors_d.attrs["skip_write_residual"] = True
            try:
                resid_d = orthogonalize(factors=factors_d, styles=style_exposures(d, codes=codes_norm, panel=panel, funda_snapshot=funda_d), inds=inds_d)
            except Exception as ex:
                log.warn(f"[NEU] orthogonalize failed @ {d}: {ex}")
                resid_d = pd.DataFrame(index=pd.Index(codes_norm, name="code"))
            # Regression for day d (writes factor_returns & metrics)
            try:
                y = forward_return(panel=panel, date=d, codes=codes_norm, fwd_days=fwd)
                resid_use = resid_d.loc[:, resid_d.notna().any(axis=0)] if isinstance(resid_d, pd.DataFrame) else resid_d
                _ = cs_factor_returns(
                    y=y, styles=style_exposures(d, codes=codes_norm, panel=panel, funda_snapshot=funda_d),
                    inds=inds_d, resid_f=resid_use, use_ridge=True, ridge_alpha=float(CFG.regression.ridge_alpha)
                )
            except Exception as ex:
                log.warn(f"[REG] cs_factor_returns failed @ {d}: {ex}")

        # ---------------- STEP 5/6: Alpha (use resid @ t; trailing mean from factor_returns) ----------------
        _alpha_lb = int(getattr(CFG.alpha, "trailing_days", 252))
        log.step(f"[STEP 5/6] Alpha projection (lookback={_alpha_lb}) ...")
        if resid_today is None or resid_today.empty:
            log.warn("[ALPHA] resid_today is empty; skip alpha.")
            alpha = pd.Series(dtype=float)
        else:
            alpha = next_alpha_from_trailing_mean(
                resid_f_today=resid_today, factor_returns_path=fr_path,
                lookback_days=_alpha_lb, codes=codes_norm
            )
            try:
                alpha.attrs["date"] = pd.Timestamp(t)
            except Exception:
                pass

        # >>> [A] Optimizer attrs: 多空 + 行业/风格中性（±5% / ±0.1）
        # 查找锚点：“Alpha projection (lookback=...) ...” 这段下面，紧跟着插入本段
        if isinstance(alpha, pd.Series) and not alpha.empty:
            try:
                # 相对中性的基准（等权；如有指数权重可替换）
                bench_raw = get_index_weights(CFG.universe.index_code, t)
                bench_w = bench_raw.reindex(pd.Index(codes_norm, name="code")).fillna(0.0)
                s = float(bench_w.sum()); bench_w = bench_w / s if s > 0 else bench_w
                alpha.attrs.update({
                    "styles": style_df_t,           # 当日风格截面（index=code）
                    "inds": inds_t,                 # 当日行业哑变量
                    "bench_weights": bench_w,       # 相对行业/风格目标
                    "allow_short": True,            # 允许做空
                    "industry_band": 0.05,          # 行业相对带宽 ±5%
                    "style_band": 0.10,             # 风格带宽 ±0.1
                    "gross_limit": 1.0,             # ∑|w| ≤ 1
                    "dollar_neutral": False,        # 若需净暴露=0 则 True
                    "solver": "auto",
                    "tc":0.003
                })
            except Exception as ex:
                log.warn(f"[ALPHA] set optimizer attrs failed: {ex}")

        # ---------------- STEP 6/6: Orders — rebalance only every fwd days ----------------
        do_rebalance = _is_rebalance_day(t, last_reb, fwd)
        if not do_rebalance:
            log.step(f"[ORDERS] non-rebalance day @ {t:%Y-%m-%d} (fwd={fwd}); skip orders.")
            orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])
        else:
            log.step(f"[STEP 6/6] Build orders from alpha (rebalance day, fwd={fwd}) ...")
            if isinstance(alpha, pd.Series) and not alpha.empty and alpha.notna().any():
                try:
                    prev_w = _load_last_positions()
                    orders = build_orders_from_alpha(
                        alpha=alpha,
                        mode="optimizer_bmark_neutral",  # 将配置设成 "optimizer" 即调用优化器
                        top_k=int(CFG.portfolio.top_k),
                        max_weight=float(CFG.portfolio.max_weight),
                        neutral=bool(CFG.portfolio.neutral),
                        turnover_threshold=float(CFG.portfolio.turnover_threshold),
                        prev_target_weights=prev_w,
                        allow_short=True
                    )
                except Exception as ex:
                    log.warn(f"[ORDERS] build_orders_from_alpha failed: {ex}")
                    orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])
            else:
                log.warn("[ORDERS] skip: alpha empty/invalid; no order generated today")
                orders = pd.DataFrame(columns=["date","code","target_weight","side","px_type","note"])

            # 外部强制写一遍（同 portfolio 内部输出），确保按下单日 t 命名
            if orders is not None and not orders.empty:
                try:
                    ensure_dir(P.out_orders_dir)
                    out_path = P.out_orders_dir / f"{_yyyymmdd(t)}_orders.csv"
                    write_csv_atomic(str(out_path), orders, index=False)
                    log.done(f"[ORDERS] K={len(orders)} mode={CFG.portfolio.mode}")
                except Exception as ex:
                    log.warn(f"[ORDERS] write failed: {ex}")

                # 同步更新 positions.csv（用目标权重）
                try:
                    pos_weights = orders.set_index("code")["target_weight"].astype(float)
                    _save_positions(t, pos_weights, keep_last_days=252)
                    log.done(f"[POS] positions updated @ {t:%Y-%m-%d}, names={len(pos_weights)}")
                except Exception as ex:
                    log.warn(f"[POS] write positions failed: {ex}")

                # 更新上次换仓日
                last_reb = t
                manifest["last_rebalance"] = _date_str(last_reb)

        # 最终返回（仅当 end 是交易日且为换仓日时返回当日 orders）
        if t == days[-1] and is_trading_day(t) and do_rebalance:
            orders_today = orders

        # Persist manifest（每日推进 last_processed）
        try:
            manifest["last_processed"] = _date_str(t)
            save_manifest(manifest, str(P.manifest_path))
        except Exception as ex:
            log.warn(f"[STATE] save_manifest failed @ {t}: {ex}")

    log.done("[DONE] run_daily finished.")
    return orders_today


if __name__ == "__main__":  # pragma: no cover
    s, e = os.environ.get("RUN_START", "2024-01-02"), os.environ.get("RUN_END", "2024-01-07")
    run_daily(s, e, fq=os.environ.get("RUN_FQ", "pre"))
