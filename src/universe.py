# """
# [CN] 股票池契约：按交易日构建当日 universe（比如中证500），并提供缓存/增量更新。
# [Purpose] Build the investable universe per trade day.
#
# Interfaces:
# - build_universe(index_code: str, date: pd.Timestamp) -> list[str]
#   Returns the list of investable codes (members) for `index_code` on `date`.
# - cache_members(index_code: str, dates: list[pd.Timestamp]) -> None
#   Optional cache warm-up. No return.
#
# Notes:
# - Exclude ST* if desired (policy TBD in config).
# - Log changes in membership for audit.
# """
#
# from __future__ import annotations
# from __future__ import annotations
#
# # # file: src/universe.py
# # # -*- coding: utf-8 -*-
# # """
# # Universe module — CSI500 membership (day-t snapshot, cache-first, delta-aware persistence).
# #
# # 职责
# # ----
# # - 给定交易日 t，返回当日 CSI500 成分（仅决定“谁在池子里”）。
# # - 本地缓存：仅在「首日无缓存」或「与前一交易日相比发生变动」时写入 YYYYMMDD.csv；
# #   若当日与上一交易日完全一致，则不写当日文件（日志打印 unchanged）。
# # - 不做任何可交易性过滤/行情/因子计算。
# #
# # 公开接口
# # --------
# # csi500(date: pd.Timestamp) -> list[str]
# # csi500_series(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, list[str]]
# # is_member(date: pd.Timestamp, code: str) -> bool
# # rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> list[str]
# #
# # 口径
# # ----
# # - 代码形如 "000001.SZ"/"600000.SH"，后缀大写，升序稳定。
# # - 非交易日输入：向下取最近交易日并 WARN。
# # - 缓存路径：data/ref/index_members/YYYYMMDD.csv（列：date, code）。
# # - 新增ever_seen.csv，记录每只股票的首次与最后一次出现日期。
# # """
# #
# # from pathlib import Path
# # from typing import Dict, List, Optional
# # import re
# # import pandas as pd
# # from config import CFG  # 导入整个配置对象
# # EXCLUDE_B_PREFIX = CFG.universe.EXCLUDE_B_PREFIX  # 从配置中获取 B 股前缀
# # from src.trading_calendar import most_recent_trading_day, last_n_trading_days, prev_trading_day  # type: ignore
# #
# #
# #
# #
# #
# # # --------------------------- Imports -----------------------------------------
# # try:
# #     if __package__:
# #         from .api.myquant_io import get_index_members  # type: ignore
# #         from .trading_calendar import most_recent_trading_day, last_n_trading_days  # type: ignore
# #         from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
# #         from .utils import logging as log  # type: ignore
# #         from config import INDEX_CODE, REF_DIR  # type: ignore
# #     else:
# #         raise ImportError
# # except Exception:
# #     from src.api.myquant_io import get_index_members  # type: ignore
# #     from src.trading_calendar import most_recent_trading_day, last_n_trading_days  # type: ignore
# #     from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
# #     from src.utils import logging as log  # type: ignore
# #     from config import INDEX_CODE, REF_DIR  # type: ignore
# #
# # # --------------------------- Paths & Patterns --------------------------------
# # REF_ROOT = Path(REF_DIR)
# # IDX_DIR = REF_ROOT / "index_members"
# # EVER_SEEN_PATH = REF_ROOT / "ever_seen.csv"  # New file to track first and last seen dates
# # A_SHARE_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ)$", re.IGNORECASE)
# #
# # # --------------------------- Helpers -----------------------------------------
# # def _cache_path(d: pd.Timestamp) -> Path:
# #     return IDX_DIR / f"{pd.Timestamp(d).strftime('%Y%m%d')}.csv"
# #
# #
# # def _normalize_code(code: str) -> Optional[str]:
# #     """Normalize to '######.(SH|SZ)' and filter non A-shares."""
# #     if not code:
# #         return None
# #     c = code.strip()
# #     if "." in c and (c.upper().startswith("SHSE.") or c.upper().startswith("SZSE.")):
# #         mkt, num = c.split(".", 1)
# #         suffix = "SH" if mkt.upper().startswith("SH") else "SZ"
# #         c = f"{num}.{suffix}"
# #     if A_SHARE_CODE_RE.match(c):
# #         num, suf = c.split(".")
# #         if num[:3] in EXCLUDE_B_PREFIX:
# #             return None
# #         return f"{num}.{suf.upper()}"
# #     if len(c) == 8 and c[:6].isdigit() and c[6:].isalpha():
# #         num, suf = c[:6], c[6:].upper()
# #         return _normalize_code(f"{num}.{suf}")
# #     return None
# #
# #
# # def _normalize_and_filter(codes: List[str]) -> List[str]:
# #     norm = []
# #     for x in codes:
# #         nx = _normalize_code(x)
# #         if nx:
# #             norm.append(nx)
# #     return sorted(set(norm))  # stable ascending order
# #
# #
# # def _read_cached_exact(d: pd.Timestamp) -> List[str]:
# #     """Read exact-day cache if exists; otherwise empty list."""
# #     p = _cache_path(d)
# #     if not p.exists():
# #         return []
# #     df = read_csv_safe(p, parse_dates=["date"])
# #     if df.empty or "code" not in df.columns:
# #         return []
# #     codes = [c for c in df["code"].astype(str).tolist() if _normalize_code(c)]
# #     return sorted(set(codes))
# #
# #
# # def _write_cache(d: pd.Timestamp, codes: List[str]) -> None:
# #     ensure_dir(IDX_DIR)
# #     df = pd.DataFrame({"date": [pd.Timestamp(d).strftime("%Y-%m-%d")] * len(codes), "code": codes})
# #     write_csv_atomic(_cache_path(d), df, index=False)
# #
# #
# # def _read_ever_seen() -> pd.DataFrame:
# #     """Read ever_seen.csv to track first and last seen dates of codes."""
# #     if EVER_SEEN_PATH.exists():
# #         return read_csv_safe(EVER_SEEN_PATH)
# #     return pd.DataFrame(columns=["code", "first_seen", "last_seen"])
# #
# #
# # def _write_ever_seen(df: pd.DataFrame) -> None:
# #     """Write the ever_seen.csv to persist the seen stocks' first and last seen dates."""
# #     write_csv_atomic(EVER_SEEN_PATH, df, index=False)
# #
# #
# # def _update_ever_seen(codes: List[str], date: pd.Timestamp) -> pd.DataFrame:
# #     """Update the ever_seen.csv with new seen stocks."""
# #     ever_seen_df = _read_ever_seen()
# #     current_date = pd.Timestamp(date).strftime('%Y-%m-%d')
# #
# #     for code in codes:
# #         if code not in ever_seen_df["code"].values:
# #             ever_seen_df = ever_seen_df.append({"code": code, "first_seen": current_date, "last_seen": current_date}, ignore_index=True)
# #         else:
# #             ever_seen_df.loc[ever_seen_df["code"] == code, "last_seen"] = current_date
# #
# #     _write_ever_seen(ever_seen_df)
# #     return ever_seen_df
# #
# #
# # # --------------------------- Public API --------------------------------------
# # def rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
# #     """
# #     Return the union of all CSI500 members within [start, end].
# #     Ensures no missing members across days in the range and writes updates to `ever_seen.csv`.
# #     """
# #     start, end = pd.Timestamp(start), pd.Timestamp(end)
# #     all_codes = set()
# #     days_in_range = pd.date_range(start=start, end=end, freq='B')
# #
# #     for day in days_in_range:
# #         codes = csi500(day)
# #         all_codes.update(codes)
# #
# #     ever_seen_df = _update_ever_seen(list(all_codes), end)
# #     return sorted(list(all_codes))
# #
# #
# # # def csi500(date: pd.Timestamp) -> List[str]:
# # #     """
# # #     Return CSI500 membership for trading day `date` (cache-first).
# # #     """
# #     # d_req = pd.Timestamp(date).normalize()
# #     # d = most_recent_trading_day(d_req)
# #     # if d != d_req:
# #     #     log.warn(f"[UNIV] {d_req.date()} not a trading day. Snap to {d.date()}.")
# #     #
# #     # # 1) 尝试命中当日缓存
# #     # cached_today = _read_cached_exact(d)
# #     # if cached_today:
# #     #     log.step(f"[UNIV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(cached_today)} [CACHE] hit")
# #     #     return cached_today
# #     #
# #     # # 2) 缓存未命中：从源取当日
# #     # log.step(f"[UNV] Universe CSI500 @ t={d.strftime('%Y%m%d')} — fetch from source")
# #     # raw_today = get_index_members(INDEX_CODE, d)
# #     # codes_today = _normalize_and_filter(raw_today)
# #     # _write_cache(d, codes_today)
# #     # return codes_today
# #     # Pseudocode inside csi500(date), keep public signature unchanged
# # def csi500(date: pd.Timestamp) -> List[str]:
# #         """
# #         Return CSI500 membership for trading day `date` (cache-first, sparse fallback).
# #         """
# #         d_req = pd.Timestamp(date).normalize()
# #         d = most_recent_trading_day(d_req)
# #         if d != d_req:
# #             # 仍保留对「查询日不是交易日」的提示
# #             log.warn(f"[UNIV] {d_req.date()} not a trading day. Snap to {d.date()}.")
# #
# #         # 1) 尝试命中当日缓存
# #         cached_today = _read_cached_exact(d)
# #         if cached_today:
# #             log.step(f"[UNIV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(cached_today)} [CACHE] hit")
# #             return cached_today
# #
# #         # 2) 直接用数据源取（多数 SDK 自带 snap）
# #         codes = get_index_members(INDEX_CODE, d)
# #
# #         # 3) 若仍拿不到，做「稀疏回退」（月度/季度锚点），避免逐日 while 刷屏
# #         if not codes or len(codes) == 0:
# #             anchors = [d] + [prev_trading_day(d, k) for k in (21, 42, 63, 84, 126, 189, 252)]
# #             snapped = None
# #             for a in anchors:
# #                 codes_try = get_index_members(INDEX_CODE, a)
# #                 if codes_try:
# #                     codes = codes_try
# #                     snapped = a
# #                     break
# #             if snapped is None:
# #                 log.warn(f"[UNIV] CSI500 snapshot not found up to {d.date()}-252d; returning empty.")
# #                 return []
# #             if snapped != d:
# #                 log.debug(f"[UNIV] CSI500 snapshot snapped {d.date()} -> {snapped.date()}")
# #
# #         # 4) 规范化并写入「以 d 为名」的缓存（方便下次直接命中）
# #         codes = _normalize_and_filter(codes)
# #         _write_cache(d, codes)
# #         log.step(f"[UNV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(codes)} [SRC] fetched")
# #         return codes
# #
# #
# #
# # def csi500_series(dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, List[str]]:
# #     """Batch helper: returns {date_t: [codes_t]} with per-day caching."""
# #     out: Dict[pd.Timestamp, List[str]] = {}
# #     if not dates:
# #         return out
# #     uniq_dates = sorted({pd.Timestamp(d).normalize() for d in dates})
# #     for d in uniq_dates:
# #         out[d] = csi500(d)
# #     return out
# #
# #
# # def is_member(date: pd.Timestamp, code: str) -> bool:
# #     """Check if a stock is a member of CSI500 on `date`."""
# #     norm = _normalize_code(code)
# #     if not norm:
# #         return False
# #     codes = csi500(date)
# #     return norm in set(codes)
# #
# # # --------------------------- Smoke (optional) --------------------------------
# # if __name__ == "__main__":  # pragma: no cover
# #     log.set_verbosity("STEP")
# #     today = pd.Timestamp.today().normalize()
# #     start = today - pd.Timedelta(days=30)
# #     union_codes = rolling_union_codes(start, today)
# #     log.done(f"[SMOKE] rolling union codes from {start.date()} to {today.date()}: {len(union_codes)} codes")
# # file: src/universe.py
# # -*- coding: utf-8 -*-
# """
# [CN] Universe（中证500）——“谁在池子里”的唯一真相来源。
# 职责：
# - 在给定交易日 t，返回当日 CSI500 成分列表（升序、稳定），不做可交易性过滤；
# - 本地缓存到 data/ref/index_members/YYYYMMDD.csv；只以“当日”文件命名，绝不以他日文件顶替；
# - 非交易日输入仅向前对齐到最近≤t的交易日；交易日缺失缓存则强制拉取当日并落盘；
# - 命中缓存打印为 DEBUG；批量拉取使用 loop_progress 节流；
# - 增强：维护 ever_seen.csv（code, first_seen, last_seen）；提供 rolling_union_codes() 供预拉窗口并集。
#
# Public API（与契约对齐）：
# - csi500(date: pd.Timestamp) -> list[str]
# - csi500_series(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, list[str]]
# - is_member(date: pd.Timestamp, code: str) -> bool
# - rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> list[str]  # 新增，已在子线程达成共识
#
# 备注：
# - 代码与注释统一英文；日志为项目格式（STEP/LOOP/DONE/WARN/DEBUG）。
# """
#
# from pathlib import Path
# from typing import Dict, List, Optional, Iterable, Tuple, Set
# import re
# import pandas as pd
#
# # --------------------------- Project imports (package-first; script fallback) ----
# try:
#     if __package__:
#         from .utils import logging as log  # type: ignore
#         from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
#         from .utils.state import with_file_lock  # type: ignore
#         from .trading_calendar import (  # type: ignore
#             is_trading_day, most_recent_trading_day, prev_trading_day,
#             get_trading_days,
#         )
#         from .api.myquant_io import get_index_members  # type: ignore
#         from config import CFG, INDEX_CODE, REF_DIR  # type: ignore
#     else:
#         raise ImportError
# except Exception:  # pragma: no cover
#     from src.utils import logging as log  # type: ignore
#     from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
#     from src.utils.state import with_file_lock  # type: ignore
#     from src.trading_calendar import (  # type: ignore
#         is_trading_day, most_recent_trading_day, prev_trading_day,
#         get_trading_days,
#     )
#     from src.api.myquant_io import get_index_members  # type: ignore
#     # config shims (prefer real config module if available)
#     from config import CFG  # type: ignore
#     INDEX_CODE = getattr(CFG.universe, "index_code", "SHSE.000905")  # type: ignore
#     REF_DIR = getattr(CFG.paths, "ref_dir", "data/ref")  # type: ignore
#
# # --------------------------- Paths & constants -----------------------------------
# REF_ROOT = Path(REF_DIR)
# IDX_DIR = REF_ROOT / "index_members"
# EVER_SEEN_PATH = REF_ROOT / "ever_seen.csv"
# LOCK_NAME = "index_members.lock"
#
# # B-share prefix filter (placed in config via UniverseConfig)
# _EXCLUDE_B_PREFIX: Set[str] = set(getattr(CFG.universe, "EXCLUDE_B_PREFIX", {"200", "900"}))  # e.g., {"200","900"}
#
# _A_SHARE_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ)$", re.IGNORECASE)
#
#
# # --------------------------- Code normalization ----------------------------------
# def _normalize_code(code: str | None) -> Optional[str]:
#     """
#     Normalize to '######.(SH|SZ)'; filter out non A-shares and B-share prefixes.
#     Accepts various vendor styles like 'SHSE.600000' / 'SZSE.000001' / '600000.SH'.
#     """
#     if not code:
#         return None
#     c = str(code).strip().upper()
#     if "." in c and (c.startswith("SHSE.") or c.startswith("SZSE.")):
#         mkt, num = c.split(".", 1)
#         suf = "SH" if mkt.startswith("SH") else "SZ"
#         c = f"{num}.{suf}"
#
#     if _A_SHARE_CODE_RE.match(c):
#         num, suf = c.split(".")
#         if num[:3] in _EXCLUDE_B_PREFIX:
#             return None
#         return f"{num}.{suf}"
#
#     if len(c) == 8 and c[:6].isdigit() and c[6:].isalpha():
#         # e.g., "600000SH"
#         num, suf = c[:6], c[6:]
#         return _normalize_code(f"{num}.{suf}")
#
#     return None
#
#
# def _stable_sorted_unique(codes: Iterable[str]) -> List[str]:
#     uniq = list(dict.fromkeys([x for x in codes if x]))  # stable dedup
#     return sorted(uniq)
#
#
# def _cache_path(d: pd.Timestamp) -> Path:
#     return IDX_DIR / f"{pd.Timestamp(d).strftime('%Y%m%d')}.csv"
#
#
# def _read_cache(d: pd.Timestamp) -> List[str]:
#     """Read same-day cache strictly; return [] on miss/corrupt; never substitute other dates."""
#     fp = _cache_path(d)
#     if not fp.exists():
#         log.debug(f"[UNIV][CACHE] miss → {fp.name}")
#         return []
#     df = read_csv_safe(fp, parse_dates=["date"])
#     if df.empty or "code" not in df.columns:
#         log.warn(f"[UNIV][CACHE] corrupt or empty → {fp.name}; ignoring.")
#         return []
#     codes = [_normalize_code(c) for c in df["code"].astype(str)]
#     out = _stable_sorted_unique([c for c in codes if c])
#     log.debug(f"[UNIV][CACHE] hit  → {fp.name} (size={len(out)})")
#     return out
#
#
# def _write_cache(d: pd.Timestamp, codes: List[str]) -> Path:
#     """Write exact-day snapshot under a name-based lock; update ever_seen.csv."""
#     ensure_dir(IDX_DIR)
#     fp = _cache_path(d)
#     df = pd.DataFrame({"date": [pd.Timestamp(d)] * len(codes), "code": codes})
#     with with_file_lock(LOCK_NAME, timeout_sec=15.0):
#         write_csv_atomic(fp, df, index=False)
#     log.done(f"[UNIV] Saved snapshot → {fp.name} (size={len(codes)})")
#     _update_ever_seen(d, codes)
#     return fp
#
#
# def _update_ever_seen(d: pd.Timestamp, codes: List[str]) -> None:
#     """Optional enhancement: track first_seen/last_seen for each code."""
#     try:
#         ensure_dir(EVER_SEEN_PATH, is_file=True)
#         df_old = read_csv_safe(EVER_SEEN_PATH, parse_dates=["first_seen", "last_seen"])
#         if df_old.empty:
#             df_old = pd.DataFrame(columns=["code", "first_seen", "last_seen"])
#         df_old["code"] = df_old["code"].astype(str)
#
#         now_map = {c: (pd.Timestamp(d), pd.Timestamp(d)) for c in codes}
#         if not df_old.empty:
#             # update last_seen; keep earliest first_seen
#             for _, row in df_old.iterrows():
#                 c = str(row["code"])
#                 fs = row["first_seen"]
#                 ls = row["last_seen"]
#                 if c in now_map:
#                     fs_new = min(pd.Timestamp(fs), pd.Timestamp(d)) if pd.notna(fs) else pd.Timestamp(d)
#                     now_map[c] = (fs_new, pd.Timestamp(d))
#                 else:
#                     now_map[c] = (pd.Timestamp(fs), pd.Timestamp(ls))
#
#         df_new = pd.DataFrame(
#             [{"code": c, "first_seen": v[0], "last_seen": v[1]} for c, v in now_map.items()]
#         )
#         with with_file_lock(LOCK_NAME, timeout_sec=10.0):
#             write_csv_atomic(EVER_SEEN_PATH, df_new.sort_values("code"), index=False)
#     except Exception as e:  # non-fatal
#         log.warn(f"[UNIV] ever_seen update failed: {e}")
#
#
# def _fetch_from_source(d: pd.Timestamp) -> List[str]:
#     """Fetch members for exact trade day `d` and normalize."""
#     try:
#         raw = get_index_members(INDEX_CODE, pd.Timestamp(d))
#         # tolerate list/Series/DataFrame
#         if raw is None:
#             raw = []
#         elif isinstance(raw, pd.DataFrame):
#             if "code" in raw.columns:
#                 raw = raw["code"].tolist()
#             else:
#                 raw = raw.iloc[:, 0].tolist()
#         elif not isinstance(raw, (list, tuple)):
#             raw = list(raw)
#
#         codes = [_normalize_code(c) for c in raw]
#         out = _stable_sorted_unique([c for c in codes if c])
#         return out
#     except Exception as e:
#         log.warn(f"[UNIV][SRC] fetch failed for {d.date()} ({INDEX_CODE}): {e}")
#         return []
#
#
# def _snap_to_trade_day(t: pd.Timestamp) -> Tuple[pd.Timestamp, bool]:
#     """Return (trade_day, snapped). Only snap when input is NOT a trading day."""
#     t = pd.Timestamp(t).normalize()
#     if is_trading_day(t):
#         return t, False
#     td = most_recent_trading_day(t)
#     if pd.isna(td):
#         td = t  # extremely defensive; will return []
#     return pd.Timestamp(td).normalize(), True
#
#
# def _delta_vs_prev(d: pd.Timestamp, today: List[str]) -> Tuple[int, int]:
#     """Compare today list with previous trade day's snapshot, return (added, removed)."""
#     try:
#         prev_d = prev_trading_day(d, n=1)
#     except Exception:
#         return (0, 0)
#     prev_list = _read_cache(prev_d)
#     if not prev_list:
#         return (0, 0)
#     s_today = set(today)
#     s_prev = set(prev_list)
#     return (len(s_today - s_prev), len(s_prev - s_today))
#
#
# # --------------------------- Public APIs ----------------------------------------
# def csi500(date: pd.Timestamp | str) -> List[str]:
#     """
#     Return the CSI500 membership list on trade day `date` (snap to ≤date only if non-trading-day).
#     - Never returns None; worst case returns [] and WARN.
#     - Strict same-day file naming: data/ref/index_members/YYYYMMDD.csv
#     """
#     t_in = pd.Timestamp(date).normalize()
#     t, snapped = _snap_to_trade_day(t_in)
#     if snapped:
#         log.warn(f"[UNIV] input {t_in.date()} is NOT a trading day → snap to {t.date()}")
#
#     # (1) Try same-day cache only
#     cached = _read_cache(t)
#     if cached:
#         log.debug(f"[UNIV] Universe CSI500 @ t={t.strftime('%Y%m%d')}, size={len(cached)} [CACHE] hit")
#         return cached
#
#     # (2) No cache → fetch EXACT day, write EXACT day file
#     log.step(f"[UNIV] Universe CSI500 @ t={t.strftime('%Y%m%d')} — fetch from source")
#     members = _fetch_from_source(t)
#
#     if not members:
#         log.warn(f"[UNIV] Empty members from source @ {t.date()} ({INDEX_CODE}); returning [].")
#         return []
#
#     # Write snapshot for today
#     _write_cache(t, members)
#
#     # Delta vs prev (for audit)
#     add_k, rm_k = _delta_vs_prev(t, members)
#     log.done(f"[UNIV] Δ vs prev: added={add_k}, removed={rm_k}")
#
#     return members
#
#
# def csi500_series(dates: List[pd.Timestamp | str]) -> Dict[pd.Timestamp, List[str]]:
#     """
#     Batch version: get membership for each day in `dates` (dates may include non-trading days).
#     Only missing days are fetched & written; cache hits are DEBUG-level.
#     """
#     if not dates:
#         return {}
#     norm_dates = [pd.Timestamp(d).normalize() for d in dates]
#     out: Dict[pd.Timestamp, List[str]] = {}
#     total = len(norm_dates)
#     log.step(f"[UNIV] Build CSI500 series for {total} days (cache-first)")
#     tick = 0
#     for i, d in enumerate(norm_dates, start=1):
#         out[pd.Timestamp(d)] = csi500(d)
#         tick = log.loop_progress("[UNIV] series", i, total, every=20, start_time=None)
#     log.done(f"[UNIV] series built for {total} days")
#     return out
#
#
# def rolling_union_codes(start: pd.Timestamp | str, end: pd.Timestamp | str) -> List[str]:
#     """
#     Return the union of CSI500 members over trading days in [start, end] (inclusive).
#     - Read per-day cache if available; fetch only missing days (and write them).
#     - Output is unique, A-share-only, stably sorted codes.
#     """
#     s = pd.Timestamp(start).normalize()
#     e = pd.Timestamp(end).normalize()
#     days = get_trading_days(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
#     if not days:
#         return []
#     union: Set[str] = set()
#     total = len(days)
#     log.step(f"[UNIV] Rolling union in [{s.date()}, {e.date()}] — {total} trade days")
#     for i, d in enumerate(days, start=1):
#         # Try cache; if miss, fetch exactly that day and write.
#         members = _read_cache(d)
#         if not members:
#             # fetch exact day
#             log.debug(f"[UNIV][WARMUP] miss @ {d.date()} → fetch")
#             members = _fetch_from_source(d)
#             if members:
#                 _write_cache(d, members)
#         union.update(members)
#         log.loop_progress("[UNIV] rolling_union", i, total, every=20, start_time=None)
#     out = _stable_sorted_unique(union)
#     log.done(f"[UNIV] rolling_union unique={len(out)}")
#     return out
#
#
# def is_member(date: pd.Timestamp | str, code: str) -> bool:
#     """
#     Fast membership check for given day/code (snap if non-trading-day).
#     """
#     code_n = _normalize_code(code)
#     if not code_n:
#         return False
#     members = csi500(date)
#     return code_n in set(members)
#
#
# # --------------------------- Smoke test (optional) -------------------------------
# if __name__ == "__main__":  # pragma: no cover
#     # Quick local sanity without external side-effects
#     today = most_recent_trading_day()
#     _ = csi500(today)
#     prev = prev_trading_day(today, 1)
#     _ = csi500_series([prev, today])
#     _ = rolling_union_codes(prev, today)
#     print("[SMOKE][UNIV] OK")
#
# file: src/universe.py
"""
[CN] 股票池契约：按交易日构建当日 universe（比如中证500），并提供缓存/增量更新。
[Purpose] Build the investable universe per trade day.

Interfaces:
- build_universe(index_code: str, date: pd.Timestamp) -> list[str]
  Returns the list of investable codes (members) for `index_code` on `date`.
- cache_members(index_code: str, dates: list[pd.Timestamp]) -> None
  Optional cache warm-up. No return.

Notes:
- Exclude ST* if desired (policy TBD in config).
- Log changes in membership for audit.

口径
----
- 代码形如 "000001.SZ"/"600000.SH"，后缀大写，升序稳定。
- 非交易日输入：向下取最近交易日并 WARN。
- 缓存路径：data/ref/index_members/YYYYMMDD.csv（列：date, code）。
- 新增ever_seen.csv，记录每只股票的首次与最后一次出现日期。
"""

from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd
import time
from config import CFG  # 导入整个配置对象

try:
    _RAW_EXCLUDE_B_PREFIX = CFG.universe.EXCLUDE_B_PREFIX  # 可能是 str/list/set/None
except Exception:
    _RAW_EXCLUDE_B_PREFIX = None

def _coerce_prefixes(x) -> tuple[str, ...]:
    """
    Normalize EXCLUDE_B_PREFIX from config to tuple[str,...].
    Accepts: None | str | list | tuple | set.
    - If str, allow comma/pipe/semicolon/space separated values, e.g. "200,900".
    - Every token coerced to UPPER string and trimmed.
    """
    if x is None:
        return ()
    if isinstance(x, str):
        parts = re.split(r"[,\|\s;]+", x.strip())
        return tuple(p.strip().upper() for p in parts if p and p.strip())
    if isinstance(x, (list, tuple, set)):
        return tuple(str(p).strip().upper() for p in x if str(p).strip())
    return ()

EXCLUDE_B_PREFIX = _coerce_prefixes(_RAW_EXCLUDE_B_PREFIX)

from src.trading_calendar import most_recent_trading_day, last_n_trading_days, prev_trading_day  # type: ignore

# --------------------------- Imports -----------------------------------------
try:
    if __package__:
        from .api.myquant_io import get_index_members  # type: ignore
        from .trading_calendar import most_recent_trading_day, last_n_trading_days, get_trading_days  # type: ignore
        from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
        from .utils import logging as log  # type: ignore
        from config import INDEX_CODE, REF_DIR  # type: ignore
    else:
        raise ImportError
except Exception:
    from src.api.myquant_io import get_index_members  # type: ignore
    from src.trading_calendar import most_recent_trading_day, last_n_trading_days, get_trading_days  # type: ignore
    from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
    from src.utils import logging as log  # type: ignore
    from config import INDEX_CODE, REF_DIR  # type: ignore

# --------------------------- Paths & Patterns --------------------------------
REF_ROOT = Path(REF_DIR)
IDX_DIR = REF_ROOT / "index_members"
EVER_SEEN_PATH = REF_ROOT / "ever_seen.csv"
A_SHARE_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ)$", re.IGNORECASE)

def _normalize_code(c: str) -> Optional[str]:
    if not isinstance(c, str):
        return None
    c = c.strip().upper()
    if not A_SHARE_CODE_RE.match(c):
        return None
    # optional filter (e.g., exclude B-shares by prefix)
    if EXCLUDE_B_PREFIX and c.startswith(EXCLUDE_B_PREFIX):
        return None
    return c

def _cache_path(d: pd.Timestamp) -> Path:
    return IDX_DIR / f"{pd.Timestamp(d).strftime('%Y%m%d')}.csv"

def _read_cached_exact(d: pd.Timestamp) -> List[str]:
    """Read exact-day cache if exists; otherwise empty list."""
    p = _cache_path(d)
    if not p.exists():
        return []
    df = read_csv_safe(p, parse_dates=["date"])
    if df.empty or "code" not in df.columns:
        return []
    codes = [c for c in df["code"].astype(str).tolist() if _normalize_code(c)]
    return sorted(set(codes))

def _write_cache(d: pd.Timestamp, codes: List[str]) -> None:
    ensure_dir(IDX_DIR)
    df = pd.DataFrame({"date": [pd.Timestamp(d).strftime("%Y-%m-%d")] * len(codes), "code": codes})
    write_csv_atomic(_cache_path(d), df, index=False)

def _normalize_and_filter(codes: List[str]) -> List[str]:
    return sorted({_normalize_code(x) for x in codes if _normalize_code(x)})

def _write_ever_seen_df(df: pd.DataFrame) -> None:
    ensure_dir(REF_ROOT)
    write_csv_atomic(EVER_SEEN_PATH, df, index=False)

def _update_ever_seen(codes: List[str], current_date: pd.Timestamp) -> pd.DataFrame:
    """Update ever_seen table once per run."""
    ensure_dir(REF_ROOT)
    if EVER_SEEN_PATH.exists():
        ever_seen_df = read_csv_safe(EVER_SEEN_PATH, parse_dates=["first_seen", "last_seen"])
    else:
        ever_seen_df = pd.DataFrame(columns=["code", "first_seen", "last_seen"])
    ever_seen_df = ever_seen_df.copy()

    # initialize missing rows
    new_codes = [c for c in codes if c not in set(ever_seen_df["code"].astype(str).tolist())]
    if new_codes:
        add = pd.DataFrame({"code": new_codes, "first_seen": pd.Timestamp(current_date), "last_seen": pd.Timestamp(current_date)})
        ever_seen_df = pd.concat([ever_seen_df, add], ignore_index=True)

    # update last_seen for all union codes
    ever_seen_df.loc[ever_seen_df["code"].isin(codes), "last_seen"] = pd.Timestamp(current_date)

    _write_ever_seen_df(ever_seen_df)
    return ever_seen_df

# --------------------------- Public API --------------------------------------
def rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """
    Return the union of all CSI500 members within [start, end] using a single-pass
    scan over **trading days**. This avoids O(T×W) re-scans and reduces I/O.

    - Uses get_trading_days(start, end) to iterate only real trading dates.
    - Prefers exact-day cache first; if missing, falls back to csi500(date),
      which will fetch from source and write cache once.
    - Updates ever_seen.csv once at the end of the scan.
    - Emits counters for cache hits/misses to estimate source calls.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = get_trading_days(start, end)

    t0 = time.time()
    all_codes: set[str] = set()
    cache_hits = 0
    cache_miss = 0

    for i, d in enumerate(days, 1):
        cached = _read_cached_exact(d)
        if cached:
            codes_t = cached
            cache_hits += 1
        else:
            codes_t = csi500(d)  # fetch & write cache inside
            cache_miss += 1

        all_codes.update(codes_t)

        # progress log every ~10% of the loop
        if i % max(1, len(days)//10) == 0:
            log.loop_progress("union_scan", i, len(days), extra={"unique_codes": len(all_codes)})

    # write ever_seen once
    _update_ever_seen(sorted(all_codes), end)
    elapsed = time.time() - t0
    log.done(f"[UNIV] union_scan days={len(days)}, unique_days={len(days)}, "
             f"calls=get_index_members≈{cache_miss} (cache_hit:{cache_hits}, cache_miss:{cache_miss}) time={elapsed:.2f}s")
    log.step(f"[UNIV] prefetch codes size={len(all_codes)}")
    return sorted(list(all_codes))

def csi500(date: pd.Timestamp) -> List[str]:
    """
    Return CSI500 membership for trading day `date` (cache-first).
    """
    d = most_recent_trading_day(pd.Timestamp(date).normalize())
    cached_today = _read_cached_exact(d)
    if cached_today:
        log.step(f"[UNIV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(cached_today)} [CACHE] hit")
        return cached_today

    # direct pull (SDK may snap)
    codes = get_index_members(INDEX_CODE, d)
    if not codes or len(codes) == 0:
        # sparse fallback to avoid while loops
        anchors = [d] + [prev_trading_day(d, k) for k in (21, 42, 63, 84, 126, 189, 252)]
        snapped = None
        for a in anchors:
            codes_try = get_index_members(INDEX_CODE, a)
            if codes_try:
                codes = codes_try
                snapped = a
                break
        if snapped is None:
            log.warn(f"[UNIV] CSI500 snapshot not found up to {d.date()}-252d; returning empty.")
            return []
        if snapped != d:
            log.debug(f"[UNIV] CSI500 snapshot snapped {d.date()} -> {snapped.date()}")

    codes = _normalize_and_filter(codes)
    _write_cache(d, codes)
    log.step(f"[UNV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(codes)} [SRC] fetched")
    return codes

def csi500_series(dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, List[str]]:
    out: Dict[pd.Timestamp, List[str]] = {}
    for d in dates:
        out[d] = csi500(d)
    return out

def is_member(code: str, date: pd.Timestamp) -> bool:
    """Check if a stock is a member of CSI500 on `date`."""
    norm = _normalize_code(code)
    if not norm:
        return False
    codes = csi500(date)
    return norm in set(codes)

# --------------------------- Smoke (optional) --------------------------------
if __name__ == "__main__":  # pragma: no cover
    log.set_verbosity("STEP")
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=30)
    union_codes = rolling_union_codes(start, today)
    log.done(f"[SMOKE] rolling union codes from {start.date()} to {today.date()}: {len(union_codes)} codes")
