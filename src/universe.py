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
"""

from __future__ import annotations

# file: src/universe.py
# -*- coding: utf-8 -*-
"""
Universe module — CSI500 membership (day-t snapshot, cache-first, delta-aware persistence).

职责
----
- 给定交易日 t，返回当日 CSI500 成分（仅决定“谁在池子里”）。
- 本地缓存：仅在「首日无缓存」或「与前一交易日相比发生变动」时写入 YYYYMMDD.csv；
  若当日与上一交易日完全一致，则不写当日文件（日志打印 unchanged）。
- 不做任何可交易性过滤/行情/因子计算。

公开接口
--------
csi500(date: pd.Timestamp) -> list[str]
csi500_series(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, list[str]]
is_member(date: pd.Timestamp, code: str) -> bool
rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> list[str]

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
from config import CFG  # 导入整个配置对象
EXCLUDE_B_PREFIX = CFG.universe.EXCLUDE_B_PREFIX  # 从配置中获取 B 股前缀




# --------------------------- Imports -----------------------------------------
try:
    if __package__:
        from .api.myquant_io import get_index_members  # type: ignore
        from .trading_calendar import most_recent_trading_day, last_n_trading_days  # type: ignore
        from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
        from .utils import logging as log  # type: ignore
        from config import INDEX_CODE, REF_DIR  # type: ignore
    else:
        raise ImportError
except Exception:
    from src.api.myquant_io import get_index_members  # type: ignore
    from src.trading_calendar import most_recent_trading_day, last_n_trading_days  # type: ignore
    from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic  # type: ignore
    from src.utils import logging as log  # type: ignore
    from config import INDEX_CODE, REF_DIR  # type: ignore

# --------------------------- Paths & Patterns --------------------------------
REF_ROOT = Path(REF_DIR)
IDX_DIR = REF_ROOT / "index_members"
EVER_SEEN_PATH = REF_ROOT / "ever_seen.csv"  # New file to track first and last seen dates
A_SHARE_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ)$", re.IGNORECASE)

# --------------------------- Helpers -----------------------------------------
def _cache_path(d: pd.Timestamp) -> Path:
    return IDX_DIR / f"{pd.Timestamp(d).strftime('%Y%m%d')}.csv"


def _normalize_code(code: str) -> Optional[str]:
    """Normalize to '######.(SH|SZ)' and filter non A-shares."""
    if not code:
        return None
    c = code.strip()
    if "." in c and (c.upper().startswith("SHSE.") or c.upper().startswith("SZSE.")):
        mkt, num = c.split(".", 1)
        suffix = "SH" if mkt.upper().startswith("SH") else "SZ"
        c = f"{num}.{suffix}"
    if A_SHARE_CODE_RE.match(c):
        num, suf = c.split(".")
        if num[:3] in EXCLUDE_B_PREFIX:
            return None
        return f"{num}.{suf.upper()}"
    if len(c) == 8 and c[:6].isdigit() and c[6:].isalpha():
        num, suf = c[:6], c[6:].upper()
        return _normalize_code(f"{num}.{suf}")
    return None


def _normalize_and_filter(codes: List[str]) -> List[str]:
    norm = []
    for x in codes:
        nx = _normalize_code(x)
        if nx:
            norm.append(nx)
    return sorted(set(norm))  # stable ascending order


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


def _read_ever_seen() -> pd.DataFrame:
    """Read ever_seen.csv to track first and last seen dates of codes."""
    if EVER_SEEN_PATH.exists():
        return read_csv_safe(EVER_SEEN_PATH)
    return pd.DataFrame(columns=["code", "first_seen", "last_seen"])


def _write_ever_seen(df: pd.DataFrame) -> None:
    """Write the ever_seen.csv to persist the seen stocks' first and last seen dates."""
    write_csv_atomic(EVER_SEEN_PATH, df, index=False)


def _update_ever_seen(codes: List[str], date: pd.Timestamp) -> pd.DataFrame:
    """Update the ever_seen.csv with new seen stocks."""
    ever_seen_df = _read_ever_seen()
    current_date = pd.Timestamp(date).strftime('%Y-%m-%d')

    for code in codes:
        if code not in ever_seen_df["code"].values:
            ever_seen_df = ever_seen_df.append({"code": code, "first_seen": current_date, "last_seen": current_date}, ignore_index=True)
        else:
            ever_seen_df.loc[ever_seen_df["code"] == code, "last_seen"] = current_date

    _write_ever_seen(ever_seen_df)
    return ever_seen_df


# --------------------------- Public API --------------------------------------
def rolling_union_codes(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """
    Return the union of all CSI500 members within [start, end].
    Ensures no missing members across days in the range and writes updates to `ever_seen.csv`.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    all_codes = set()
    days_in_range = pd.date_range(start=start, end=end, freq='B')

    for day in days_in_range:
        codes = csi500(day)
        all_codes.update(codes)

    ever_seen_df = _update_ever_seen(list(all_codes), end)
    return sorted(list(all_codes))


def csi500(date: pd.Timestamp) -> List[str]:
    """
    Return CSI500 membership for trading day `date` (cache-first).
    """
    d_req = pd.Timestamp(date).normalize()
    d = most_recent_trading_day(d_req)
    if d != d_req:
        log.warn(f"[UNIV] {d_req.date()} not a trading day. Snap to {d.date()}.")

    # 1) 尝试命中当日缓存
    cached_today = _read_cached_exact(d)
    if cached_today:
        log.step(f"[UNIV] Universe CSI500 @ t={d.strftime('%Y%m%d')}, size={len(cached_today)} [CACHE] hit")
        return cached_today

    # 2) 缓存未命中：从源取当日
    log.step(f"[UNV] Universe CSI500 @ t={d.strftime('%Y%m%d')} — fetch from source")
    raw_today = get_index_members(INDEX_CODE, d)
    codes_today = _normalize_and_filter(raw_today)
    _write_cache(d, codes_today)
    return codes_today


def csi500_series(dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, List[str]]:
    """Batch helper: returns {date_t: [codes_t]} with per-day caching."""
    out: Dict[pd.Timestamp, List[str]] = {}
    if not dates:
        return out
    uniq_dates = sorted({pd.Timestamp(d).normalize() for d in dates})
    for d in uniq_dates:
        out[d] = csi500(d)
    return out


def is_member(date: pd.Timestamp, code: str) -> bool:
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
