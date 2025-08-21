"""
[CN] 行业哑变量契约：当日对 codes 生成行业 One-Hot。
[Purpose] Provide industry dummy matrix aligned with codes.

Signature:
def industry_dummies(date: pd.Timestamp, codes: list[str]) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code, columns = ["IND_xxx" ...], 0/1 dummies.
    Must include an intercept handling policy: either drop-one or center later in regression.
    '''

Notes:
- Industry taxonomy (e.g., Shenwan, CSRC) to be fixed in config.
- Stable column naming across days; unseen industries → zero columns (or skipped).
"""

from __future__ import annotations
from __future__ import annotations

# file: src/industry.py
# -*- coding: utf-8 -*-
"""
Industry dummies for orthogonalization (申万/可切换 GICS；无前视；全集列对齐；缓存+日志).

Design highlights
-----------------
- Source: data/ref/industry_map/QT_INDUS_CONSTITUENTS_*.csv （时间戳越新的越优）
- Standard: default SW (申万) level=1; switchable via config: IND_STANDARD='GICS'
- No look-ahead: pick latest record with effective_date <= t (fallback to updatetime/tmstamp)
- Missing mapping -> OTHER (= 'IND_<STD>_L<level>__OTHER')
- Column space stability: maintain latest_columns.json (stable union; order-preserving dedup)
- Cache: data/ref/industry_map/YYYYMMDD_industry.csv (index=code, columns=IND_*)
- Logging: [STEP][IND] ... / [WARN][IND] missing=... -> OTHER / cache hit|miss

External deps (project-internal):
- utils.logging.{set_verbosity,step,done,warn,error,loop_progress}
- utils.fileio.{ensure_dir,read_csv_safe,write_csv_atomic}
- utils.state.with_file_lock
- config.{CFG, REF_DIR, LOG_VERBOSITY}  (optional IND_* overrides read from CFG if present)

Notes:
- We normalize codes to "######.(SH|SZ)". If source has plain 'stock_code' (no suffix),
  '6xxxxx' -> SH, else -> SZ.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import unicodedata
import glob
import pandas as pd

# ---------------- Project imports (package-first, script fallback) ----------------
try:
    if __package__:
        from .utils import logging as log
        from .utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic
        from .utils.state import with_file_lock
        from config import CFG, REF_DIR, LOG_VERBOSITY  # type: ignore
    else:
        raise ImportError
except Exception:
    from src.utils import logging as log
    from src.utils.fileio import ensure_dir, read_csv_safe, write_csv_atomic
    from src.utils.state import with_file_lock
    from config import CFG, REF_DIR, LOG_VERBOSITY  # type: ignore

# ---------------- Verbosity ----------------
try:
    log.set_verbosity(LOG_VERBOSITY)
except Exception:
    pass

# ---------------- Config (with safe fallbacks) ----------------
# 可在 config.py 里附加：
#   IND_STANDARD = "SW" | "GICS"
#   IND_LEVEL = 1
#   IND_DIR = "data/ref/industry_map"
#   IND_CONS_GLOB = "data/ref/industry_map/QT_INDUS_CONSTITUENTS_*.csv"
#   IND_COLMAP = {...}  # 见 DEFAULT_COLMAP
IND_STANDARD: str = getattr(CFG, "IND_STANDARD", "SW")
IND_LEVEL: int = int(getattr(CFG, "IND_LEVEL", 1))
IND_DIR: Path = Path(getattr(CFG, "IND_DIR", str(Path(REF_DIR) / "industry_map")))
IND_CONS_GLOB: str = getattr(CFG, "IND_CONS_GLOB", str(IND_DIR / "QT_INDUS_CONSTITUENTS_*.csv"))

DEFAULT_COLMAP: Dict[str, str] = {
    # 统一字段名（按你提供的预览图/约定做大小写兼容）
    "code": "STOCK_CODE",                 # or 'code'
    "stock_name": "STOCK_NAME",           # optional
    "standard_code": "STANDARD_CODE",
    "standard_name": "STANDARD_NAME",
    "industry_code": "INDUSTRY_CODE",
    "industry_name": "INDUSTRY_NAME",
    "industry_level": "INDUSTRY_LEVEL",
    "effective_date": "INTO_DATE",        # 若无，用 updatetime / tmstamp
    "out_date": "OUT_DATE",               # optional
    "use_status": "USE_STATUS",           # =1 有效，可缺
    "entrytime": "ENTRYTIME",             # optional
    "updatetime": "UPDATETIME",           # 作为生效时间的后备
    "tmstamp": "TMSTAMP",                 # 整数时间戳（后备）
}
IND_COLMAP: Dict[str, str] = getattr(CFG, "IND_COLMAP", DEFAULT_COLMAP)

# 维护列全集文件
LATEST_COLS_PATH: Path = Path(IND_DIR) / "latest_columns.json"

# ---------------- Helpers: code & name normalization ----------------
_A_SHARE_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ)$", re.IGNORECASE)

def _norm_code(c: str) -> Optional[str]:
    """Normalize to '######.(SH|SZ)'. Accept bare 6-digit (infer suffix)."""
    if c is None:
        return None
    s = str(c).strip().upper()
    # 支持 'SHSE.600000' / 'SZSE.000001'
    if "." in s and (s.startswith("SHSE.") or s.startswith("SZSE.")):
        ex, num = s.split(".", 1)
        suf = "SH" if ex.startswith("SH") else "SZ"
        s = f"{num}.{suf}"
    if _A_SHARE_CODE_RE.match(s):
        a, b = s.split(".")
        return f"{a.zfill(6)}.{b}"
    if len(s) == 6 and s.isdigit():
        suf = "SH" if s.startswith("6") else "SZ"
        return f"{s}.{suf}"
    if len(s) == 8 and s[:6].isdigit() and s[6:].isalpha():  # e.g., 600000SH
        return f"{s[:6]}.{s[6:].upper()}"
    return None

def _to_halfwidth(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", ch) for ch in s)

def _norm_industry_name(x: str) -> str:
    """
    规则：去两端空白 -> 全角转半角 -> 非 [0-9A-Za-z汉字] 替换为 '_' -> 合并多 '_' -> 去首尾 '_'。
    仅 ASCII 转大写；中文原样保留。
    """
    if x is None:
        return "OTHER"
    s = str(x).strip()
    if not s:
        return "OTHER"
    s = _to_halfwidth(s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "OTHER"
    # 仅把 ASCII 部分 upper（中文不变）
    s = "".join(ch.upper() if ch.isascii() else ch for ch in s)
    return s

def _std_tag(std: str) -> str:
    """Map various Chinese names to canonical 'SW' or 'GICS'."""
    s = (std or "").strip().upper()
    # 常见中文变体
    zh = (std or "").strip()
    if any(k in zh for k in ("申万", "申銀萬國", "申银万国", "申萬", "申銀")):
        return "SW"
    if "GICS" in s:
        return "GICS"
    # 其他标准全部映射到自身大写（但默认我们只筛 SW/GICS）
    return s or "SW"

# ---------------- Helpers: file & columns union ----------------
def _list_constituent_files(glob_pat: str) -> List[Path]:
    """支持绝对或相对模式；返回按文件名中的时间戳排序的候选文件列表。"""
    # glob.glob 直接支持绝对路径
    files = [Path(p) for p in glob.glob(glob_pat)]
    return sorted(files)

def _extract_ts_from_name(p: Path) -> Optional[pd.Timestamp]:
    """
    从文件名 QT_INDUS_CONSTITUENTS_YYYYMMDDHHMM.csv 提取时间戳，缺失则返回 None。
    """
    m = re.search(r"(\d{12,14})", p.name)
    if not m:
        return None
    s = m.group(1)
    # 允许 12 或 14 位
    fmt = "%Y%m%d%H%M" if len(s) == 12 else "%Y%m%d%H%M%S"
    try:
        return pd.to_datetime(s, format=fmt)
    except Exception:
        return None

def _find_best_file(t: pd.Timestamp, glob_pat: str) -> Optional[Path]:
    """
    选择“时间戳 <= t 当天 23:59:59 的最近文件”；若都 > t 或无时间戳，则取最新文件做行级过滤。
    """
    files = _list_constituent_files(glob_pat)
    if not files:
        return None
    t_end = pd.Timestamp(t).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    dated = [(p, _extract_ts_from_name(p)) for p in files]
    # 可用集合：ts <= t_end
    le = [p for (p, ts) in dated if ts is not None and ts <= t_end]
    if le:
        # 取离 t 最近的
        le.sort(key=lambda p: _extract_ts_from_name(p), reverse=True)
        return le[0]
    # 否则取最新文件（后续行级 <= t 过滤）
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

def _stable_unique(seq: List[str]) -> List[str]:
    """稳定去重（保留首次出现的顺序）"""
    return list(dict.fromkeys(seq))

def _load_latest_columns() -> List[str]:
    if not LATEST_COLS_PATH.exists():
        return []
    try:
        with LATEST_COLS_PATH.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            return []
        return _stable_unique([str(x) for x in arr])
    except Exception:
        return []

def _save_latest_columns(cols: List[str]) -> None:
    cols = _stable_unique([str(c) for c in cols])
    ensure_dir(LATEST_COLS_PATH.parent)
    # 使用 name-based lock，保证并发安全
    with with_file_lock("industry_map", timeout_sec=10.0):
        with open(LATEST_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(cols, f, ensure_ascii=False, indent=2, sort_keys=False)

# ---------------- Load constituents (<= t, latest per code) ----------------
@dataclass
class ResolvedCols:
    code: str
    std_name: str
    ind_name: str
    ind_level: str
    eff: Optional[str]
    upd: Optional[str]
    tms: Optional[str]
    use: Optional[str]

def _resolve_columns(df: pd.DataFrame) -> ResolvedCols:
    """根据 IND_COLMAP 容错解析输入列名（支持大小写/不同供应商）。"""
    # 允许用户 map 覆盖；若 map 指到不存在列，下面会用宽松匹配再兜底。
    m = {k: IND_COLMAP.get(k, v) for k, v in DEFAULT_COLMAP.items()}
    # 宽松大小写匹配
    low = {c.lower(): c for c in df.columns}
    def pick(key: str, *alts: str) -> Optional[str]:
        cand = m.get(key)
        for name in ([cand] if cand else []) + list(alts):
            if not name:
                continue
            if name in df.columns:
                return name
            if name.lower() in low:
                return low[name.lower()]
        return None

    return ResolvedCols(
        code=pick("code", "code", "stock_code") or "code",
        std_name=pick("standard_name", "standard_name") or "standard_name",
        ind_name=pick("industry_name", "industry_name") or "industry_name",
        ind_level=pick("industry_level", "industry_level") or "industry_level",
        eff=pick("effective_date", "into_date"),
        upd=pick("updatetime", "update_time", "updatetime"),
        tms=pick("tmstamp", "timestamp", "tmstamp"),
        use=pick("use_status", "use_status"),
    )

def _parse_dt_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(pd.NaT, index=getattr(s, "index", None), name="dt")
    out = pd.to_datetime(s, errors="coerce")
    # 如果是纯数字时间戳（秒/毫秒），也尝试解析
    if out.isna().all():
        try:
            out = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", errors="coerce")
        except Exception:
            pass
    if out.isna().all():
        try:
            out = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="ms", errors="coerce")
        except Exception:
            pass
    return out

def _load_constituents_for_day(t: pd.Timestamp) -> pd.DataFrame:
    """返回满足“标准 & 层级 & 有效 & <=t”的最新记录（每 code 一条）。"""
    path = _find_best_file(t, IND_CONS_GLOB)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"No constituents CSV matched: {IND_CONS_GLOB}")
    df = read_csv_safe(path)  # 不传 index_col（utils.read_csv_safe 不支持该参数）
    if df.empty:
        return df

    cols = _resolve_columns(df)
    # 取必要列，缺失则填空
    use_cols = [c for c in [cols.code, cols.std_name, cols.ind_name, cols.ind_level, cols.eff, cols.upd, cols.tms, cols.use] if c]
    sub = df[use_cols].copy()

    # 标准过滤（默认 SW；可切 GICS）
    std_tag = sub[cols.std_name].map(_std_tag)
    sub = sub.loc[std_tag == IND_STANDARD.upper()]

    # 层级过滤
    try:
        lvl = pd.to_numeric(sub[cols.ind_level], errors="coerce")
    except Exception:
        lvl = pd.Series(pd.NA, index=sub.index)
    sub = sub.loc[lvl == int(IND_LEVEL)]

    # use_status==1（若有）
    if cols.use in sub.columns:
        try:
            sub = sub.loc[pd.to_numeric(sub[cols.use], errors="coerce") == 1]
        except Exception:
            pass

    # 生效时间（无前视）
    eff = _parse_dt_series(sub[cols.eff]) if cols.eff in sub.columns else pd.Series(pd.NaT, index=sub.index)
    upd = _parse_dt_series(sub[cols.upd]) if cols.upd in sub.columns else pd.Series(pd.NaT, index=sub.index)
    tms = _parse_dt_series(sub[cols.tms]) if cols.tms in sub.columns else pd.Series(pd.NaT, index=sub.index)
    # 组合优先级：effective_date > updatetime > tmstamp
    eff_coalesced = eff
    eff_coalesced = eff_coalesced.where(eff_coalesced.notna(), upd)
    eff_coalesced = eff_coalesced.where(eff_coalesced.notna(), tms)
    sub["__eff__"] = eff_coalesced

    # 代码归一
    sub["__code__"] = sub[cols.code].astype(str).map(_norm_code)
    sub["__iname__"] = sub[cols.ind_name].astype(str).map(_norm_industry_name)
    sub = sub.dropna(subset=["__code__"])

    # 仅保留 <= t 的记录
    t_end = pd.Timestamp(t).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    sub = sub.loc[(sub["__eff__"].notna()) & (sub["__eff__"] <= t_end)]

    if sub.empty:
        return pd.DataFrame(columns=["code", "ind_name", "eff"])

    # 对每个 code 取最新一条（最大 __eff__）
    sub = sub.sort_values(["__code__", "__eff__"], kind="mergesort").groupby("__code__", as_index=False).tail(1)
    out = sub.rename(columns={"__code__": "code", "__iname__": "ind_name", "__eff__": "eff"})[["code", "ind_name", "eff"]]
    return out

# ---------------- One-hot & cache ----------------
def _col_name(ind_name: str) -> str:
    return f"IND_{IND_STANDARD.upper()}_L{int(IND_LEVEL)}__{ind_name}"

def _one_hot(codes: List[str], mapping: pd.DataFrame) -> pd.DataFrame:
    """codes 顺序输出；按映射 one-hot；缺失 -> OTHER。"""
    idx = pd.Index([_norm_code(c) for c in codes if _norm_code(c)], name="code")
    mp = mapping.set_index("code")["ind_name"] if not mapping.empty else pd.Series(index=idx, dtype=object)
    mp = mp.reindex(idx)
    # 缺失 -> OTHER
    missing = mp.isna().sum()
    if missing:
        log.warn(f"[IND] missing={int(missing)} -> {_col_name('OTHER')}")
    mp = mp.fillna("OTHER").map(_norm_industry_name)

    cols = [_col_name(x) for x in mp.unique().tolist()]
    cols = _stable_unique(cols)
    # 构造 0/1 矩阵
    mat = {c: (mp.map(lambda x, c=c: 1 if _col_name(x) == c else 0)) for c in cols}
    df = pd.DataFrame(mat, index=idx)
    return df

def _align_with_union(today_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], bool]:
    """用 latest_columns.json 的全集列与今天列做并集，稳定去重并持久化。返回 (对齐后的df, union_cols, appeared_new)."""
    exist = _load_latest_columns()
    today_cols = list(today_df.columns)
    union_cols = _stable_unique(exist + today_cols)
    appeared_new = len(union_cols) > len(exist)

    # 至少保证 OTHER 列存在
    other_col = _col_name("OTHER")
    if other_col not in union_cols:
        union_cols.append(other_col)

    # 用并集对齐；这里的 union_cols 经过稳定去重，不会触发 reindex duplicate
    aligned = today_df.reindex(columns=union_cols, fill_value=0)

    if appeared_new:
        _save_latest_columns(union_cols)
    return aligned, union_cols, appeared_new

def _cache_today_matrix(t: pd.Timestamp, mat: pd.DataFrame) -> Path:
    ensure_dir(IND_DIR)
    ymd = pd.Timestamp(t).strftime("%Y%m%d")
    path = Path(IND_DIR) / f"{ymd}_industry.csv"
    # 保证 index=code；columns=全集
    df = mat.copy()
    df.index.name = "code"
    with with_file_lock("industry_map", timeout_sec=10.0):
        write_csv_atomic(path, df.reset_index(), index=False)
    return path

def _read_today_cache(t: pd.Timestamp) -> pd.DataFrame:
    ymd = pd.Timestamp(t).strftime("%Y%m%d")
    fp = Path(IND_DIR) / f"{ymd}_industry.csv"
    if not fp.exists():
        return pd.DataFrame()
    df = read_csv_safe(fp, parse_dates=False)
    if df.empty:
        return df
    if "code" in df.columns:
        df = df.set_index("code")
    # 去重保护：如果历史写入中过了重复列，这里也会稳定去重一次
    df = df.loc[:, _stable_unique(list(df.columns))]
    return df

# ---------------- Public API ----------------
def industry_dummies(date: pd.Timestamp, codes: List[str]) -> pd.DataFrame:
    """
    Return one-hot industry matrix for given codes on trade day `date`.
    Index = code, Columns = IND_<STD>_L<level>__<NAME> (包含 OTHER).
    - 无前视：仅用 <= t 的行业归属记录。
    - 列全集稳定：与 latest_columns.json 对齐；遇新行业列会自动并入并持久化。
    """
    t = pd.Timestamp(date).normalize()
    codes = [c for c in ([_norm_code(x) for x in codes]) if c]
    log.step(f"[IND] t={t.strftime('%Y%m%d')} provider=static codes_in={len(codes)}")

    # 1) 读当日缓存
    cached = _read_today_cache(t)
    cache_hit = not cached.empty

    if cache_hit:
        mat_full = cached
        union_cols = list(mat_full.columns)
        log.done(f"[IND] snapped={len(mat_full)} uniq_inds={sum(mat_full.sum(axis=0) > 0)} cache=hit")
    else:
        # 2) 从 constituents 回放
        cons = _load_constituents_for_day(t)
        try:
            mat_today = _one_hot(codes, cons)
        except Exception as e:
            # 兜底：全部 OTHER（防止接口中断影响主流程）
            idx = pd.Index(codes, name="code")
            mat_today = pd.DataFrame({_col_name("OTHER"): 1}, index=idx)
            log.warn(f"[IND] one_hot failed; fallback OTHER-only. err={type(e).__name__}: {e}")

        # 与全集列对齐并可能更新 latest_columns.json
        mat_full, union_cols, appeared_new = _align_with_union(mat_today)

        # 缓存当日矩阵（全集列）
        _cache_today_matrix(t, mat_full)
        log.done(f"[IND] snapped={len(mat_full)} uniq_inds={sum(mat_full.sum(axis=0) > 0)} cache=miss "
                 f"{'(new columns merged)' if appeared_new else ''}")

    # 3) 仅返回请求的 codes 子集（保持传入顺序）
    idx = pd.Index(codes, name="code")
    out = mat_full.reindex(index=idx).fillna(0)

    # 对于完全没在 cons 里的 code，确保 OTHER=1
    other_col = _col_name("OTHER")
    if other_col in out.columns:
        need_other = out.sum(axis=1) == 0
        if need_other.any():
            out.loc[need_other, other_col] = 1

    return out

# ---------------- Smoke (optional) ----------------
if __name__ == "__main__":  # pragma: no cover
    log.set_verbosity("STEP")
    today = pd.Timestamp.today().normalize()
    demo_codes = [f"{600000+i:06d}.SH" if i % 2 == 0 else f"{300000+i:06d}.SZ" for i in range(100)]
    try:
        df = industry_dummies(today, demo_codes)
        log.done(f"[SMOKE][IND] shape={df.shape}, cols={len(df.columns)}; OTHER in cols? {any('OTHER' in c for c in df.columns)}")
    except Exception as e:
        # 兜底：即使失败，也不让异常在 smoke 中静默
        idx = pd.Index(demo_codes, name="code")
        df = pd.DataFrame({_col_name("OTHER"): 1}, index=idx)
        log.done(f"[SMOKE][IND] fallback OTHER-only shape={df.shape}; err={e}")
