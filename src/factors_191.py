"""
[CN] 191 因子契约：输出 191 列暴露，允许 NaN；每个因子内部决定最小窗长；当日按 codes 与 panel 对齐。
[Purpose] Compute raw 191-factor exposures prior to orthogonalization.

Signature:
def factor_exposures_191(date: pd.Timestamp, panel: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code with columns f1..f191.
    NaNs allowed (insufficient history); columns fixed and ordered.
    Cleaning: winsorize(WINSOR_PCT) + z-score if STANDARDIZE.
    '''

Notes:
- Factor formulas follow the bank’s 191 definitions (to be implemented).
- Window requirements differ per factor; log effective sample size per day.
"""
from __future__ import annotations
from __future__ import annotations

# file: src/factors_191.py
# -*- coding: utf-8 -*-
"""
[CN] 固化版 191 因子：运行期不读 YAML。提前定义算子库 Ops，再以函数 alphaXX(ops) 写死每个因子。
[契约] 已对照 docs/api_contract.csv：factor_exposures_191(date, panel, codes) 签名一致。
[返回] DataFrame(index=codes, columns=f1..f191, float)，允许 NaN（不清洗）。

设计要点
- 不读取/翻译 YAML；所有因子以 Python 函数形式写死（可逐步补齐）。
- 算子库 Ops 覆盖：逐日/滚动、秩/相关、SMA 三参、REGBETA(SEQUENCE)、逻辑/比较、基准指数缺失一次性 WARN。
- 列级容错：单因子异常不影响其他列；最终 clip_inf_nan。
- 日志：step/loop/error/done；对未实现因子打印 UNIMPLEMENTED 摘要。
"""

from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
import time

# project utils
try:
    if __package__:
        from .utils import logging as log
        from .utils.numerics import clip_inf_nan
        from config import CFG, LOG_VERBOSITY  # type: ignore
    else:
        raise ImportError
except Exception:
    from src.utils import logging as log
    from src.utils.numerics import clip_inf_nan
    from config import CFG, LOG_VERBOSITY  # type: ignore

try:
    log.set_verbosity(LOG_VERBOSITY)
except Exception:
    pass


# =========================
# Runtime operator library
# =========================
class Ops:
    """Operator runtime with wide (date×code) variables and vectorized ops."""
    EPS = 1e-12  # numeric safety for logs/division

    def __init__(self, date: pd.Timestamp, panel: pd.DataFrame, codes: List[str]):
        self._stat_build = 0
        self._stat_hit = 0
        self.date = pd.Timestamp(date).normalize()
        self.codes = list(dict.fromkeys([str(c) for c in codes]))
        self.panel = panel.copy()
        if "date" in self.panel.columns:
            self.panel["date"] = pd.to_datetime(self.panel["date"], errors="coerce").dt.normalize()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._missing: set = set()
        self._bench_warned = False
        # ---- column name resolver (case-insensitive + aliases) ----
        cols_lower = {c.lower(): c for c in self.panel.columns}

        def pick(*names):
            for n in names:
                if n in cols_lower:
                    return cols_lower[n]
            return None

        # 标准名 -> 实际列
        self._colmap = {
            "OPEN": pick("open"),
            "HIGH": pick("high"),
            "LOW": pick("low"),
            "CLOSE": pick("close"),
            "VOLUME": pick("volume", "vol", "volumn"),  # 兼容 volumn 手误
            "AMOUNT": pick("amount", "money"),  # 兼容 money
            "PRECLOSE": pick("preclose", "prev_close", "pre_close"),
            "PAUSED": pick("paused"),
            "VWAP": pick("vwap"),  # 仍有 VWAP 的兜底逻辑（amount/volume 或 (H+L+C)/3）
        }

    # ---------- helpers ----------
    @staticmethod
    def _src(x) -> str:
        """Return a stable source tag for DataFrame/Series/scalars."""
        try:
            s = getattr(x, "attrs", {}).get("src", None)
            if s:
                return str(s)
        except Exception:
            pass
        if hasattr(x, "name") and x.name is not None:
            return f"SERIES:{x.name}"
        if hasattr(x, "columns") and hasattr(x, "shape"):
            return f"DF:{getattr(x, 'shape', None)}"
        return "SCALAR"

    @staticmethod
    def _tag(obj, src: str):
        """Attach a 'src' tag to pandas objects (no-op for scalars)."""
        try:
            if hasattr(obj, "attrs"):
                obj.attrs["src"] = str(src)
        except Exception:
            pass
        return obj
    def _dates_index(self) -> pd.DatetimeIndex:
        if "date" not in self.panel.columns:
            return pd.DatetimeIndex([])
        return (pd.to_datetime(self.panel["date"], errors="coerce")
                .dt.normalize().dropna().drop_duplicates().sort_values())

    # def _wide(self, col: str) -> pd.DataFrame:
    #     key = f"wide::{col}"
    #     if key in self._cache:
    #         return self._cache[key]
    #     need = {"date", "code", col}
    #     if not need.issubset(self.panel.columns):
    #         self._missing.add(col.upper())
    #         df = pd.DataFrame(np.nan, index=self._dates_index(),
    #                           columns=pd.Index(self.codes, name="code"))
    #         self._cache[key] = df
    #         return df
    #     df = self.panel[["date", "code", col]].copy()
    #     df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    #     df = df.dropna(subset=["date", "code"])
    #     wide = df.pivot(index="date", columns="code", values=col).sort_index()
    #     wide = wide.reindex(columns=self.codes)
    #     self._cache[key] = wide
    #     return wide
    def _wide(self, col: str) -> pd.DataFrame:
        key = f"wide::{col}"
        if key in self._cache:
            return self._cache[key]
        need = {"date", "code", col}
        if not need.issubset(self.panel.columns):
            self._missing.add(col.upper())
            df = pd.DataFrame(np.nan, index=self._dates_index(),
                              columns=pd.Index(self.codes, name="code"))
            df.attrs["src"] = f"WIDE:{col}"
            self._cache[key] = df
            return df
        df = self.panel[["date", "code", col]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "code"])
        wide = df.pivot(index="date", columns="code", values=col).sort_index()
        wide = wide.reindex(columns=self.codes)
        # ★ 给宽表打标签，后续 _roll 用它做缓存 key
        wide.attrs["src"] = f"WIDE:{col}"
        self._cache[key] = wide
        return wide

    # --- paste INSIDE class Ops -----------------------------------------------
    def _as_wide(self, x):
        """Coerce scalars/Series/DataFrame into a wide (date x code) DataFrame."""
        idx = self._dates_index()
        cols = pd.Index(self.codes, name="code")
        if isinstance(x, pd.DataFrame):
            return x.reindex(index=idx, columns=cols)
        if isinstance(x, pd.Series):
            # replicate a date-indexed Series across codes
            s = x.reindex(idx)
            df = pd.concat([s] * len(cols), axis=1)
            df.columns = cols
            return df
        # scalar
        return pd.DataFrame(float(x), index=idx, columns=cols)

    def IF_WIDE(self, cond, a, b) -> pd.DataFrame:
        """
        Elementwise if-else on wide frames (keeps time dimension).
        Use this before any rolling TS ops. For final cross-section branching, use IF().
        """
        A = self._as_wide(a)
        B = self._as_wide(b)
        C = self._as_wide(cond).astype(bool)
        return A.where(C, B)

    # --------------------------------------------------------------------------

    def _bench(self, name: str) -> pd.DataFrame:
        """Return NaN-wide for benchmark columns but warn once if used."""
        if not self._bench_warned:
            log.warn(f"[F191] MISSING_BENCH date={self.date.date()}")
            self._bench_warned = True
        return pd.DataFrame(np.nan, index=self._dates_index(),
                            columns=pd.Index(self.codes, name="code"))

    # ---------- exposed variables (wide) ----------
    @property
    def OPEN(self): return self._wide("open")
    @property
    def HIGH(self): return self._wide("high")
    @property
    def LOW(self): return self._wide("low")
    @property
    def CLOSE(self): return self._wide("close")
    @property
    def VWAP(self):
        key = "wide::VWAP"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        vol = self._wide("volume")
        amt = self._wide("amount") if "amount" in self.panel.columns else None
        if amt is not None:
            vwap = Ops._safe_div(amt, vol, name="VWAP_SAFE")  # volume<=0 → NaN
        else:
            high = self._wide("high")
            low = self._wide("low")
            close = self._wide("close")
            vwap = (high + low + close) / 3.0
            vwap = Ops._tag(vwap, "TYPPRICE")
        self._cache[key] = vwap
        return vwap
    @property
    def VOLUME(self): return self._wide("volume")
    @property
    def AMOUNT(self): return self._wide("amount")
    @property
    def PREV_CLOSE(self): return self._wide("preclose")

    # Benchmark (保留历史拼写兼容；两者等价均返回 NaN 并告警一次)
    @property
    def BENCHMARKINDEXOPEN(self): return self._bench("open")
    @property
    def BENCHMARKINDEXHIGH(self): return self._bench("high")
    @property
    def BENCHMARKINDEXLOW(self): return self._bench("low")
    @property
    def BENCHMARKINDEXCLOSE(self): return self._bench("close")
    @property
    def BENCHMARKINDEXVOLUME(self): return self._bench("volume")
    @property
    def BANCHMARKINDEXOPEN(self): return self.BENCHMARKINDEXOPEN
    @property
    def BANCHMARKINDEXHIGH(self): return self.BENCHMARKINDEXHIGH
    @property
    def BANCHMARKINDEXLOW(self): return self.BENCHMARKINDEXLOW
    @property
    def BANCHMARKINDEXCLOSE(self): return self.BENCHMARKINDEXCLOSE
    @property
    def BANCHMARKINDEXVOLUME(self): return self.BENCHMARKINDEXVOLUME

    def SAFE_DIV(self, a, b, name: str | None = None):
        return Ops._safe_div(a, b, name=name)

    @property
    # def RET(self):
    #     return self._safe_div(self.CLOSE, self.DELAY(self.CLOSE, 1)) - 1.0
    def RET(self):
        key = "wide::RET"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        close = self._wide("close")
        prev = self.DELAY(close, 1)  # 注意是 n=1，而不是 n:1
        ret = Ops._safe_div(close, prev, name="RET")  # 用静态方法也行
        ret = ret - 1.0  # 括号外减 1.0
        ret.attrs["src"] = "RET"
        self._cache[key] = ret
        return ret

    # ---------- core numerics ----------
    @staticmethod
    # def _roll(df: pd.DataFrame, n: int, fn: str) -> pd.DataFrame:
    #     r = df.rolling(window=int(n), min_periods=int(n))
    #     return getattr(r, fn)()
    @staticmethod
    def _roll_naive(df: pd.DataFrame, n: int, fn: str) -> pd.DataFrame:
        r = df.rolling(window=int(n), min_periods=int(n))
        return getattr(r, fn)()

    def _roll(self, df: pd.DataFrame, n: int, fn: str) -> pd.DataFrame:
        n = int(n)
        src = getattr(df, "attrs", {}).get("src", f"OBJ:{id(df)}")
        key = f"roll::{src}::{fn}::n={n}"
        cached = self._cache.get(key)
        if cached is not None:
            self._stat_hit += 1
            return cached
        self._stat_build += 1
        out = self._roll_naive(df, n, fn)
        # 继承源标记，便于后续再缓存“滚动后的再滚动”
        out.attrs["src"] = f"ROLL({src},{fn},{n})"
        self._cache[key] = out
        return out

    @staticmethod
    def _safe_div(a, b, name: str | None = None):
        """
        Safe elementwise division with alignment; b==0 -> NaN.
        Also tags result with a stable 'src' so rolling-cache keys are stable.
        """
        try:
            bb = b.where(b != 0)  # DataFrame/Series path
        except Exception:
            bb = b  # scalar/ndarray fallback
        try:
            out = a.divide(bb)  # pandas-align first
        except Exception:
            out = a / bb  # numpy fallback
        sa, sb = Ops._src(a), Ops._src(b)
        tag = name if name else f"DIV({sa},{sb})"
        return Ops._tag(out, tag)


    # inside class Ops
    def BOOL_TO_FLOAT(self, cond) -> pd.DataFrame:
        return self.IF_WIDE(cond, 1.0, 0.0)

    # ---------- time-series operators ----------
    # def DELAY(self, x, n: int = 1): return x.shift(int(n)) if hasattr(x, "shift") else x
    def DELAY(self, x, n: int):
        n = int(n)
        src = getattr(x, "attrs", {}).get("src", f"OBJ:{id(x)}")
        key = f"delay::{src}::n={n}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        y = x.shift(n)
        y.attrs["src"] = f"DELAY({src},{n})"
        self._cache[key] = y
        return y

    def DELTA(self, x, n: int):     return x.diff(int(n)) if hasattr(x, "diff") else x
    def SUM(self, x, n: int):       return self._roll(x, n, "sum")
    def MEAN(self, x, n: int):      return self._roll(x, n, "mean")
    def STD(self, x, n: int = 1):   return self._roll(x, int(n), "std")
    def VAR(self, x, n: int):       return self._roll(x, n, "var")
    def TSMAX(self, x, n: int):     return self._roll(x, n, "max")
    def TSMIN(self, x, n: int):     return self._roll(x, n, "min")
    def SMA(self, x, n: int, m: Optional[int] = None, *_):
        """Two flavors: SMA(x,n)=mean; SMA(x,n,m)=TDX recursive (ewm alpha=m/n, adjust=False)."""
        n = int(n)
        if m is None:
            return self.MEAN(x, n)
        alpha = float(m) / float(n) if n != 0 else 0.0
        return x.ewm(alpha=alpha, adjust=False).mean()
    def WMA(self, x, n: int):       return self.DECAY_LINEAR(x, n)
    def DECAY_LINEAR(self, x, n: int):
        n = int(n)
        if n <= 0 or (hasattr(x, "empty") and x.empty): return x * np.nan
        w = np.arange(1, n + 1, dtype=float); w = w / w.sum()
        return x.rolling(n, min_periods=n).apply(lambda a: float(np.dot(a, w)), raw=True)
    def COUNT(self, cond, n: int):
        if hasattr(cond, "rolling"):
            return cond.rolling(int(n), min_periods=int(n)).sum()
        return cond
    def PROD(self, x, n: int):
        return x.rolling(int(n), min_periods=int(n)).apply(lambda a: float(np.prod(a)), raw=True)
    def SUMIF(self, val, n: int, cond):
        if isinstance(cond, pd.Series):
            cond = pd.concat([cond] * val.shape[1], axis=1); cond.columns = val.columns
        masked = val.where(cond.astype(bool), 0.0)
        return masked.rolling(int(n), min_periods=int(n)).sum()
    def FILTER(self, x, cond):
        if isinstance(cond, pd.Series):
            cond = pd.concat([cond] * x.shape[1], axis=1); cond.columns = x.columns
        return x.where(cond.astype(bool))
    def HIGHDAY(self, x, n: int):
        n = int(n)
        def _pos(a):
            j = int(np.nanargmax(a)) if np.isfinite(a).any() else np.nan
            return float((n - 1) - j) if isinstance(j, int) else np.nan
        return x.rolling(n, min_periods=n).apply(_pos, raw=True)
    def LOWDAY(self, x, n: int):
        n = int(n)
        def _pos(a):
            j = int(np.nanargmin(a)) if np.isfinite(a).any() else np.nan
            return float((n - 1) - j) if isinstance(j, int) else np.nan
        return x.rolling(n, min_periods=n).apply(_pos, raw=True)

    # ---------- corr/cov & regression ----------
    def _cov_corr(self, x, y, n: int, mode: str):
        n = int(n)
        mx = x.rolling(n, min_periods=n).mean(); my = y.rolling(n, min_periods=n).mean()
        cov = (x * y).rolling(n, min_periods=n).mean() - mx * my
        if mode == "cov": return cov
        vx = (x * x).rolling(n, min_periods=n).mean() - mx * mx
        vy = (y * y).rolling(n, min_periods=n).mean() - my * my
        return self._safe_div(cov, np.sqrt(vx * vy))
    # def COV(self, x, y, n: int):  return self._cov_corr(x, y, n, "cov")
    # def CORR(self, x, y, n: int): return self._cov_corr(x, y, n, "corr")
    def CORR(self, x, y, n: int):
        n = int(n)
        xs = getattr(x, "attrs", {}).get("src", f"OBJ:{id(x)}")
        ys = getattr(y, "attrs", {}).get("src", f"OBJ:{id(y)}")
        key = f"corr::{xs}::{ys}::n={n}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._cov_corr(x, y, n, "corr")  # 你原来的实现
        out.attrs["src"] = f"CORR({xs},{ys},{n})"
        self._cache[key] = out
        return out

    def COV(self, x, y, n: int):
        n = int(n)
        xs = getattr(x, "attrs", {}).get("src", f"OBJ:{id(x)}")
        ys = getattr(y, "attrs", {}).get("src", f"OBJ:{id(y)}")
        key = f"cov::{xs}::{ys}::n={n}"
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._cov_corr(x, y, n, 'cov')
        out.attrs["src"] = f"COV({xs},{ys},{n})"
        self._cache[key] = out
        return out

    def REGBETA(self, y, x, n: Optional[int] = None):
        """REGBETA(y, SEQUENCE, n) or REGBETA(y, SEQUENCE(n))."""
        if isinstance(x, tuple) and len(x) == 2 and str(x[0]).upper() == "SEQUENCE":
            n = int(x[1])
        elif isinstance(x, str) and str(x).upper() == "SEQUENCE":
            if n is None: raise ValueError("REGBETA(y, SEQUENCE, n) requires n")
            n = int(n)
        elif hasattr(x, "rolling"):
            return self._safe_div(self.COV(y, x, int(n or 2)), self.VAR(x, int(n or 2)))
        if n is None or int(n) <= 1: return y * np.nan
        n = int(n)
        t = np.arange(1, n + 1, dtype=float); t_mean = t.mean()
        denom = (t * t).sum() - n * (t_mean ** 2)
        if np.isclose(denom, 0.0): return y * np.nan
        def _slope(arr):
            y_mean = float(np.nanmean(arr)); ty = float(np.nansum(t * arr))
            return (ty - n * t_mean * y_mean) / denom
        return y.rolling(window=n, min_periods=n).apply(_slope, raw=True)

    def SEQUENCE(self, n: int) -> Tuple[str, int]: return ("SEQUENCE", int(n))

    # ---------- cross-section & elementwise ----------
    def CS(self, x: Union[pd.Series, pd.DataFrame, float, int]) -> pd.Series:
        """Pick today's cross-section; align to codes."""
        idx = pd.Index(self.codes, name="code")
        if isinstance(x, pd.DataFrame):
            if x.empty: return pd.Series(np.nan, index=idx)
            if self.date in x.index: s = x.loc[self.date]
            else:
                part = x.loc[: self.date]
                s = part.tail(1).T.squeeze() if not part.empty else pd.Series(np.nan, index=x.columns)
            return pd.to_numeric(s.reindex(idx), errors="coerce")
        if isinstance(x, pd.Series):
            return pd.to_numeric(x.reindex(idx), errors="coerce")
        return pd.Series(float(x), index=idx)

    def RANK(self, x) -> pd.Series:
        """Cross-sectional rank per day (sort across stocks) at self.date."""
        return self.CS(x).rank(pct=True, method="average")

    def TSRANK(self, x: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Time-series rolling rank of "today vs history" per code, returning wide DataFrame.
        Later caller should take CS(...) to get today's cross-section.
        """
        n = int(n)
        def _rank_last(arr: np.ndarray) -> float:
            if not np.isfinite(arr).any(): return np.nan
            last = arr[-1]
            return float((np.sum(arr <= last)) / len(arr))
        return x.rolling(n, min_periods=n).apply(_rank_last, raw=True)

    def IF(self, cond, a, b) -> pd.Series:
        c = self.CS(cond).astype(bool); sa = self.CS(a); sb = self.CS(b)
        return pd.Series(np.where(c, sa, sb), index=sa.index)

    def LOG(self, x):
        """Elementwise natural log with domain guard: log(max(x, EPS))."""
        return np.log(np.clip(x, self.EPS, None))

    def XRANK(self, x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank per day on a wide DataFrame."""
        return x.rank(axis=1, pct=True)

    def SIGN(self, x):
        # 避免对 bool 调 astype 的告警，统一用 IF_WIDE 做布尔→浮点
        pos = self.IF_WIDE(x > 0, 1.0, 0.0)
        neg = self.IF_WIDE(x < 0, 1.0, 0.0)
        return pos - neg

    # ---------- diagnostics ----------
    @property
    def missing_columns(self) -> List[str]:
        return sorted(list(self._missing))

    def DECAYLINEAR(self, x, n: int):   # alias -> existing DECAY_LINEAR
        return self.DECAY_LINEAR(x, int(n))

    def COVIANCE(self, x, y, n: int):   # alias -> existing COV
        return self.COV(x, y, int(n))

    def sdiv(self, a, b):               # alias -> existing _safe_div
        return self._safe_div(a, b)



# =========================
# Alpha functions (1..9)
# =========================
Series = pd.Series

def alpha1(ops: Ops) -> Series:
    """
    Alpha1:
        (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),
                   RANK(((CLOSE-OPEN)/OPEN)), 6))
    Steps:
      A) cross-sectional ranks per day on dlogV and intraday return
      B) 6-day rolling correlation per code
      C) today's cross-section * (-1)
    """
    vol = ops.VOLUME
    close = ops.CLOSE
    open_ = ops.OPEN

    dlogv = ops.DELTA(ops.LOG(vol), 1)
    rank_a = ops.XRANK(dlogv)
    intraday_ret = ops._safe_div(close - open_, open_)
    rank_b = ops.XRANK(intraday_ret)

    corr6 = ops.CORR(rank_a, rank_b, 6)
    return -1.0 * ops.CS(corr6)


def alpha2(ops: Ops) -> Series:
    """
    Alpha2:
        (-1 * DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)), 1))
    """
    high, low, close = ops.HIGH, ops.LOW, ops.CLOSE
    num = (close - low) - (high - close)
    den = ops._safe_div(1.0, (high - low))
    ratio = num * den
    delta1 = ops.DELTA(ratio, 1)
    return -1.0 * ops.CS(delta1)


def alpha3(ops: Ops) -> pd.Series:
    """
    Alpha3:
        SUM((CLOSE==DELAY(CLOSE,1)
             ? 0
             : CLOSE - (CLOSE>DELAY(CLOSE,1)
                        ? MIN(LOW,DELAY(CLOSE,1))
                        : MAX(HIGH,DELAY(CLOSE,1)))),
            6)
    Implementation notes:
      - Avoid ops.IF here because ops.IF collapses to today's cross-section.
      - Use elementwise pandas .where to keep everything as wide (date x code),
        then take CS() only at the end.
    """
    close = ops.CLOSE
    high, low = ops.HIGH, ops.LOW
    c1 = ops.DELAY(close, 1)

    # Elementwise MIN/ MAX without collapsing shape
    min_low_c1  = low.where(low <= c1, c1)    # = MIN(low, c1)
    max_high_c1 = high.where(high >= c1, c1)  # = MAX(high, c1)

    # inner = IF(close > c1, MIN(low, c1), MAX(high, c1))
    inner = min_low_c1.where(close > c1, max_high_c1)

    # term = IF(close == c1, 0, close - inner)
    term = (close - inner).where(close != c1, 0.0)

    sum6 = ops.SUM(term, 6)  # rolling over time, min_periods=6
    return ops.CS(sum6)



def alpha4(ops: Ops) -> pd.Series:
    """
    Alpha4:
        ((((SUM(CLOSE,8)/8)+STD(CLOSE,8)) < (SUM(CLOSE,2)/2)) ? (-1)
          : (((SUM(CLOSE,2)/2) < ((SUM(CLOSE,8)/8) - STD(CLOSE,8))) ? 1
             : (((1 < (VOLUME/MEAN(VOLUME,20))) || ((VOLUME/MEAN(VOLUME,20)) == 1)) ? 1 : -1)))
    Clean implementation:
      - Use MEAN(CLOSE, n) instead of SUM/8, SUM/2 to avoid scalar-div warnings.
      - Use r_vol = VOLUME / MEAN(VOLUME,20); condC becomes (r_vol >= 1.0).
      - CORRECT: all ops are right-aligned to `ops.date` with min_periods=window.
    """
    close, vol = ops.CLOSE, ops.VOLUME

    ma8  = ops.MEAN(close, 8)          # mean over 8
    std8 = ops.STD(close, 8)
    ma2  = ops.MEAN(close, 2)          # mean over 2
    r_vol = ops._safe_div(vol, ops.MEAN(vol, 20))

    condA = (ma8 + std8) < ma2
    condB = ma2 < (ma8 - std8)
    condC = (r_vol >= 1.0)

    # Nested IFs; CS() is applied inside IF implementation
    out = ops.IF(condA, -1.0,
                 ops.IF(condB,  1.0,
                        ops.IF(condC, 1.0, -1.0)))
    return out



def alpha5(ops: Ops) -> Series:
    """
    Alpha5:
        (-1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3))
    Note:
        - TSRANK returns a wide DataFrame (time-series), not CS.
        - Then CORR(window=5) per code, then TSMAX over 3, then take CS and negate.
    """
    r_vol = ops.TSRANK(ops.VOLUME, 5)
    r_high = ops.TSRANK(ops.HIGH, 5)
    corr5 = ops.CORR(r_vol, r_high, 5)
    tsmax3 = ops.TSMAX(corr5, 3)
    return -1.0 * ops.CS(tsmax3)


def alpha6(ops: Ops) -> Series:
    """
    Alpha6:
        (RANK(SIGN(DELTA(((OPEN*0.85)+(HIGH*0.15)), 4))) * -1)
    """
    x = ops.DELTA(ops.OPEN * 0.85 + ops.HIGH * 0.15, 4)
    s = np.sign(x)
    return -1.0 * ops.RANK(s)


def alpha7(ops: Ops) -> Series:
    """
    Alpha7:
        ((RANK(MAX((VWAP-CLOSE),3)) + RANK(MIN((VWAP-CLOSE),3))) * RANK(DELTA(VOLUME,3)))
    """
    diff = ops.VWAP - ops.CLOSE
    r1 = ops.RANK(ops.TSMAX(diff, 3))
    r2 = ops.RANK(ops.TSMIN(diff, 3))
    r3 = ops.RANK(ops.DELTA(ops.VOLUME, 3))
    return (r1 + r2) * r3


def alpha8(ops: Ops) -> Series:
    """
    Alpha8:
        RANK( DELTA( ((HIGH+LOW)/2 * 0.2 + VWAP*0.8), 4) * -1 )
    """
    w = ((ops.HIGH + ops.LOW) * 0.5) * 0.2 + ops.VWAP * 0.8
    val = -1.0 * ops.DELTA(w, 4)
    return ops.RANK(val)


def alpha9(ops: Ops) -> Series:
    """
    Alpha9:
        SMA( ((HIGH+LOW)/2 - (DELAY(HIGH,1)+DELAY(LOW,1))/2) * (HIGH-LOW) / VOLUME , 7, 2 )
    """
    high, low, vol = ops.HIGH, ops.LOW, ops.VOLUME
    prev_h = ops.DELAY(high, 1)
    prev_l = ops.DELAY(low, 1)
    mid_diff = ((high + low) * 0.5) - ((prev_h + prev_l) * 0.5)
    rng = high - low
    core = ops._safe_div(mid_diff * rng, vol)
    sm = ops.SMA(core, 7, 2)
    return ops.CS(sm)

def alpha11(ops: Ops) -> pd.Series:
    """
    SUM( ((CLOSE-LOW) - (HIGH-CLOSE)) / (HIGH-LOW) * VOLUME , 6 )
    """
    high, low, close, vol = ops.HIGH, ops.LOW, ops.CLOSE, ops.VOLUME
    num = (close - low) - (high - close)
    den = (high - low)
    core = ops._safe_div(num, den) * vol
    return ops.CS(ops.SUM(core, 6))


def alpha12(ops: Ops) -> pd.Series:
    """
    RANK(OPEN - MEAN(VWAP,10)) * (-1 * RANK(ABS(CLOSE - VWAP)))
    """
    part1 = ops.RANK(ops.OPEN - ops.MEAN(ops.VWAP, 10))
    part2 = -1.0 * ops.RANK(np.abs(ops.CLOSE - ops.VWAP))
    return part1 * part2


def alpha13(ops: Ops) -> pd.Series:
    """
    ( (HIGH*LOW)^(0.5) - VWAP )
    """
    prod = ops.HIGH * ops.LOW
    sqrt_hl = np.sqrt(np.clip(prod, 0.0, None))
    return ops.CS(sqrt_hl - ops.VWAP)


def alpha14(ops: Ops) -> pd.Series:
    """
    CLOSE - DELAY(CLOSE,5)
    """
    return ops.CS(ops.CLOSE - ops.DELAY(ops.CLOSE, 5))


def alpha15(ops: Ops) -> pd.Series:
    """
    OPEN / DELAY(CLOSE,1) - 1
    """
    return ops.CS(ops._safe_div(ops.OPEN, ops.DELAY(ops.CLOSE, 1)) - 1.0)


def alpha16(ops: Ops) -> pd.Series:
    """
    Alpha16 (robust):
        -1 * TSMAX( RANK( CORR(RANK(VOLUME), RANK(VWAP), 5) ), 5 )

    Rationale:
      - If a 5-day window has zero variance on either side (very common when XRANK
        is flat), Pearson corr is undefined. We set corr=0 on those windows to avoid
        all-NaN collapse, while keeping 'not-ready' windows (min_periods<5) as NaN.
    """
    # cross-sectional ranks per day (keep time dimension!)
    r_vol  = ops.XRANK(ops.VOLUME)   # wide: date x code
    r_vwap = ops.XRANK(ops.VWAP)     # wide: date x code

    n = 5
    mx = r_vol.rolling(n, min_periods=n).mean()
    my = r_vwap.rolling(n, min_periods=n).mean()
    cov = (r_vol * r_vwap).rolling(n, min_periods=n).mean() - mx * my
    vx  = (r_vol * r_vol).rolling(n, min_periods=n).mean() - mx * mx
    vy  = (r_vwap * r_vwap).rolling(n, min_periods=n).mean() - my * my
    den = np.sqrt(vx * vy)

    # corr5: default 0, but only fill ratio where denominator is valid
    corr5 = pd.DataFrame(0.0, index=cov.index, columns=cov.columns)
    ready = mx.notna() & my.notna()                  # min_periods satisfied
    valid = (den > ops.EPS) & ready & np.isfinite(den)
    corr5 = corr5.where(~valid, cov.where(valid) / den.where(valid))

    # daily cross-sectional rank -> 5-day max -> today's cross-section -> negate
    day_rank = ops.XRANK(corr5)
    return -1.0 * ops.CS(ops.TSMAX(day_rank, 5))




def alpha17(ops: Ops) -> pd.Series:
    """
    RANK(VWAP - TSMAX(VWAP,15)) ^ DELTA(CLOSE,5)
    (原式 MAX(VWAP,15) 按时序极大理解为 TSMAX)
    """
    base = ops.RANK(ops.VWAP - ops.TSMAX(ops.VWAP, 15))  # Series in [0,1]
    exp  = ops.CS(ops.DELTA(ops.CLOSE, 5))               # Series
    return pd.Series(np.power(np.clip(base, ops.EPS, None), exp), index=base.index)


def alpha18(ops: Ops) -> pd.Series:
    """
    CLOSE / DELAY(CLOSE,5)
    """
    return ops.CS(ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 5)))


def alpha19(ops: Ops) -> pd.Series:
    """
    IF(CLOSE < DELAY(CLOSE,5), (CLOSE-C5)/C5,
       IF(CLOSE == C5, 0, (CLOSE-C5)/CLOSE))
    —— 用宽表 where，保持时序维度，最后再 CS。
    """
    c  = ops.CLOSE
    c5 = ops.DELAY(c, 5)
    up  = ops._safe_div(c - c5, c5)
    alt = ops._safe_div(c - c5, c)
    wide = up.where(c < c5, alt.where(c != c5, 0.0))
    return ops.CS(wide)


def alpha20(ops: Ops) -> pd.Series:
    """
    (CLOSE - DELAY(CLOSE,6)) / DELAY(CLOSE,6) * 100
    """
    c6 = ops.DELAY(ops.CLOSE, 6)
    return ops.CS(ops._safe_div(ops.CLOSE - c6, c6) * 100.0)


def alpha21(ops: Ops) -> pd.Series:
    """
    REGBETA(MEAN(CLOSE,6), SEQUENCE(6), 6)
    """
    y = ops.MEAN(ops.CLOSE, 6)
    b = ops.REGBETA(y, ops.SEQUENCE(6))
    return ops.CS(b)


def alpha22(ops: Ops) -> pd.Series:
    """
    SMEAN( ((CLOSE-mean6)/mean6 - DELAY((CLOSE-mean6)/mean6, 3)), 12, 1 )
    —— SMEAN 按 SMA 处理（通达信三参口径）。
    """
    mean6 = ops.MEAN(ops.CLOSE, 6)
    x = ops._safe_div(ops.CLOSE - mean6, mean6)
    delta3 = x - ops.DELAY(x, 3)
    return ops.CS(ops.SMA(delta3, 12, 1))


def alpha23(ops: Ops) -> pd.Series:
    """
    SMA( (CLOSE>DELAY(CLOSE,1)? STD(CLOSE,20) : 0), 20, 1 )
    / ( SMA( cond?STD:0, 20, 1 ) + SMA( (!cond)?STD:0, 20, 1) ) * 100
    —— 用 IF_WIDE 保持时序维度，再做 SMA。
    """
    c = ops.CLOSE
    cond = c > ops.DELAY(c, 1)         # wide bool
    std20 = ops.STD(c, 20)             # wide

    pos = ops.SMA(ops.IF_WIDE(cond,  std20, 0.0), 20, 1)
    neg = ops.SMA(ops.IF_WIDE(~cond, std20, 0.0), 20, 1)

    ratio = ops._safe_div(pos, (pos + neg)) * 100.0
    return ops.CS(ratio)


def alpha24(ops: Ops) -> pd.Series:
    """
    SMA( CLOSE - DELAY(CLOSE,5), 5, 1 )
    """
    core = ops.CLOSE - ops.DELAY(ops.CLOSE, 5)
    return ops.CS(ops.SMA(core, 5, 1))


def alpha25(ops: Ops) -> pd.Series:
    """
    (-1 * RANK( DELTA(CLOSE,7) * (1 - RANK(DECAY_LINEAR(VOLUME/MEAN(VOLUME,20), 9))) ))
    * (1 + RANK(SUM(RET,250)))
    """
    # Part A
    delta7 = ops.CS(ops.DELTA(ops.CLOSE, 7))         # Series
    r_dec  = ops.RANK(ops.DECAY_LINEAR(ops._safe_div(ops.VOLUME, ops.MEAN(ops.VOLUME, 20)), 9))  # Series
    partA  = -1.0 * ops.RANK(delta7 * (1.0 - r_dec))

    # Part B
    sum_ret_250 = ops.SUM(ops.RET, 250)              # wide
    partB = (1.0 + ops.RANK(sum_ret_250))            # Series

    return partA * partB


def alpha26(ops: Ops) -> pd.Series:
    """
    ( (SUM(CLOSE,7)/7 - CLOSE) + CORR(VWAP, DELAY(CLOSE,5), 230) )
    """
    mean7 = ops.MEAN(ops.CLOSE, 7)
    corr = ops.CORR(ops.VWAP, ops.DELAY(ops.CLOSE, 5), 230)
    return ops.CS((mean7 - ops.CLOSE) + corr)


def alpha27(ops: Ops) -> pd.Series:
    """
    WMA( (CLOSE-C3)/C3*100 + (CLOSE-C6)/C6*100 , 12 )
    """
    c = ops.CLOSE
    c3 = ops.DELAY(c, 3)
    c6 = ops.DELAY(c, 6)
    p1 = ops._safe_div(c - c3, c3) * 100.0
    p2 = ops._safe_div(c - c6, c6) * 100.0
    return ops.CS(ops.WMA(p1 + p2, 12))


def alpha28(ops: Ops) -> pd.Series:
    """
    3*SMA(RSV,3,1) - 2*SMA(SMA(RSV,3,1),3,1)
    where RSV = (CLOSE - TSMIN(LOW,9)) / (TSMAX(HIGH,9) - TSMIN(LOW,9)) * 100
    （原式里 MAX/TSMAX 混写，这里按 KDJ 标准口径实现）
    """
    low_min9  = ops.TSMIN(ops.LOW, 9)
    high_max9 = ops.TSMAX(ops.HIGH, 9)
    rsv = ops._safe_div(ops.CLOSE - low_min9, (high_max9 - low_min9)) * 100.0
    k = ops.SMA(rsv, 3, 1)
    d = ops.SMA(k,   3, 1)
    j = 3.0 * k - 2.0 * d
    return ops.CS(j)


def alpha29(ops: Ops) -> pd.Series:
    """
    (CLOSE - DELAY(CLOSE,6)) / DELAY(CLOSE,6) * VOLUME
    """
    c = ops.CLOSE
    c6 = ops.DELAY(c, 6)
    core = ops._safe_div(c - c6, c6) * ops.VOLUME
    return ops.CS(core)

# --- Alpha31 ~ Alpha41 -----------------------------------------------------

def alpha31(ops: Ops) -> pd.Series:
    """
    (CLOSE - MEAN(CLOSE,12)) / MEAN(CLOSE,12) * 100
    """
    mean12 = ops.MEAN(ops.CLOSE, 12)
    val = ops._safe_div(ops.CLOSE - mean12, mean12) * 100.0
    return ops.CS(val)


def alpha32(ops: Ops) -> pd.Series:
    """
    Alpha32 (robust):
        - SUM( RANK( CORR( RANK(HIGH), RANK(VOLUME), 3 ) ), 3 )

    Notes:
      - Use XRANK (cross-sectional per day, keep time dimension).
      - Pearson corr is undefined when either side has zero variance in a 3-day window.
        We set corr=0 on those windows to avoid all-NaN collapse; min_periods logic kept.
    """
    r_high = ops.XRANK(ops.HIGH)     # wide (date x code)
    r_vol  = ops.XRANK(ops.VOLUME)   # wide

    n = 3
    mx = r_high.rolling(n, min_periods=n).mean()
    my = r_vol.rolling(n, min_periods=n).mean()
    cov = (r_high * r_vol).rolling(n, min_periods=n).mean() - mx * my
    vx  = (r_high * r_high).rolling(n, min_periods=n).mean() - mx * mx
    vy  = (r_vol  * r_vol ).rolling(n, min_periods=n).mean() - my * my
    den = np.sqrt(vx * vy)

    # corr3: default 0; only fill cov/den where denominator is valid
    corr3 = pd.DataFrame(0.0, index=cov.index, columns=cov.columns)
    ready = mx.notna() & my.notna()
    valid = (den > ops.EPS) & ready & np.isfinite(den)
    corr3 = corr3.where(~valid, cov.where(valid) / den.where(valid))

    # per-day cross-sectional rank, then sum over last 3 days, then today's CS, negate
    day_rank = ops.XRANK(corr3)      # wide
    sum3 = ops.SUM(day_rank, 3)
    return -1.0 * ops.CS(sum3)



def alpha33(ops: Ops) -> pd.Series:
    """
    (((-1 * TSMIN(LOW,5)) + DELAY(TSMIN(LOW,5),5))
     * RANK( (SUM(RET,240) - SUM(RET,20)) / 220 )
     * TSRANK(VOLUME,5))
    """
    tmin5 = ops.TSMIN(ops.LOW, 5)
    diff = (-1.0 * tmin5) + ops.DELAY(tmin5, 5)   # wide
    # rank term (today's cross-sectional rank)
    r_term = ops.RANK(ops._safe_div(ops.SUM(ops.RET, 240) - ops.SUM(ops.RET, 20), 220.0))
    # tsrank term (today only)
    tsr_v = ops.CS(ops.TSRANK(ops.VOLUME, 5))
    out = diff.mul(r_term, axis=1).mul(tsr_v, axis=1)
    return ops.CS(out)


def alpha34(ops: Ops) -> pd.Series:
    """
    MEAN(CLOSE,12) / CLOSE
    """
    return ops.CS(ops._safe_div(ops.MEAN(ops.CLOSE, 12), ops.CLOSE))


def alpha35(ops: Ops) -> pd.Series:
    """
    MIN( RANK(DECAY_LINEAR(DELTA(OPEN,1),15)),
         RANK(DECAY_LINEAR(CORR(VOLUME, OPEN,17),7)) ) * -1
    Note: (OPEN*0.65 + OPEN*0.35) == OPEN
    """
    a = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(ops.OPEN, 1), 15))
    b = ops.RANK(ops.DECAY_LINEAR(ops.CORR(ops.VOLUME, ops.OPEN, 17), 7))
    return -1.0 * pd.Series(np.minimum(a, b), index=a.index)


def alpha37(ops: Ops) -> pd.Series:
    """
    - RANK( (SUM(OPEN,5) * SUM(RET,5)) - DELAY(SUM(OPEN,5) * SUM(RET,5), 10) )
    """
    s = ops.SUM(ops.OPEN, 5) * ops.SUM(ops.RET, 5)
    diff = s - ops.DELAY(s, 10)
    return -1.0 * ops.RANK(diff)


def alpha38(ops: Ops) -> pd.Series:
    """
    IF( MEAN(HIGH,20) < HIGH,  -DELTA(HIGH,2),  0 )
    (原式 SUM(HIGH,20)/20 等价于 MEAN(HIGH,20))
    """
    cond = ops.MEAN(ops.HIGH, 20) < ops.HIGH   # wide bool
    val  = ops.IF_WIDE(cond, -ops.DELTA(ops.HIGH, 2), 0.0)  # keep time dimension
    return ops.CS(val)


def alpha39(ops: Ops) -> pd.Series:
    """
    ( RANK(DECAY_LINEAR(DELTA(CLOSE,2),8))
      - RANK(DECAY_LINEAR(CORR( 0.3*VWAP+0.7*OPEN,  SUM(MEAN(VOLUME,180),37), 14 ), 12)) ) * -1
    """
    left  = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(ops.CLOSE, 2), 8))
    wprice = ops.VWAP * 0.3 + ops.OPEN * 0.7
    vol_smooth = ops.SUM(ops.MEAN(ops.VOLUME, 180), 37)
    corr14 = ops.CORR(wprice, vol_smooth, 14)
    right = ops.RANK(ops.DECAY_LINEAR(corr14, 12))
    return -1.0 * (left - right)


def alpha40(ops: Ops) -> pd.Series:
    """
    SUM( (CLOSE>DELAY(CLOSE,1)? VOLUME : 0), 26 )
    / SUM( (CLOSE<=DELAY(CLOSE,1)? VOLUME : 0), 26 ) * 100
    """
    c = ops.CLOSE
    up_vol   = ops.IF_WIDE(c > ops.DELAY(c, 1),  ops.VOLUME, 0.0)
    down_vol = ops.IF_WIDE(c <= ops.DELAY(c, 1), ops.VOLUME, 0.0)
    num = ops.SUM(up_vol, 26)
    den = ops.SUM(down_vol, 26)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha41(ops: Ops) -> pd.Series:
    """
    - RANK( TSMAX( DELTA(VWAP,3), 5 ) )
    """
    mx = ops.TSMAX(ops.DELTA(ops.VWAP, 3), 5)
    return -1.0 * ops.RANK(mx)

# --- Alpha42 ~ Alpha60 ------------------------------------------------------

def alpha42(ops: Ops) -> pd.Series:
    """
    (-1 * RANK(STD(HIGH,10))) * CORR(HIGH, VOLUME, 10)
    """
    left  = -1.0 * ops.RANK(ops.STD(ops.HIGH, 10))           # Series
    right = ops.CS(ops.CORR(ops.HIGH, ops.VOLUME, 10))       # Series
    return left * right


def alpha43(ops: Ops) -> pd.Series:
    """
    SUM( CLOSE>DELAY(CLOSE,1)?VOLUME : (CLOSE<DELAY(CLOSE,1)? -VOLUME : 0), 6 )
    """
    c = ops.CLOSE; c1 = ops.DELAY(c, 1)
    term = ops.IF_WIDE(c > c1,  ops.VOLUME,
                       ops.IF_WIDE(c < c1, -ops.VOLUME, 0.0))
    return ops.CS(ops.SUM(term, 6))


def alpha44(ops: Ops) -> pd.Series:
    """
    TSRANK( DECAY_LINEAR( CORR(LOW, MEAN(VOLUME,10), 7 ), 6 ), 4 )
    + TSRANK( DECAY_LINEAR( DELTA(VWAP,3), 10 ), 15 )
    """
    part1 = ops.TSRANK(ops.DECAY_LINEAR(ops.CORR(ops.LOW, ops.MEAN(ops.VOLUME, 10), 7), 6), 4)
    part2 = ops.TSRANK(ops.DECAY_LINEAR(ops.DELTA(ops.VWAP, 3), 10), 15)
    return ops.CS(part1) + ops.CS(part2)


def alpha45(ops: Ops) -> pd.Series:
    """
    RANK( DELTA( 0.6*CLOSE + 0.4*OPEN, 1 ) ) * RANK( CORR(VWAP, MEAN(VOLUME,150), 15) )
    """
    left  = ops.RANK(ops.DELTA(ops.CLOSE*0.6 + ops.OPEN*0.4, 1))
    right = ops.RANK(ops.CORR(ops.VWAP, ops.MEAN(ops.VOLUME, 150), 15))
    return left * right


def alpha46(ops: Ops) -> pd.Series:
    """
    (MEAN(CLOSE,3) + MEAN(CLOSE,6) + MEAN(CLOSE,12) + MEAN(CLOSE,24)) / (4*CLOSE)
    """
    m = ops.MEAN
    num = m(ops.CLOSE, 3) + m(ops.CLOSE, 6) + m(ops.CLOSE, 12) + m(ops.CLOSE, 24)
    return ops.CS(num / (4.0 * ops.CLOSE))


def alpha47(ops: Ops) -> pd.Series:
    """
    SMA( (TSMAX(HIGH,6) - CLOSE) / (TSMAX(HIGH,6) - TSMIN(LOW,6)) * 100 , 9, 1 )
    """
    h6 = ops.TSMAX(ops.HIGH, 6)
    l6 = ops.TSMIN(ops.LOW, 6)
    core = ops._safe_div(h6 - ops.CLOSE, (h6 - l6)) * 100.0
    return ops.CS(ops.SMA(core, 9, 1))


def alpha48(ops: Ops) -> pd.Series:
    """
    - ( RANK(SIGN(CLOSE-C1) + SIGN(C1-C2) + SIGN(C2-C3)) * SUM(VOLUME,5) ) / SUM(VOLUME,20)
    """
    c  = ops.CLOSE
    c1 = ops.DELAY(c, 1); c2 = ops.DELAY(c, 2); c3 = ops.DELAY(c, 3)
    s = np.sign(c - c1) + np.sign(c1 - c2) + np.sign(c2 - c3)
    rank_s = ops.RANK(s)
    vol_ratio = ops.CS(ops.SUM(ops.VOLUME, 5) / ops.SUM(ops.VOLUME, 20))
    return -1.0 * rank_s * vol_ratio


def alpha49(ops: Ops) -> pd.Series:
    """
    A12 / (A12 + B12)
    where
      A = cond_down? 0 : MAX(ABS(HIGH- H1), ABS(LOW- L1)), cond_down: (H+L)>=(H1+L1)
      B = cond_up  ? 0 : MAX(ABS(HIGH- H1), ABS(LOW- L1)), cond_up  : (H+L)<=(H1+L1)
      H1=DELAY(HIGH,1), L1=DELAY(LOW,1); sums over 12.
    """
    H, L = ops.HIGH, ops.LOW
    H1, L1 = ops.DELAY(H, 1), ops.DELAY(L, 1)
    cond_ge = (H + L) >= (H1 + L1)
    cond_le = (H + L) <= (H1 + L1)
    step = np.maximum((H - H1).abs(), (L - L1).abs())
    A = ops.IF_WIDE(cond_ge, 0.0, step)
    B = ops.IF_WIDE(cond_le, 0.0, step)
    A12 = ops.SUM(A, 12); B12 = ops.SUM(B, 12)
    return ops.CS(ops._safe_div(A12, (A12 + B12)))


def alpha50(ops: Ops) -> pd.Series:
    """
    [A12/(A12+B12)] - [C12/(C12+D12)]
    First bracket uses cond_le (<=); second uses cond_ge (>=), symmetric to Alpha49.
    """
    H, L = ops.HIGH, ops.LOW
    H1, L1 = ops.DELAY(H, 1), ops.DELAY(L, 1)
    cond_ge = (H + L) >= (H1 + L1)
    cond_le = (H + L) <= (H1 + L1)
    step = np.maximum((H - H1).abs(), (L - L1).abs())

    A = ops.IF_WIDE(cond_le, 0.0, step)
    B = ops.IF_WIDE(cond_ge, 0.0, step)
    C = ops.IF_WIDE(cond_ge, 0.0, step)
    D = ops.IF_WIDE(cond_le, 0.0, step)

    A12, B12, C12, D12 = ops.SUM(A,12), ops.SUM(B,12), ops.SUM(C,12), ops.SUM(D,12)
    term1 = ops._safe_div(A12, (A12 + B12))
    term2 = ops._safe_div(C12, (C12 + D12))
    return ops.CS(term1 - term2)


def alpha51(ops: Ops) -> pd.Series:
    """
    Same as the first term of Alpha50:
    A12 / (A12 + B12) with A for cond_le (<=), B for cond_ge (>=).
    """
    H, L = ops.HIGH, ops.LOW
    H1, L1 = ops.DELAY(H, 1), ops.DELAY(L, 1)
    cond_ge = (H + L) >= (H1 + L1)
    cond_le = (H + L) <= (H1 + L1)
    step = np.maximum((H - H1).abs(), (L - L1).abs())

    A = ops.IF_WIDE(cond_le, 0.0, step)
    B = ops.IF_WIDE(cond_ge, 0.0, step)
    A12, B12 = ops.SUM(A, 12), ops.SUM(B, 12)
    return ops.CS(ops._safe_div(A12, (A12 + B12)))


def alpha52(ops: Ops) -> pd.Series:
    """
    SUM( MAX(0, HIGH - DELAY(TP,1)), 26 ) / SUM( MAX(0, DELAY(TP,1) - LOW), 26 ) * 100
    where TP=(HIGH+LOW+CLOSE)/3 ; YAML 'L' is interpreted as LOW.
    """
    TP1 = ops.DELAY((ops.HIGH + ops.LOW + ops.CLOSE) / 3.0, 1)
    up   = np.maximum(0.0, ops.HIGH - TP1)
    down = np.maximum(0.0, TP1 - ops.LOW)
    num = ops.SUM(up, 26); den = ops.SUM(down, 26)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha53(ops: Ops) -> pd.Series:
    """
    COUNT(CLOSE > DELAY(CLOSE,1), 12) / 12 * 100
    """
    c, c1 = ops.CLOSE, ops.DELAY(ops.CLOSE, 1)
    # wide numeric 1/0, no astype needed
    cond = ops.IF_WIDE(c > c1, 1.0, 0.0)
    return ops.CS(ops.SUM(cond, 12) / 12.0 * 100.0)



def alpha54(ops: Ops) -> pd.Series:
    """
    - RANK( STD(ABS(CLOSE-OPEN), win=10) + (CLOSE-OPEN) + CORR(CLOSE, OPEN, 10) )
    Note: YAML omitted the window for STD; we adopt win=10 (practical default).
    """
    diff = ops.CLOSE - ops.OPEN
    std10 = ops.STD(np.abs(diff), 10)
    corr10 = ops.CORR(ops.CLOSE, ops.OPEN, 10)
    return -1.0 * ops.RANK(std10 + diff + corr10)


def alpha55(ops: Ops) -> pd.Series:
    """
    SUM( 16 * N / D * M , 20 )
    where:
      N = CLOSE - C1 + (CLOSE-OPEN)/2 + C1 - O1
      D =
        if A>B and A>C : A + B/2 + |C1-O1|/4
        elif B>A and B>C: B + A/2 + |C1-O1|/4
        else            : C + |C1-O1|/4
      A = |HIGH - C1|
      B = |LOW  - C1|
      C = |HIGH - L1|   (as in YAML; keep literal)
      M = MAX(A, B)
      C1=DELAY(CLOSE,1), O1=DELAY(OPEN,1), L1=DELAY(LOW,1)
    """
    C1 = ops.DELAY(ops.CLOSE, 1); O1 = ops.DELAY(ops.OPEN, 1); L1 = ops.DELAY(ops.LOW, 1)
    A = (ops.HIGH - C1).abs()
    B = (ops.LOW  - C1).abs()
    C = (ops.HIGH - L1).abs()
    M = np.maximum(A, B)
    N = (ops.CLOSE - C1) + (ops.CLOSE - ops.OPEN)/2.0 + (C1 - O1)
    base = (C1 - O1).abs() / 4.0
    D1 = A + B/2.0 + base
    D2 = B + A/2.0 + base
    D3 = C + base
    D = ops.IF_WIDE((A > B) & (A > C), D1,
         ops.IF_WIDE((B > A) & (B > C), D2, D3))
    core = 16.0 * ops._safe_div(N, D) * M
    return ops.CS(ops.SUM(core, 20))


def alpha56(ops: Ops) -> pd.Series:
    """
    ( RANK(OPEN - TSMIN(OPEN,12)) < RANK((RANK(CORR(SUM((HIGH+LOW)/2,19),
                                                       SUM(MEAN(VOLUME,40),19),13))**5)) )
    Boolean -> float {0,1}
    """
    left = ops.RANK(ops.OPEN - ops.TSMIN(ops.OPEN, 12))
    s1 = ops.SUM((ops.HIGH + ops.LOW) / 2.0, 19)
    s2 = ops.SUM(ops.MEAN(ops.VOLUME, 40), 19)
    corr = ops.CORR(s1, s2, 13)
    right = ops.RANK(ops.RANK(corr) ** 5)

    # avoid .astype on a bool Series to please the linter
    # avoid .astype on a bool Series / chained to_numpy on a typed expr
    mask = np.less(left.to_numpy(), right.to_numpy())  # ndarray[bool]
    return pd.Series(mask.astype(float), index=left.index)


def alpha57(ops: Ops) -> pd.Series:
    """
    SMA( (CLOSE - TSMIN(LOW,9)) / (TSMAX(HIGH,9) - TSMIN(LOW,9)) * 100 , 3, 1 )
    """
    low9  = ops.TSMIN(ops.LOW, 9)
    high9 = ops.TSMAX(ops.HIGH, 9)
    rsv = ops._safe_div(ops.CLOSE - low9, (high9 - low9)) * 100.0
    return ops.CS(ops.SMA(rsv, 3, 1))


def alpha58(ops: Ops) -> pd.Series:
    """
    COUNT(CLOSE > DELAY(CLOSE,1), 20) / 20 * 100
    """
    c, c1 = ops.CLOSE, ops.DELAY(ops.CLOSE, 1)
    cond = ops.IF_WIDE(c > c1, 1.0, 0.0)
    return ops.CS(ops.SUM(cond, 20) / 20.0 * 100.0)



def alpha59(ops: Ops) -> pd.Series:
    """
    SUM( CLOSE==C1 ? 0 : CLOSE - (CLOSE>C1 ? MIN(LOW,C1) : MAX(HIGH,C1)) , 20 )
    """
    c = ops.CLOSE; c1 = ops.DELAY(c, 1)
    min_low_c1  = ops.LOW.where(ops.LOW <= c1, c1)
    max_high_c1 = ops.HIGH.where(ops.HIGH >= c1, c1)
    inner = min_low_c1.where(c > c1, max_high_c1)
    term  = (c - inner).where(c != c1, 0.0)
    return ops.CS(ops.SUM(term, 20))


def alpha60(ops: Ops) -> pd.Series:
    """
    SUM( ((CLOSE-LOW) - (HIGH-CLOSE)) / (HIGH-LOW) * VOLUME , 20 )
    """
    num = (ops.CLOSE - ops.LOW) - (ops.HIGH - ops.CLOSE)
    den = (ops.HIGH - ops.LOW)
    core = ops._safe_div(num, den) * ops.VOLUME
    return ops.CS(ops.SUM(core, 20))
# --- Alpha61 ~ Alpha80 ------------------------------------------------------

def alpha61(ops: Ops) -> pd.Series:
    """
    ( MAX( RANK(DECAY_LINEAR(DELTA(VWAP,1),12)),
           RANK(DECAY_LINEAR( XRANK(CORR(LOW, MEAN(VOLUME,80),8)), 17 )) ) * -1 )
    """
    left  = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(ops.VWAP, 1), 12))
    corr8 = ops.CORR(ops.LOW, ops.MEAN(ops.VOLUME, 80), 8)       # wide
    r_corr = ops.XRANK(corr8)                                     # keep time dimension
    right = ops.RANK(ops.DECAY_LINEAR(r_corr, 17))
    return -1.0 * pd.Series(np.maximum(left, right), index=left.index)


def alpha62(ops: Ops) -> pd.Series:
    """
    - CORR(HIGH, RANK(VOLUME), 5)
    Use robust rolling corr (zero-variance window -> corr=0).
    """
    r_vol = ops.XRANK(ops.VOLUME)

    n = 5
    x, y = ops.HIGH, r_vol
    mx = x.rolling(n, min_periods=n).mean()
    my = y.rolling(n, min_periods=n).mean()
    cov = (x * y).rolling(n, min_periods=n).mean() - mx * my
    vx  = (x * x).rolling(n, min_periods=n).mean() - mx * mx
    vy  = (y * y).rolling(n, min_periods=n).mean() - my * my
    den = np.sqrt(vx * vy)

    corr = pd.DataFrame(0.0, index=cov.index, columns=cov.columns)
    ready = mx.notna() & my.notna()
    valid = (den > ops.EPS) & ready & np.isfinite(den)
    corr = corr.where(~valid, cov.where(valid) / den.where(valid))
    return -1.0 * ops.CS(corr)


def alpha63(ops: Ops) -> pd.Series:
    """
    SMA(MAX(CLOSE-DELTA1,0), 6, 1) / SMA(ABS(CLOSE-DELTA1), 6, 1) * 100
    """
    c = ops.CLOSE
    diff = c - ops.DELAY(c, 1)
    num = ops.SMA(np.maximum(diff, 0.0), 6, 1)
    den = ops.SMA(np.abs(diff), 6, 1)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha64(ops: Ops) -> pd.Series:
    """
    ( MAX( RANK(DECAY_LINEAR(CORR(XRANK(VWAP), XRANK(VOLUME), 4), 4)),
           RANK(DECAY_LINEAR( TSMAX(CORR(XRANK(CLOSE), XRANK(MEAN(VOLUME,60)),4), 13), 14)) ) * -1 )
    """
    # left branch
    lv = ops.CORR(ops.XRANK(ops.VWAP), ops.XRANK(ops.VOLUME), 4)
    left = ops.RANK(ops.DECAY_LINEAR(lv, 4))

    # right branch
    rc = ops.CORR(ops.XRANK(ops.CLOSE), ops.XRANK(ops.MEAN(ops.VOLUME, 60)), 4)
    rc_max = ops.TSMAX(rc, 13)
    right = ops.RANK(ops.DECAY_LINEAR(rc_max, 14))

    return -1.0 * pd.Series(np.maximum(left, right), index=left.index)


def alpha65(ops: Ops) -> pd.Series:
    """ MEAN(CLOSE,6) / CLOSE """
    return ops.CS(ops._safe_div(ops.MEAN(ops.CLOSE, 6), ops.CLOSE))


def alpha66(ops: Ops) -> pd.Series:
    """ (CLOSE - MEAN(CLOSE,6)) / MEAN(CLOSE,6) * 100 """
    ma6 = ops.MEAN(ops.CLOSE, 6)
    return ops.CS(ops._safe_div(ops.CLOSE - ma6, ma6) * 100.0)


def alpha67(ops: Ops) -> pd.Series:
    """ SMA(MAX(CLOSE-C1,0),24,1) / SMA(ABS(CLOSE-C1),24,1) * 100 """
    c = ops.CLOSE
    d = c - ops.DELAY(c, 1)
    num = ops.SMA(np.maximum(d, 0.0), 24, 1)
    den = ops.SMA(np.abs(d), 24, 1)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha68(ops: Ops) -> pd.Series:
    """
    SMA( ( ((H+L)/2 - (H1+L1)/2) * (H-L) / VOLUME ), 15, 2 )
    """
    H, L = ops.HIGH, ops.LOW
    H1, L1 = ops.DELAY(H, 1), ops.DELAY(L, 1)
    core = ((H + L) / 2.0 - (H1 + L1) / 2.0) * ops._safe_div((H - L), ops.VOLUME)
    return ops.CS(ops.SMA(core, 15, 2))


def alpha69(ops: Ops) -> pd.Series:
    """
    (SUM(DTM,20) > SUM(DBM,20) ? (SD-SB)/SD : (SD==SB ? 0 : (SD-SB)/SB))
    DTM/DBM definitions (classic):
      if OPEN > O1: DTM = max(HIGH-OPEN, OPEN-O1) else 0
      else if OPEN < O1: DBM = max(OPEN-LOW, O1-OPEN) else 0
    """
    O = ops.OPEN; O1 = ops.DELAY(O, 1)
    H, L = ops.HIGH, ops.LOW

    dtm = ops.IF_WIDE(O > O1, np.maximum(H - O, O - O1), 0.0)
    dbm = ops.IF_WIDE(O < O1, np.maximum(O - L, O1 - O), 0.0)

    SD = ops.SUM(dtm, 20)
    SB = ops.SUM(dbm, 20)

    gt = SD > SB
    eq = (SD == SB)

    term_a = ops._safe_div(SD - SB, SD)
    term_b = ops._safe_div(SD - SB, SB)
    out = ops.IF_WIDE(gt, term_a, ops.IF_WIDE(eq, 0.0, term_b))
    return ops.CS(out)


def alpha70(ops: Ops) -> pd.Series:
    """ STD(AMOUNT, 6) """
    return ops.CS(ops.STD(ops.AMOUNT, 6))


def alpha71(ops: Ops) -> pd.Series:
    """ (CLOSE - MEAN(CLOSE,24)) / MEAN(CLOSE,24) * 100 """
    ma24 = ops.MEAN(ops.CLOSE, 24)
    return ops.CS(ops._safe_div(ops.CLOSE - ma24, ma24) * 100.0)


def alpha72(ops: Ops) -> pd.Series:
    """
    SMA( (TSMAX(HIGH,6) - CLOSE) / (TSMAX(HIGH,6) - TSMIN(LOW,6)) * 100 , 15, 1 )
    """
    h6 = ops.TSMAX(ops.HIGH, 6)
    l6 = ops.TSMIN(ops.LOW, 6)
    rsv = ops._safe_div(h6 - ops.CLOSE, (h6 - l6)) * 100.0
    return ops.CS(ops.SMA(rsv, 15, 1))


def alpha73(ops: Ops) -> pd.Series:
    """
    ( TSRANK( DECAY_LINEAR( DECAY_LINEAR(CORR(CLOSE, VOLUME,10),16),4 ), 5 )
      - RANK( DECAY_LINEAR( CORR(VWAP, MEAN(VOLUME,30),4), 3 ) ) ) * -1
    """
    a = ops.CORR(ops.CLOSE, ops.VOLUME, 10)
    a = ops.DECAY_LINEAR(ops.DECAY_LINEAR(a, 16), 4)
    left = ops.CS(ops.TSRANK(a, 5))

    b = ops.CORR(ops.VWAP, ops.MEAN(ops.VOLUME, 30), 4)
    right = ops.RANK(ops.DECAY_LINEAR(b, 3))
    return -1.0 * (left - right)


def alpha74(ops: Ops) -> pd.Series:
    """
    RANK( CORR( SUM(0.35*LOW+0.65*VWAP, 20), SUM(MEAN(VOLUME,40),20), 7) )
    + RANK( CORR( XRANK(VWAP), XRANK(VOLUME), 6) )
    """
    s1 = ops.SUM(0.35 * ops.LOW + 0.65 * ops.VWAP, 20)
    s2 = ops.SUM(ops.MEAN(ops.VOLUME, 40), 20)
    t1 = ops.RANK(ops.CORR(s1, s2, 7))

    t2 = ops.RANK(ops.CORR(ops.XRANK(ops.VWAP), ops.XRANK(ops.VOLUME), 6))
    return t1 + t2


def alpha75(ops: Ops) -> pd.Series:
    """
    COUNT( CLOSE>OPEN & (BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN) , 50 )
    / COUNT( (BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN) , 50 )
    """
    idx_c = ops.BANCHMARKINDEXCLOSE    # wide (broadcast)
    idx_o = ops.BANCHMARKINDEXOPEN
    cond_bench = idx_c < idx_o
    cond_stock = ops.CLOSE > ops.OPEN
    both = cond_stock & cond_bench

    num = ops.SUM(ops.BOOL_TO_FLOAT(both), 50)
    den = ops.SUM(ops.BOOL_TO_FLOAT(cond_bench), 50)
    return ops.CS(ops._safe_div(num, den))


def alpha76(ops: Ops) -> pd.Series:
    """
    STD( ABS(RET) / VOLUME , 20 ) / MEAN( ABS(RET) / VOLUME , 20 )
    where RET = CLOSE/DELAY(CLOSE,1) - 1
    """
    ret = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 1)) - 1.0
    x = ops._safe_div(np.abs(ret), ops.VOLUME)
    return ops.CS(ops._safe_div(ops.STD(x, 20), ops.MEAN(x, 20)))


def alpha77(ops: Ops) -> pd.Series:
    """
    MIN( RANK(DECAY_LINEAR( ((H+L)/2 - VWAP), 20 )),
         RANK(DECAY_LINEAR( CORR( (H+L)/2, MEAN(VOLUME,40),3 ), 6 )) )
    """
    left  = ops.RANK(ops.DECAY_LINEAR(((ops.HIGH + ops.LOW) / 2.0 - ops.VWAP), 20))
    right = ops.RANK(ops.DECAY_LINEAR(ops.CORR((ops.HIGH + ops.LOW)/2.0, ops.MEAN(ops.VOLUME, 40), 3), 6))
    return pd.Series(np.minimum(left, right), index=left.index)


def alpha78(ops: Ops) -> pd.Series:
    """
    ((TP - MEAN(TP,12)) / (0.015 * MEAN(ABS(CLOSE - MEAN(TP,12)), 12)))
    where TP = (H+L+C)/3
    """
    TP = (ops.HIGH + ops.LOW + ops.CLOSE) / 3.0
    ma_tp = ops.MEAN(TP, 12)
    num = TP - ma_tp
    den = 0.015 * ops.MEAN(np.abs(ops.CLOSE - ma_tp), 12)
    return ops.CS(ops._safe_div(num, den))


def alpha79(ops: Ops) -> pd.Series:
    """ SMA(MAX(CLOSE-C1,0),12,1) / SMA(ABS(CLOSE-C1),12,1) * 100 """
    c = ops.CLOSE
    d = c - ops.DELAY(c, 1)
    num = ops.SMA(np.maximum(d, 0.0), 12, 1)
    den = ops.SMA(np.abs(d), 12, 1)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha80(ops: Ops) -> pd.Series:
    """ (VOLUME - DELAY(VOLUME,5)) / DELAY(VOLUME,5) * 100 """
    v1 = ops.DELAY(ops.VOLUME, 5)
    core = ops._safe_div(ops.VOLUME - v1, v1) * 100.0
    return ops.CS(core)

def alpha81(ops: Ops) -> pd.Series:
    """SMA(VOLUME,21,2)"""
    return ops.CS(ops.SMA(ops.VOLUME, 21, 2))


def alpha82(ops: Ops) -> pd.Series:
    """SMA( (TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100 , 20, 1)"""
    h6 = ops.TSMAX(ops.HIGH, 6)
    l6 = ops.TSMIN(ops.LOW, 6)
    num = (h6 - ops.CLOSE) * 100.0
    den = (h6 - l6)
    rsv = ops._safe_div(num, den)
    return ops.CS(ops.SMA(rsv, 20, 1))


def alpha83(ops: Ops) -> pd.Series:
    """(-1 * RANK(COV(RANK(HIGH), RANK(VOLUME), 5)))"""
    cov5 = ops.COV(ops.XRANK(ops.HIGH), ops.XRANK(ops.VOLUME), 5)
    return -1.0 * ops.RANK(cov5)


def alpha84(ops: Ops) -> pd.Series:
    """SUM( (CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)? -VOLUME:0)), 20)"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    up = (ops.CLOSE > c1)
    dn = (ops.CLOSE < c1)
    signed_v = ops.IF_WIDE(up, ops.VOLUME, ops.IF_WIDE(dn, -ops.VOLUME, 0.0))
    return ops.CS(ops.SUM(signed_v, 20))


def alpha85(ops: Ops) -> pd.Series:
    """TSRANK((VOLUME/MEAN(VOLUME,20)),20) * TSRANK((-1 * DELTA(CLOSE,7)),8)"""
    v_rel = ops._safe_div(ops.VOLUME, ops.MEAN(ops.VOLUME, 20))
    r1 = ops.TSRANK(v_rel, 20)
    r2 = ops.TSRANK(-ops.DELTA(ops.CLOSE, 7), 8)
    return ops.CS(r1 * r2)


def alpha86(ops: Ops) -> pd.Series:
    """Piecewise:
       a = ((DELAY(C,20)-DELAY(C,10))/10) - ((DELAY(C,10)-C)/10)
       if 0.25 < a -> -1
       elif a < 0 -> 1
       else -> -1*(C - DELAY(C,1))
    """
    c = ops.CLOSE
    c10 = ops.DELAY(c, 10)
    c20 = ops.DELAY(c, 20)
    a = ops._safe_div(c20 - c10, 10.0) - ops._safe_div(c10 - c, 10.0)
    choice = ops.IF_WIDE(a > 0.25, -1.0, ops.IF_WIDE(a < 0.0, 1.0, -(c - ops.DELAY(c, 1))))
    return ops.CS(choice)


def alpha87(ops: Ops) -> pd.Series:
    """( RANK(DECAY_LINEAR(DELTA(VWAP,4),7)) + TSRANK(DECAY_LINEAR(((LOW - VWAP)/(OPEN-((HIGH+LOW)/2))),11),7) ) * -1"""
    a1 = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(ops.VWAP, 4), 7))
    den = (ops.OPEN - (ops.HIGH + ops.LOW) / 2.0)
    core = ops._safe_div(ops.LOW - ops.VWAP, den)
    a2 = ops.CS(ops.TSRANK(ops.DECAY_LINEAR(core, 11), 7))
    return -(a1 + a2)


def alpha88(ops: Ops) -> pd.Series:
    """(CLOSE - DELAY(CLOSE,20)) / DELAY(CLOSE,20) * 100"""
    d20 = ops.DELAY(ops.CLOSE, 20)
    return ops.CS(ops._safe_div(ops.CLOSE - d20, d20) * 100.0)


def alpha89(ops: Ops) -> pd.Series:
    """2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))"""
    s13 = ops.SMA(ops.CLOSE, 13, 2)
    s27 = ops.SMA(ops.CLOSE, 27, 2)
    diff = s13 - s27
    s_diff = ops.SMA(diff, 10, 2)
    return ops.CS(2.0 * (s13 - s27 - s_diff))


def alpha90(ops: Ops) -> pd.Series:
    """(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)"""
    corr5 = ops.CORR(ops.XRANK(ops.VWAP), ops.XRANK(ops.VOLUME), 5)
    # 以 t=ops.date 做截面
    # r1_std5 = ops.CS(ops.STD(ops.XRANK(ops.VWAP), 5))  # 每只股票，近5日 XRANK(VWAP) 的 std
    # r2_std5 = ops.CS(ops.STD(ops.XRANK(ops.VOLUME), 5))  # 每只股票，近5日 XRANK(VOLUME) 的 std
    #
    # # 哪些股票在任一输入上 std==0（=> 对应相关系数 NaN）
    # mask_den0 = (r1_std5 == 0) | (r2_std5 == 0)
    # print("f90 分母为0的股票数：", int(mask_den0.sum()))

    return -1.0 * ops.RANK(corr5)


def alpha91(ops: Ops) -> pd.Series:
    """((RANK((CLOSE-TSMAX(CLOSE,5))) * RANK(CORR((MEAN(VOLUME,40)),LOW,5))) * -1)"""
    left = ops.RANK(ops.CLOSE - ops.TSMAX(ops.CLOSE, 5))
    right = ops.RANK(ops.CORR(ops.MEAN(ops.VOLUME, 40), ops.LOW, 5))
    return -(left * right)


def alpha92(ops: Ops) -> pd.Series:
    """(MAX(RANK(DECAY_LINEAR(DELTA(0.35*C+0.65*VWAP,2),3)),
            TSRANK(DECAY_LINEAR(ABS(CORR(MEAN(VOLUME,180),CLOSE,13)),5),15)) * -1)"""
    combo = 0.35 * ops.CLOSE + 0.65 * ops.VWAP
    term1 = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(combo, 2), 3))
    v180 = ops.MEAN(ops.VOLUME, 180)
    corr13 = ops.CORR(v180, ops.CLOSE, 13).abs()
    term2 = ops.CS(ops.TSRANK(ops.DECAY_LINEAR(corr13, 5), 15))
    return -pd.Series(np.maximum(term1.to_numpy(), term2.to_numpy()), index=term1.index)


def alpha93(ops: Ops) -> pd.Series:
    """SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)"""
    o = ops.OPEN
    o1 = ops.DELAY(o, 1)
    comp = pd.DataFrame(np.maximum((o - ops.LOW).to_numpy(), (o - o1).to_numpy()),
                        index=o.index, columns=o.columns)
    term = ops.IF_WIDE(o >= o1, 0.0, comp)
    return ops.CS(ops.SUM(term, 20))


def alpha94(ops: Ops) -> pd.Series:
    """SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)? -VOLUME:0)),30)"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    up = (ops.CLOSE > c1)
    dn = (ops.CLOSE < c1)
    signed_v = ops.IF_WIDE(up, ops.VOLUME, ops.IF_WIDE(dn, -ops.VOLUME, 0.0))
    return ops.CS(ops.SUM(signed_v, 30))


def alpha95(ops: Ops) -> pd.Series:
    """STD(AMOUNT,20)"""
    return ops.CS(ops.STD(ops.AMOUNT, 20))


def alpha96(ops: Ops) -> pd.Series:
    """SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)"""
    h9 = ops.TSMAX(ops.HIGH, 9)
    l9 = ops.TSMIN(ops.LOW, 9)
    num = (ops.CLOSE - l9) * 100.0
    den = (h9 - l9)
    rsv = ops._safe_div(num, den)
    return ops.CS(ops.SMA(ops.SMA(rsv, 3, 1), 3, 1))


def alpha97(ops: Ops) -> pd.Series:
    """STD(VOLUME,10)"""
    return ops.CS(ops.STD(ops.VOLUME, 10))


def alpha98(ops: Ops) -> pd.Series:
    """((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<=0.05)?(-(CLOSE-TSMIN(CLOSE,100))):(-DELTA(CLOSE,3))))"""
    c = ops.CLOSE
    avg100 = ops.SUM(c, 100) / 100.0
    m = ops._safe_div(ops.DELTA(avg100, 100), ops.DELAY(c, 100))
    left = -(c - ops.TSMIN(c, 100))
    right = -ops.DELTA(c, 3)
    return ops.CS(ops.IF_WIDE(m <= 0.05, left, right))


def alpha99(ops: Ops) -> pd.Series:
    """(-1 * RANK(COV(RANK(CLOSE), RANK(VOLUME), 5)))"""
    cov5 = ops.COV(ops.XRANK(ops.CLOSE), ops.XRANK(ops.VOLUME), 5)
    return -1.0 * ops.RANK(cov5)


def alpha100(ops: Ops) -> pd.Series:
    """STD(VOLUME,20)"""
    return ops.CS(ops.STD(ops.VOLUME, 20))

def alpha101(ops: Ops) -> pd.Series:
    """
    ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37), 15)) <
      RANK(CORR(RANK(0.1*HIGH+0.9*VWAP), RANK(VOLUME), 11))) * -1)
    """
    left  = ops.RANK(ops.CORR(ops.CLOSE, ops.SUM(ops.MEAN(ops.VOLUME, 30), 37), 15))
    right = ops.RANK(ops.CORR(ops.XRANK(0.1 * ops.HIGH + 0.9 * ops.VWAP),
                              ops.XRANK(ops.VOLUME), 11))
    return ops.IF(left < right, -1.0, 0.0)


def alpha102(ops: Ops) -> pd.Series:
    """SMA(MAX(V-Delay(V,1),0),6,1) / SMA(ABS(V-Delay(V,1)),6,1) * 100"""
    dv = ops.VOLUME - ops.DELAY(ops.VOLUME, 1)
    up = ops.IF_WIDE(dv > 0, dv, 0.0)
    num = ops.SMA(up, 6, 1)
    den = ops.SMA(dv.abs(), 6, 1)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha103(ops: Ops) -> pd.Series:
    """((20 - LOWDAY(LOW,20)) / 20) * 100"""
    ld = ops.LOWDAY(ops.LOW, 20)
    return ops.CS((20.0 - ld) / 20.0 * 100.0)


def alpha104(ops: Ops) -> pd.Series:
    """(-1 * ( DELTA(CORR(HIGH,VOLUME,5),5) * RANK(STD(CLOSE,20)) ))"""
    d_corr = ops.DELTA(ops.CORR(ops.HIGH, ops.VOLUME, 5), 5)
    return -ops.CS(d_corr) * ops.RANK(ops.STD(ops.CLOSE, 20))


def alpha105(ops: Ops) -> pd.Series:
    """(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))"""
    corr = ops.CORR(ops.XRANK(ops.OPEN), ops.XRANK(ops.VOLUME), 10)

    # std_o = ops.CS(ops.STD(ops.XRANK(ops.OPEN), 10))
    # std_v = ops.CS(ops.STD(ops.XRANK(ops.VOLUME), 10))
    # den0_cnt = int(((std_o == 0) | (std_v == 0)).sum())
    # print("f105 | 分母为 0 的股票数 =", den0_cnt)

    return -ops.CS(corr)


def alpha106(ops: Ops) -> pd.Series:
    """CLOSE - DELAY(CLOSE,20)"""
    return ops.CS(ops.CLOSE - ops.DELAY(ops.CLOSE, 20))


def alpha107(ops: Ops) -> pd.Series:
    """((-1*RANK(OPEN-Delay(HIGH,1))) * RANK(OPEN-Delay(CLOSE,1)) * RANK(OPEN-Delay(LOW,1)))"""
    t1 = -ops.RANK(ops.OPEN - ops.DELAY(ops.HIGH, 1))
    t2 =  ops.RANK(ops.OPEN - ops.DELAY(ops.CLOSE, 1))
    t3 =  ops.RANK(ops.OPEN - ops.DELAY(ops.LOW, 1))
    return t1 * t2 * t3


def alpha108(ops: Ops) -> pd.Series:
    """((RANK(HIGH - TSMIN(HIGH,2)) ^ RANK(CORR(VWAP, MEAN(VOLUME,120), 6))) * -1)"""
    t1 = ops.RANK(ops.HIGH - ops.TSMIN(ops.HIGH, 2))
    t2 = ops.RANK(ops.CORR(ops.VWAP, ops.MEAN(ops.VOLUME, 120), 6))
    return -(t1.pow(t2))


def alpha109(ops: Ops) -> pd.Series:
    """SMA(HIGH-LOW,10,2) / SMA(SMA(HIGH-LOW,10,2),10,2)"""
    x   = ops.HIGH - ops.LOW
    s1  = ops.SMA(x, 10, 2)
    s2  = ops.SMA(s1, 10, 2)
    return ops.CS(ops._safe_div(s1, s2))


def alpha110(ops: Ops) -> pd.Series:
    """SUM(MAX(0,HIGH-Delay(CLOSE,1)),20)/SUM(MAX(0,Delay(CLOSE,1)-LOW),20)*100"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    up  = ops.IF_WIDE(ops.HIGH - c1 > 0, ops.HIGH - c1, 0.0)
    dn  = ops.IF_WIDE(c1 - ops.LOW > 0, c1 - ops.LOW, 0.0)
    return ops.CS(ops._safe_div(ops.SUM(up, 20), ops.SUM(dn, 20)) * 100.0)


def alpha111(ops: Ops) -> pd.Series:
    """SMA(V*((C-L)-(H-C))/(H-L),11,2) - SMA(V*((C-L)-(H-C))/(H-L),4,2)"""
    num = ops.VOLUME * ((ops.CLOSE - ops.LOW) - (ops.HIGH - ops.CLOSE))
    den = ops.HIGH - ops.LOW
    x = ops._safe_div(num, den)
    return ops.CS(ops.SMA(x, 11, 2) - ops.SMA(x, 4, 2))


def alpha112(ops: Ops) -> pd.Series:
    """(SUM(pos,12) - SUM(neg,12)) / (SUM(pos,12) + SUM(neg,12)) * 100, where pos=max(ΔC,0), neg=max(-ΔC,0)"""
    dc = ops.CLOSE - ops.DELAY(ops.CLOSE, 1)
    pos = ops.IF_WIDE(dc > 0, dc, 0.0)
    neg = ops.IF_WIDE(dc < 0, -dc, 0.0)
    sp, sn = ops.SUM(pos, 12), ops.SUM(neg, 12)
    return ops.CS(ops._safe_div(sp - sn, sp + sn) * 100.0)


def alpha113(ops: Ops) -> pd.Series:
    """(-1 * ( RANK(SUM(Delay(C,5),20)/20) * CS(CORR(C,V,2)) * RANK(CORR(SUM(C,5), SUM(C,20), 2)) ))"""
    t1 = ops.RANK(ops.SUM(ops.DELAY(ops.CLOSE, 5), 20) / 20.0)
    t2 = ops.CS(ops.CORR(ops.CLOSE, ops.VOLUME, 2))
    t3 = ops.RANK(ops.CORR(ops.SUM(ops.CLOSE, 5), ops.SUM(ops.CLOSE, 20), 2))
    return -(t1 * t2 * t3)


def alpha114(ops: Ops) -> pd.Series:
    """( RANK(Delay(((H-L)/(SUM(C,5)/5)),2)) * RANK(RANK(V)) ) / ( ((H-L)/(SUM(C,5)/5)) / (VWAP-C) )"""
    top = ops._safe_div(ops.HIGH - ops.LOW, ops.SUM(ops.CLOSE, 5) / 5.0)
    num = ops.RANK(ops.DELAY(top, 2)) * ops.RANK(ops.RANK(ops.VOLUME))
    den = ops._safe_div(top, ops.VWAP - ops.CLOSE)
    return ops.CS(ops._safe_div(num, den))


def alpha115(ops: Ops) -> pd.Series:
    """RANK(CORR(0.9*H+0.1*C, MEAN(V,30),10)) ^ RANK(CORR(TSRANK((H+L)/2,4), TSRANK(V,10),7))"""
    t1 = ops.RANK(ops.CORR(0.9 * ops.HIGH + 0.1 * ops.CLOSE, ops.MEAN(ops.VOLUME, 30), 10))
    t2 = ops.RANK(ops.CORR(ops.TSRANK((ops.HIGH + ops.LOW) / 2.0, 4),
                           ops.TSRANK(ops.VOLUME, 10), 7))
    return t1.pow(t2)


def alpha116(ops: Ops) -> pd.Series:
    """REGBETA(CLOSE, SEQUENCE, 20)"""
    return ops.CS(ops.REGBETA(ops.CLOSE, ops.SEQUENCE(20)))


def alpha117(ops: Ops) -> pd.Series:
    """(TSRANK(V,32) * (1-TSRANK(C+H-L,16)) * (1-TSRANK(RET,32)))"""
    t1 = ops.TSRANK(ops.VOLUME, 32)
    t2 = 1.0 - ops.TSRANK((ops.CLOSE + ops.HIGH) - ops.LOW, 16)
    t3 = 1.0 - ops.TSRANK(ops.RET, 32)
    return ops.CS(t1 * t2 * t3)


def alpha118(ops: Ops) -> pd.Series:
    """SUM(HIGH-OPEN,20) / SUM(OPEN-LOW,20) * 100"""
    num = ops.SUM(ops.HIGH - ops.OPEN, 20)
    den = ops.SUM(ops.OPEN - ops.LOW, 20)
    return ops.CS(ops._safe_div(num, den) * 100.0)


def alpha119(ops: Ops) -> pd.Series:
    """RANK(DECAY_LINEAR(CORR(VWAP, SUM(MEAN(V,5),26),5),7)) - RANK(DECAY_LINEAR(TSRANK(TSMIN(CORR(RANK(OPEN),RANK(MEAN(V,15)),21),9),7),8))"""
    t1 = ops.RANK(ops.DECAY_LINEAR(ops.CORR(ops.VWAP, ops.SUM(ops.MEAN(ops.VOLUME, 5), 26), 5), 7))
    corr = ops.CORR(ops.XRANK(ops.OPEN), ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 21)
    tmin = ops.TSMIN(corr, 9)
    t2 = ops.RANK(ops.DECAY_LINEAR(ops.TSRANK(tmin, 7), 8))
    # # 左半链路
    # yL1 = ops.SUM(ops.MEAN(ops.VOLUME, 5), 26)
    # corrL = ops.CORR(ops.VWAP, yL1, 5)
    # print("f119 | 左半 corrL(最后1天) 非NaN个数 =", int(ops.CS(~corrL.isna()).sum()))
    #
    # # 右半链路（最容易卡在“std=0”的是这条）
    # corrR = ops.CORR(ops.XRANK(ops.OPEN), ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 21)
    # m9 = ops.TSMIN(corrR, 9)
    # tr7 = ops.TSRANK(m9, 7)
    # dec8 = ops.DECAY_LINEAR(tr7, 8)
    # print("f119 | 右半 corrR 非NaN个数 =", int(ops.CS(~corrR.isna()).sum()))
    # print("f119 | 右半 最终 dec8 非NaN个数 =", int(ops.CS(~dec8.isna()).sum()))
    # std_o21 = ops.CS(ops.STD(ops.XRANK(ops.OPEN), 21))
    # std_mv = ops.CS(ops.STD(ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 21))
    # print("f119 | 右半 分母为0股票数 =", int(((std_o21 == 0) | (std_mv == 0)).sum()))

    return t1 - t2


def alpha120(ops: Ops) -> pd.Series:
    """RANK(VWAP-CLOSE) / RANK(VWAP+CLOSE)"""
    num = ops.RANK(ops.VWAP - ops.CLOSE)
    den = ops.RANK(ops.VWAP + ops.CLOSE)
    return ops._safe_div(num, den)


def alpha121(ops: Ops) -> pd.Series:
    """( RANK(VWAP - TSMIN(VWAP,12)) ^ TSRANK(CORR(TSRANK(VWAP,20), TSRANK(MEAN(V,60),2), 18), 3) ) * -1"""
    left  = ops.RANK(ops.VWAP - ops.TSMIN(ops.VWAP, 12))
    right = ops.CS(ops.TSRANK(ops.CORR(ops.TSRANK(ops.VWAP, 20),
                                       ops.TSRANK(ops.MEAN(ops.VOLUME, 60), 2), 18), 3))
    return -(left.pow(right))


def alpha122(ops: Ops) -> pd.Series:
    """(S3 - Delay(S3,1)) / Delay(S3,1), where S3 = SMA(SMA(SMA(LOG(C),13,2),13,2),13,2)"""
    s1 = ops.SMA(ops.LOG(ops.CLOSE), 13, 2)
    s2 = ops.SMA(s1, 13, 2)
    s3 = ops.SMA(s2, 13, 2)
    d1 = s3 - ops.DELAY(s3, 1)
    return ops.CS(ops._safe_div(d1, ops.DELAY(s3, 1)))


def alpha123(ops: Ops) -> pd.Series:
    """((RANK(CORR(SUM((H+L)/2,20), SUM(MEAN(V,60),20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)"""
    left  = ops.RANK(ops.CORR(ops.SUM((ops.HIGH + ops.LOW) / 2.0, 20),
                              ops.SUM(ops.MEAN(ops.VOLUME, 60), 20), 9))
    right = ops.RANK(ops.CORR(ops.LOW, ops.VOLUME, 6))
    return ops.IF(left < right, -1.0, 0.0)


def alpha124(ops: Ops) -> pd.Series:
    """(CLOSE - VWAP) / DECAY_LINEAR(RANK(TSMAX(CLOSE,30)), 2)"""
    den = ops.DECAY_LINEAR(ops.XRANK(ops.TSMAX(ops.CLOSE, 30)), 2)
    return ops.CS(ops._safe_div(ops.CLOSE - ops.VWAP, den))


def alpha125(ops: Ops) -> pd.Series:
    """RANK(DECAY_LINEAR(CORR(VWAP, MEAN(V,80),17),20)) / RANK(DECAY_LINEAR(DELTA(0.5*C+0.5*VWAP,3),16))"""
    t1 = ops.RANK(ops.DECAY_LINEAR(ops.CORR(ops.VWAP, ops.MEAN(ops.VOLUME, 80), 17), 20))
    t2 = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(0.5 * ops.CLOSE + 0.5 * ops.VWAP, 3), 16))
    return ops._safe_div(t1, t2)


def alpha126(ops: Ops) -> pd.Series:
    """(CLOSE + HIGH + LOW) / 3"""
    return ops.CS((ops.CLOSE + ops.HIGH + ops.LOW) / 3.0)

def alpha128(ops: Ops) -> pd.Series:
    """100 - 100 / (1 + SUM(up_flow,14) / SUM(dn_flow,14)),
       where up_flow = tp*V if tp>tp1 else 0, dn_flow = tp*V if tp<tp1 else 0, tp=(H+L+C)/3"""
    tp  = (ops.HIGH + ops.LOW + ops.CLOSE) / 3.0
    tp1 = ops.DELAY(tp, 1)
    up  = ops.IF_WIDE(tp > tp1, tp * ops.VOLUME, 0.0)
    dn  = ops.IF_WIDE(tp < tp1, tp * ops.VOLUME, 0.0)
    s_up = ops.SUM(up, 14)
    s_dn = ops.SUM(dn, 14)
    ratio = ops._safe_div(s_up, s_dn)
    out = 100.0 - 100.0 / (1.0 + ratio)
    return ops.CS(out)


def alpha129(ops: Ops) -> pd.Series:
    """SUM( (ΔC<0 ? |ΔC| : 0), 12 )"""
    dc = ops.CLOSE - ops.DELAY(ops.CLOSE, 1)
    neg = ops.IF_WIDE(dc < 0, -dc, 0.0)
    return ops.CS(ops.SUM(neg, 12))


def alpha130(ops: Ops) -> pd.Series:
    """RANK(DECAY_LINEAR(CORR((H+L)/2, MEAN(V,40), 9), 10)) / RANK(DECAY_LINEAR(CORR(RANK(VWAP), RANK(V), 7), 3))"""
    left  = ops.RANK(ops.DECAY_LINEAR(ops.CORR((ops.HIGH + ops.LOW) / 2.0, ops.MEAN(ops.VOLUME, 40), 9), 10))
    right = ops.RANK(ops.DECAY_LINEAR(ops.CORR(ops.XRANK(ops.VWAP), ops.XRANK(ops.VOLUME), 7), 3))

    # std_vwap7 = ops.CS(ops.STD(ops.XRANK(ops.VWAP), 7))
    # std_vol7 = ops.CS(ops.STD(ops.XRANK(ops.VOLUME), 7))
    # nonNa_L = int(ops.CS(~ops.CORR((ops.HIGH + ops.LOW) / 2.0, ops.MEAN(ops.VOLUME, 40), 9).isna()).sum())
    # nonNa_R = int(ops.CS(~ops.CORR(ops.XRANK(ops.VWAP), ops.XRANK(ops.VOLUME), 7).isna()).sum())
    # print(f"[QC f130] den0={int(((std_vwap7 == 0) | (std_vol7 == 0)).sum())}/{len(codes)} "
    #       f"| left_nonNa={nonNa_L} right_nonNa={nonNa_R}")
    return ops._safe_div(ops.CS(left), ops.CS(right))


def alpha131(ops: Ops) -> pd.Series:
    """( RANK(DELTA(VWAP,1)) ^ TSRANK(CORR(CLOSE, MEAN(V,50), 18), 18) )"""
    left  = ops.RANK(ops.DELTA(ops.VWAP, 1))
    right = ops.TSRANK(ops.CORR(ops.CLOSE, ops.MEAN(ops.VOLUME, 50), 18), 18)
    return ops.CS(left.pow(right))


def alpha132(ops: Ops) -> pd.Series:
    """MEAN(AMOUNT, 20)"""
    return ops.CS(ops.MEAN(ops.AMOUNT, 20))


def alpha133(ops: Ops) -> pd.Series:
    """((20 - HIGHDAY(H,20))/20)*100 - ((20 - LOWDAY(L,20))/20)*100"""
    hd = ops.HIGHDAY(ops.HIGH, 20)
    ld = ops.LOWDAY(ops.LOW, 20)
    return ops.CS(((20.0 - hd) / 20.0 - (20.0 - ld) / 20.0) * 100.0)


def alpha134(ops: Ops) -> pd.Series:
    """(C - C12)/C12 * V"""
    c12 = ops.DELAY(ops.CLOSE, 12)
    ret = ops._safe_div(ops.CLOSE - c12, c12)
    return ops.CS(ret * ops.VOLUME)


def alpha135(ops: Ops) -> pd.Series:
    """SMA( DELAY(C/DELAY(C,20),1), 20, 1 )"""
    ratio = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 20))
    return ops.CS(ops.SMA(ops.DELAY(ratio, 1), 20, 1))


def alpha136(ops: Ops) -> pd.Series:
    """((-1 * RANK(DELTA(RET,3))) * CORR(OPEN, VOLUME, 10))"""
    t1 = -ops.RANK(ops.DELTA(ops.RET, 3))
    t2 = ops.CORR(ops.OPEN, ops.VOLUME, 10)
    return ops.CS(t1 * t2)


def alpha137(ops: Ops) -> pd.Series:
    """复杂条件分母的动量/波动组合"""
    c1, o1 = ops.DELAY(ops.CLOSE, 1), ops.DELAY(ops.OPEN, 1)
    a = (ops.HIGH - c1).abs()
    b = (ops.LOW  - c1).abs()
    c = (ops.HIGH - ops.DELAY(ops.LOW, 1)).abs()
    q = (c1 - o1).abs() / 4.0

    cond1 = (a > b) & (a > c)
    cond2 = (b > c) & (b > a)

    denom1 = a + b / 2.0 + q
    denom2 = b + a / 2.0 + q
    denom3 = c + q
    denom  = ops.IF_WIDE(cond1, denom1, ops.IF_WIDE(cond2, denom2, denom3))

    numer = 16.0 * ( (ops.CLOSE - c1) + 0.5 * (ops.CLOSE - ops.OPEN) + (c1 - o1) )
    scale = pd.DataFrame(np.maximum(a.to_numpy(), b.to_numpy()), index=a.index, columns=a.columns)
    val   = ops._safe_div(numer, denom) * scale
    return ops.CS(val)


def alpha138(ops: Ops) -> pd.Series:
    """( RANK(DECAY_LINEAR(DELTA(0.7*L + 0.3*VWAP, 3), 20)) - TSRANK(DECAY_LINEAR(TSRANK(CORR(TSRANK(L,8), TSRANK(MEAN(V,60),17), 5), 19), 16), 7) ) * -1"""
    t1 = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(0.7 * ops.LOW + 0.3 * ops.VWAP, 3), 20))
    t2_core = ops.CORR(ops.TSRANK(ops.LOW, 8), ops.TSRANK(ops.MEAN(ops.VOLUME, 60), 17), 5)
    t2 = ops.TSRANK(ops.DECAY_LINEAR(ops.TSRANK(t2_core, 19), 16), 7)

    # ts_low8 = ops.TSRANK(ops.LOW, 8)
    # ts_mv60_17 = ops.TSRANK(ops.MEAN(ops.VOLUME, 60), 17)
    # std_low5 = ops.CS(ops.STD(ts_low8, 5))
    # std_mv5 = ops.CS(ops.STD(ts_mv60_17, 5))
    # nonNa_corr5 = int(ops.CS(~ops.CORR(ts_low8, ts_mv60_17, 5).isna()).sum())
    # print(f"[QC f138] den0={int(((std_low5 == 0) | (std_mv5 == 0)).sum())}/{len(codes)} "
    #       f"| corr5_nonNa={nonNa_corr5}")

    return -(ops.CS(t1) - ops.CS(t2))


def alpha139(ops: Ops) -> pd.Series:
    """(-1 * CORR(OPEN, VOLUME, 10))"""
    return -ops.CS(ops.CORR(ops.OPEN, ops.VOLUME, 10))


def alpha140(ops: Ops) -> pd.Series:
    """MIN( RANK(DECAY_LINEAR((XRANK(O)+XRANK(L)-XRANK(H)-XRANK(C)),8)),
             TSRANK(DECAY_LINEAR(CORR(TSRANK(C,8), TSRANK(MEAN(V,60),20),8),7),3) )"""
    left  = ops.RANK(ops.DECAY_LINEAR(ops.XRANK(ops.OPEN) + ops.XRANK(ops.LOW)
                                      - ops.XRANK(ops.HIGH) - ops.XRANK(ops.CLOSE), 8))
    right = ops.TSRANK(ops.DECAY_LINEAR(ops.CORR(ops.TSRANK(ops.CLOSE, 8),
                                                 ops.TSRANK(ops.MEAN(ops.VOLUME, 60), 20), 8), 7), 3)
    L = ops.CS(left); R = ops.CS(right)

    # ts_c8 = ops.TSRANK(ops.CLOSE, 8)
    # ts_mv60_20 = ops.TSRANK(ops.MEAN(ops.VOLUME, 60), 20)
    # std_c8 = ops.CS(ops.STD(ts_c8, 8))
    # std_mv8 = ops.CS(ops.STD(ts_mv60_20, 8))
    # nonNa_corr8 = int(ops.CS(~ops.CORR(ts_c8, ts_mv60_20, 8).isna()).sum())
    # print(f"[QC f140] den0={int(((std_c8 == 0) | (std_mv8 == 0)).sum())}/{len(codes)} "
    #       f"| corr8_nonNa={nonNa_corr8}")

    return pd.Series(np.minimum(L.to_numpy(), R.to_numpy()), index=L.index)


def alpha141(ops: Ops) -> pd.Series:
    """(-1 * RANK(CORR(RANK(HIGH), RANK(MEAN(V,15)), 9)))"""
    corr = ops.CORR(ops.XRANK(ops.HIGH), ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 9)

    # std_h9 = ops.CS(ops.STD(ops.XRANK(ops.HIGH), 9))
    # std_mv9 = ops.CS(ops.STD(ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 9))
    # nonNa_c9 = int(ops.CS(~ops.CORR(ops.XRANK(ops.HIGH),
    #                                 ops.XRANK(ops.MEAN(ops.VOLUME, 15)), 9).isna()).sum())
    # print(f"[QC f141] den0={int(((std_h9 == 0) | (std_mv9 == 0)).sum())}/{len(codes)} "
    #       f"| corr9_nonNa={nonNa_c9}")

    return -ops.CS(ops.RANK(corr))


def alpha142(ops: Ops) -> pd.Series:
    """((-1*RANK(TSRANK(C,10))) * RANK(DELTA(DELTA(C,1),1)) * RANK(TSRANK(V/MEAN(V,20),5)))"""
    t1 = -ops.RANK(ops.TSRANK(ops.CLOSE, 10))
    t2 =  ops.RANK(ops.DELTA(ops.DELTA(ops.CLOSE, 1), 1))
    vrel = ops._safe_div(ops.VOLUME, ops.MEAN(ops.VOLUME, 20))
    t3 =  ops.RANK(ops.TSRANK(vrel, 5))
    return ops.CS(t1 * t2 * t3)


def alpha143(ops: Ops) -> pd.Series:
    """
    f143（递推版，无前视）：
      SELF_t = SELF_{t-1} * ret_t   if CLOSE_t > CLOSE_{t-1}
             = SELF_{t-1}           otherwise
      其中 ret_t = (CLOSE_t - CLOSE_{t-1}) / CLOSE_{t-1}
    实现方式：在历史窗口内构造逐日乘子 g_t = ret_t(若上涨) 否则 1.0，然后做按时间的 cumprod，取当日截面。
    暖机≈2（需要 DELAY(CLOSE,1)）。
    """
    c  = ops.CLOSE
    c1 = ops.DELAY(c, 1)
    ret = ops._safe_div(c - c1, c1)             # 当日涨幅（分母安全）
    cond = (c > c1)                              # 仅在上涨日更新
    g = ops.IF_WIDE(cond, ret, 1.0)              # 乘子：涨则乘 ret，否则乘 1
    cum = g.cumprod()                            # 沿时间轴累乘（各列独立）
    return ops.CS(cum)                           # 取 t 日截面



def alpha144(ops: Ops) -> pd.Series:
    """SUMIF(|C/C1 - 1| / AMOUNT, 20, C<C1) / COUNT(C<C1, 20)"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    ret_abs = (ops._safe_div(ops.CLOSE, c1) - 1.0).abs()
    val = ops._safe_div(ret_abs, ops.AMOUNT)
    cond = ops.CLOSE < c1
    num = ops.SUM(ops.IF_WIDE(cond, val, 0.0), 20)
    den = ops.SUM(ops.IF_WIDE(cond, 1.0, 0.0), 20)
    return ops.CS(ops._safe_div(num, den))


def alpha145(ops: Ops) -> pd.Series:
    """(MEAN(V,9) - MEAN(V,26)) / MEAN(V,12) * 100"""
    num = ops.MEAN(ops.VOLUME, 9) - ops.MEAN(ops.VOLUME, 26)
    den = ops.MEAN(ops.VOLUME, 12)
    return ops.CS(ops._safe_div(num, den) * 100.0)

def alpha147(ops: Ops) -> pd.Series:
    """REGBETA(MEAN(CLOSE,12), SEQUENCE(12), 12)"""
    y = ops.MEAN(ops.CLOSE, 12)
    return ops.CS(ops.REGBETA(y, ops.SEQUENCE(12)))


def alpha148(ops: Ops) -> pd.Series:
    """((RANK(CORR(OPEN, SUM(MEAN(V,60),9), 6)) < RANK(OPEN - TSMIN(OPEN,14))) * -1)"""
    left  = ops.RANK(ops.CORR(ops.OPEN, ops.SUM(ops.MEAN(ops.VOLUME, 60), 9), 6))
    right = ops.RANK(ops.OPEN - ops.TSMIN(ops.OPEN, 14))
    return ops.IF(left < right, -1.0, 0.0)


def alpha149(ops: Ops) -> pd.Series:
    bclose = ops.BANCHMARKINDEXCLOSE
    mask = bclose < ops.DELAY(bclose, 1)                         # DataFrame[bool]
    stock_ret = ops.RET
    bench_ret = ops._safe_div(bclose, ops.DELAY(bclose, 1)) - 1.0
    y = ops.IF_WIDE(mask, stock_ret, np.nan)                     # 等价于 FILTER(...)
    x = ops.IF_WIDE(mask, bench_ret, np.nan)
    return ops.CS(ops.REGBETA(y, x, 252))

def alpha150(ops: Ops) -> pd.Series:
    """(CLOSE + HIGH + LOW)/3 * VOLUME"""
    return ops.CS(((ops.CLOSE + ops.HIGH + ops.LOW) / 3.0) * ops.VOLUME)


def alpha151(ops: Ops) -> pd.Series:
    """SMA(CLOSE - DELAY(CLOSE,20), 20, 1)"""
    return ops.CS(ops.SMA(ops.CLOSE - ops.DELAY(ops.CLOSE, 20), 20, 1))


def alpha152(ops: Ops) -> pd.Series:
    """SMA( MEAN( DELAY( SMA( DELAY(C/DELAY(C,9),1), 9,1 ), 1 ), 12 ) - MEAN( DELAY( SMA( DELAY(C/DELAY(C,9),1), 9,1 ), 1 ), 26 ), 9, 1)"""
    ratio = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 9))
    core  = ops.SMA(ops.DELAY(ratio, 1), 9, 1)
    core1 = ops.DELAY(core, 1)
    diff  = ops.MEAN(core1, 12) - ops.MEAN(core1, 26)
    return ops.CS(ops.SMA(diff, 9, 1))


def alpha153(ops: Ops) -> pd.Series:
    """(MEAN(C,3) + MEAN(C,6) + MEAN(C,12) + MEAN(C,24)) / 4"""
    return ops.CS((ops.MEAN(ops.CLOSE, 3) + ops.MEAN(ops.CLOSE, 6)
                   + ops.MEAN(ops.CLOSE, 12) + ops.MEAN(ops.CLOSE, 24)) / 4.0)


def alpha154(ops: Ops) -> pd.Series:
    """((VWAP - TSMIN(VWAP,16)) < CORR(VWAP, MEAN(V,180), 18))  → 1/0"""
    left  = ops.VWAP - ops.TSMIN(ops.VWAP, 16)
    right = ops.CORR(ops.VWAP, ops.MEAN(ops.VOLUME, 180), 18)
    return ops.IF(left < right, 1.0, 0.0)


def alpha155(ops: Ops) -> pd.Series:
    """SMA(V,13,2) - SMA(V,27,2) - SMA(SMA(V,13,2) - SMA(V,27,2), 10, 2)"""
    s13 = ops.SMA(ops.VOLUME, 13, 2)
    s27 = ops.SMA(ops.VOLUME, 27, 2)
    diff = s13 - s27
    return ops.CS(s13 - s27 - ops.SMA(diff, 10, 2))


def alpha156(ops: Ops) -> pd.Series:
    """-( MAX( RANK(DECAY_LINEAR(DELTA(VWAP,5),3)),
                RANK(DECAY_LINEAR( -( DELTA(0.15*OPEN+0.85*LOW,2) / (0.15*OPEN+0.85*LOW) ), 3 )) ) )"""
    t1 = ops.RANK(ops.DECAY_LINEAR(ops.DELTA(ops.VWAP, 5), 3))
    base = 0.15 * ops.OPEN + 0.85 * ops.LOW
    frac = -ops._safe_div(ops.DELTA(base, 2), base)
    t2 = ops.RANK(ops.DECAY_LINEAR(frac, 3))
    L = ops.CS(t1); R = ops.CS(t2)
    return -(pd.Series(np.maximum(L.to_numpy(), R.to_numpy()), index=L.index))


def alpha157(ops: Ops) -> pd.Series:
    """
    修正点：
    - z 用 XRANK 保持宽表，否则 TSMIN 会拿到当日 Series 触发类型错误。
    - 保留 LOG 下界保护与右侧 ret 的显式计算。
    暖机≈6。
    """
    # 左半：到 LOG 前加下界保护
    t0 = -ops.RANK(ops.DELTA(ops.CLOSE, 5))
    t1 = ops.RANK(ops.RANK(t0))
    s1 = ops.TSMIN(t1, 2)                              # SUM(...,1) 等价自身
    s1_pos = ops.IF_WIDE(s1 <= 0, 1e-12, s1)           # LOG 下界保护

    # 关键修复：用 XRANK 保持“时间×股票”的宽表，不要把它压成当日 Series
    z  = ops.XRANK(ops.XRANK(ops.LOG(s1_pos)))         # ← 原来是 ops.RANK(...)

    m5 = ops.TSMIN(z, 5)                               # 现在可按时间滚动

    # 右半：显式用 CLOSE 现算日收益
    ret = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 1)) - 1.0
    ts  = ops.TSRANK(-ops.DELAY(ret, 6), 5)

    return ops.CS(m5 + ts)




def alpha158(ops: Ops) -> pd.Series:
    """((HIGH - SMA(C,15,2)) - (LOW - SMA(C,15,2))) / CLOSE"""
    s = ops.SMA(ops.CLOSE, 15, 2)
    num = (ops.HIGH - s) - (ops.LOW - s)
    return ops.CS(ops._safe_div(num, ops.CLOSE))


def alpha159(ops: Ops) -> pd.Series:
    """
    6/12/24 三档动量的加权合成（修正 HGIH→HIGH）：
    sum_w = 6*12 + 6*24 + 12*24
    """
    def block(w, k):
        c1 = ops.DELAY(ops.CLOSE, 1)
        num = ops.CLOSE - ops.SUM(ops.IF_WIDE(ops.LOW < c1, ops.LOW, c1), w)
        den = ops.SUM(pd.DataFrame(np.maximum(ops.HIGH.to_numpy(), c1.to_numpy()),
                                   index=ops.HIGH.index, columns=ops.HIGH.columns) -
                      pd.DataFrame(np.minimum(ops.LOW.to_numpy(),  c1.to_numpy()),
                                   index=ops.LOW.index,  columns=ops.LOW.columns), w)
        return ops._safe_div(num, den) * k
    term = block(6, 12*24) + block(12, 6*24) + block(24, 6*24)
    return ops.CS(term * 100.0 / float(6*12 + 6*24 + 12*24))


def alpha160(ops: Ops) -> pd.Series:
    """SMA( (C<=C1 ? STD(C,20) : 0), 20, 1 )"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    val = ops.IF_WIDE(ops.CLOSE <= c1, ops.STD(ops.CLOSE, 20), 0.0)
    return ops.CS(ops.SMA(val, 20, 1))


def alpha161(ops: Ops) -> pd.Series:
    """MEAN( MAX( MAX(H-L, |C1-H|), |C1-L| ), 12 )"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    a = ops.HIGH - ops.LOW
    b = (c1 - ops.HIGH).abs()
    c = (c1 - ops.LOW).abs()
    mx = pd.DataFrame(np.maximum(np.maximum(a.to_numpy(), b.to_numpy()),
                                 c.to_numpy()), index=a.index, columns=a.columns)
    return ops.CS(ops.MEAN(mx, 12))


def alpha162(ops: Ops) -> pd.Series:
    """(RS - TSMIN(RS,12)) / (TSMAX(RS,12) - TSMIN(RS,12)), where RS = SMA(MAX(ΔC,0),12,1)/SMA(ABS(ΔC),12,1)*100"""
    dc  = ops.CLOSE - ops.DELAY(ops.CLOSE, 1)
    up  = ops.SMA(ops.IF_WIDE(dc > 0, dc, 0.0), 12, 1)
    tot = ops.SMA(dc.abs(), 12, 1)
    RS  = ops._safe_div(up, tot) * 100.0
    lo  = ops.TSMIN(RS, 12)
    hi  = ops.TSMAX(RS, 12)
    return ops.CS(ops._safe_div(RS - lo, hi - lo))


def alpha163(ops: Ops) -> pd.Series:
    """RANK( ((-RET) * MEAN(V,20) * VWAP * (H - C)) )"""
    core = (-ops.RET) * ops.MEAN(ops.VOLUME, 20) * ops.VWAP * (ops.HIGH - ops.CLOSE)
    return ops.RANK(core)


def alpha164(ops: Ops) -> pd.Series:
    """SMA( ( x - TSMIN(x,12) ) / (H - L) * 100 , 13, 2), where x = (C>C1 ? 1/(C-C1) : 1)"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    x  = ops.IF_WIDE(ops.CLOSE > c1, ops._safe_div(1.0, ops.CLOSE - c1), 1.0)
    num = x - ops.TSMIN(x, 12)
    den = ops.HIGH - ops.LOW
    return ops.CS(ops.SMA(ops._safe_div(num, den) * 100.0, 13, 2))

def alpha166(ops: Ops) -> pd.Series:
    """
    Skewness-like over 20 days (robust):
    r = C/Delay(C,1)-1
    num = SUM( (r - MEAN(r,20))^3 , 20 )
    den = ( SUM( (r - MEAN(r,20))^2 , 20 ) )^(3/2)
    out = - ((20-1)**1.5) * num / ( (20-1)*(20-2) * den )
    """
    r = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 1)) - 1.0
    mu = ops.MEAN(r, 20)
    dm = r - mu
    num = ops.SUM(dm.pow(3), 20)
    den = ops.SUM(dm.pow(2), 20).pow(1.5)
    coef = (19.0 ** 1.5) / (19.0 * 18.0)  # = sqrt(19)/18
    return ops.CS(-coef * ops._safe_div(num, den))


def alpha167(ops: Ops) -> pd.Series:
    """SUM( max(ΔC,0) , 12 )"""
    dc = ops.CLOSE - ops.DELAY(ops.CLOSE, 1)
    pos = ops.IF_WIDE(dc > 0, dc, 0.0)
    return ops.CS(ops.SUM(pos, 12))


def alpha168(ops: Ops) -> pd.Series:
    """- V / MEAN(V,20)"""
    return ops.CS(-ops._safe_div(ops.VOLUME, ops.MEAN(ops.VOLUME, 20)))


def alpha169(ops: Ops) -> pd.Series:
    """SMA( MEAN( DELAY(SMA(C-Delay(C,1),9,1),1),12 ) - MEAN( DELAY(SMA(C-Delay(C,1),9,1),1),26 ), 10, 1)"""
    core = ops.SMA(ops.CLOSE - ops.DELAY(ops.CLOSE, 1), 9, 1)
    x = ops.DELAY(core, 1)
    diff = ops.MEAN(x, 12) - ops.MEAN(x, 26)
    return ops.CS(ops.SMA(diff, 10, 1))


def alpha170(ops: Ops) -> pd.Series:
    """
    (((RANK(1/C)*V)/MEAN(V,20)) * ((H*RANK(H-C))/(SUM(H,5)/5))) - RANK(VWAP - DELAY(VWAP,5))
    """
    t1 = ops._safe_div(ops.RANK(ops._safe_div(1.0, ops.CLOSE)) * ops.VOLUME, ops.MEAN(ops.VOLUME, 20))
    t2 = ops._safe_div(ops.HIGH * ops.RANK(ops.HIGH - ops.CLOSE), ops.SUM(ops.HIGH, 5) / 5.0)
    t3 = ops.RANK(ops.VWAP - ops.DELAY(ops.VWAP, 5))
    return ops.CS(t1 * t2 - t3)


def alpha171(ops: Ops) -> pd.Series:
    """(-1 *((LOW-CLOSE)*(OPEN^5))) / ((CLOSE-HIGH)*(CLOSE^5))"""
    num = (ops.LOW - ops.CLOSE) * ops.OPEN.pow(5)
    den = (ops.CLOSE - ops.HIGH) * ops.CLOSE.pow(5)
    return ops.CS(-ops._safe_div(num, den))


def alpha172(ops: Ops) -> pd.Series:
    """
    DX(14) 的 6日均值：|+DI - -DI|/(+DI + -DI)*100 的 6日均
    +DM = (HD>0 & HD>LD) ? HD : 0 ,  HD=H-H1
    -DM = (LD>0 & LD>HD) ? LD : 0 ,  LD=L1-L
    TR = max(H-L, |H-C1|, |L-C1|)
    """
    H1, C1, L1 = ops.DELAY(ops.HIGH, 1), ops.DELAY(ops.CLOSE, 1), ops.DELAY(ops.LOW, 1)
    HD = ops.HIGH - H1
    LD = L1 - ops.LOW
    TR = pd.DataFrame(np.maximum.reduce([
        (ops.HIGH - ops.LOW).to_numpy(),
        (ops.HIGH - C1).abs().to_numpy(),
        (ops.LOW - C1).abs().to_numpy()
    ]), index=ops.HIGH.index, columns=ops.HIGH.columns)
    pos_dm = ops.IF_WIDE((HD > 0) & (HD > LD), HD, 0.0)
    neg_dm = ops.IF_WIDE((LD > 0) & (LD > HD), LD, 0.0)
    tr14   = ops.SUM(TR, 14)
    pdi = ops._safe_div(ops.SUM(pos_dm, 14) * 100.0, tr14)
    ndi = ops._safe_div(ops.SUM(neg_dm, 14) * 100.0, tr14)
    dx  = ops._safe_div((pdi - ndi).abs() * 100.0, pdi + ndi)
    return ops.CS(ops.MEAN(dx, 6))


def alpha173(ops: Ops) -> pd.Series:
    """3*SMA(C,13,2) - 2*SMA(SMA(C,13,2),13,2) + SMA(SMA(SMA(LOG(C),13,2),13,2),13,2)"""
    s1 = ops.SMA(ops.CLOSE, 13, 2)
    term = 3.0 * s1 - 2.0 * ops.SMA(s1, 13, 2)
    log_s = ops.SMA(ops.SMA(ops.SMA(ops.LOG(ops.CLOSE), 13, 2), 13, 2), 13, 2)
    return ops.CS(term + log_s)


def alpha174(ops: Ops) -> pd.Series:
    """SMA( (C<=C1 ? STD(C,20) : 0), 20, 1 )  —— 与 160 同口径"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    val = ops.IF_WIDE(ops.CLOSE <= c1, ops.STD(ops.CLOSE, 20), 0.0)
    return ops.CS(ops.SMA(val, 20, 1))


def alpha175(ops: Ops) -> pd.Series:
    """MEAN( MAX(MAX(H-L, |C1-H|), |C1-L|), 6 )"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    a = ops.HIGH - ops.LOW
    b = (c1 - ops.HIGH).abs()
    c = (c1 - ops.LOW).abs()
    mx = pd.DataFrame(np.maximum.reduce([a.to_numpy(), b.to_numpy(), c.to_numpy()]),
                      index=a.index, columns=a.columns)
    return ops.CS(ops.MEAN(mx, 6))


def alpha176(ops: Ops) -> pd.Series:
    """CORR( RANK((C - TSMIN(L,12)) / (TSMAX(H,12) - TSMIN(L,12))), RANK(V), 6 )"""
    lo12 = ops.TSMIN(ops.LOW, 12)
    hi12 = ops.TSMAX(ops.HIGH, 12)
    rng  = ops._safe_div(ops.CLOSE - lo12, hi12 - lo12)
    return ops.CS(ops.CORR(ops.XRANK(rng), ops.XRANK(ops.VOLUME), 6))


def alpha177(ops: Ops) -> pd.Series:
    """((20 - HIGHDAY(H,20))/20) * 100"""
    return ops.CS((20.0 - ops.HIGHDAY(ops.HIGH, 20)) / 20.0 * 100.0)


def alpha178(ops: Ops) -> pd.Series:
    """((C - C1)/C1) * V"""
    c1 = ops.DELAY(ops.CLOSE, 1)
    ret = ops._safe_div(ops.CLOSE - c1, c1)
    return ops.CS(ret * ops.VOLUME)


def alpha179(ops: Ops) -> pd.Series:
    """RANK(CORR(VWAP, V, 4)) * RANK(CORR(RANK(LOW), RANK(MEAN(V,50)), 12))"""
    t1 = ops.RANK(ops.CORR(ops.VWAP, ops.VOLUME, 4))
    t2 = ops.RANK(ops.CORR(ops.XRANK(ops.LOW), ops.XRANK(ops.MEAN(ops.VOLUME, 50)), 12))


    return ops.CS(t1 * t2)


def alpha180(ops: Ops) -> pd.Series:
    """ if MEAN(V,20) < V then -(TSRANK(|ΔC7|,60))*SIGN(ΔC7) else -V """
    cond = ops.MEAN(ops.VOLUME, 20) < ops.VOLUME
    d7   = ops.DELTA(ops.CLOSE, 7)
    fast = -ops.TSRANK(d7.abs(), 60) * ops.SIGN(d7)
    slow = -ops.VOLUME
    return ops.CS(ops.IF_WIDE(cond, fast, slow))

def alpha182(ops: Ops) -> pd.Series:
    """
    COUNT(((C>O & BenchC>BenchO) | (C<O & BenchC<BenchO)), 20) / 20
    """
    c_gt_o = ops.CLOSE > ops.OPEN
    c_lt_o = ops.CLOSE < ops.OPEN
    b_gt_o = ops.BANCHMARKINDEXCLOSE > ops.BANCHMARKINDEXOPEN
    b_lt_o = ops.BANCHMARKINDEXCLOSE < ops.BANCHMARKINDEXOPEN
    cond = (c_gt_o & b_gt_o) | (c_lt_o & b_lt_o)
    return ops.CS(ops._safe_div(ops.COUNT(cond, 20), 20.0))


def alpha184(ops: Ops) -> pd.Series:
    """RANK(CORR(DELAY(OPEN-CLOSE,1), CLOSE, 200)) + RANK(OPEN - CLOSE)"""
    x = ops.DELAY(ops.OPEN - ops.CLOSE, 1)
    y = ops.CLOSE
    return ops.RANK(ops.CORR(x, y, 200)) + ops.RANK(ops.OPEN - ops.CLOSE)


def alpha185(ops: Ops) -> pd.Series:
    """RANK( - (1 - OPEN/CLOSE)^2 )"""
    ratio = ops._safe_div(ops.OPEN, ops.CLOSE)
    core = - (1.0 - ratio).pow(2)
    return ops.RANK(core)


def alpha186(ops: Ops) -> pd.Series:
    """
    ADX(14) 的 6日均与其 6日延迟的均值：
    out = ( MEAN(DX14,6) + DELAY(MEAN(DX14,6), 6) ) / 2
    其中 DX14 = |+DI - -DI|/(+DI + -DI)*100
    """
    H1, C1, L1 = ops.DELAY(ops.HIGH, 1), ops.DELAY(ops.CLOSE, 1), ops.DELAY(ops.LOW, 1)
    HD = ops.HIGH - H1
    LD = L1 - ops.LOW
    TR = pd.DataFrame(np.maximum.reduce([
        (ops.HIGH - ops.LOW).to_numpy(),
        (ops.HIGH - C1).abs().to_numpy(),
        (ops.LOW  - C1).abs().to_numpy(),
    ]), index=ops.HIGH.index, columns=ops.HIGH.columns)
    pos_dm = ops.IF_WIDE((HD > 0) & (HD > LD), HD, 0.0)
    neg_dm = ops.IF_WIDE((LD > 0) & (LD > HD), LD, 0.0)
    tr14   = ops.SUM(TR, 14)
    pdi = ops._safe_div(ops.SUM(pos_dm, 14) * 100.0, tr14)
    ndi = ops._safe_div(ops.SUM(neg_dm, 14) * 100.0, tr14)
    dx  = ops._safe_div((pdi - ndi).abs() * 100.0, pdi + ndi)
    adx6 = ops.MEAN(dx, 6)
    return ops.CS((adx6 + ops.DELAY(adx6, 6)) / 2.0)


def alpha187(ops: Ops) -> pd.Series:
    """SUM( OPEN<=Delay(OPEN,1) ? 0 : MAX(H-O, O-Delay(O,1)) , 20)"""
    o1 = ops.DELAY(ops.OPEN, 1)
    comp = pd.DataFrame(np.maximum((ops.HIGH - ops.OPEN).to_numpy(),
                                   (ops.OPEN - o1).to_numpy()),
                        index=ops.OPEN.index, columns=ops.OPEN.columns)
    x = ops.IF_WIDE(ops.OPEN <= o1, 0.0, comp)
    return ops.CS(ops.SUM(x, 20))


def alpha188(ops: Ops) -> pd.Series:
    """( (H-L) - SMA(H-L,11,2) ) / SMA(H-L,11,2) * 100"""
    x = ops.HIGH - ops.LOW
    ma = ops.SMA(x, 11, 2)
    return ops.CS(ops._safe_div(x - ma, ma) * 100.0)


def alpha189(ops: Ops) -> pd.Series:
    """MEAN( |C - MEAN(C,6)| , 6 )"""
    mu6 = ops.MEAN(ops.CLOSE, 6)
    return ops.CS(ops.MEAN((ops.CLOSE - mu6).abs(), 6))


def alpha190(ops: Ops) -> pd.Series:
    """
    LOG( ((COUNT(ret>g,20)-1) * SUMIF((ret-g)^2,20,ret<g))
         / ( COUNT(ret<g,20) * SUMIF((ret-g)^2,20,ret>g) ) )
    其中 g = (C/Delay(C,19))^(1/20) - 1, ret = C/Delay(C,1) - 1
    """
    ret = ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 1)) - 1.0
    g   = (ops._safe_div(ops.CLOSE, ops.DELAY(ops.CLOSE, 19))).pow(1.0/20.0) - 1.0
    diff2 = (ret - g).pow(2)
    cnt_pos = ops.COUNT(ret > g, 20) - 1.0
    cnt_neg = ops.COUNT(ret < g, 20)
    sum_neg = ops.SUMIF(diff2, 20, ret < g)
    sum_pos = ops.SUMIF(diff2, 20, ret > g)
    num = cnt_pos * sum_neg
    den = cnt_neg * sum_pos
    return ops.CS(ops.LOG(ops._safe_div(num, den)))


def alpha191(ops: Ops) -> pd.Series:
    """CORR(MEAN(V,20), L, 5) + (H+L)/2 - C"""
    corr = ops.CORR(ops.MEAN(ops.VOLUME, 20), ops.LOW, 5)
    mid  = (ops.HIGH + ops.LOW) / 2.0
    return ops.CS(corr + mid - ops.CLOSE)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------



# ============== Placeholders for not-yet-implemented factors =================
def _unimplemented(_ops: Ops) -> Series:
    return pd.Series(np.nan, index=pd.Index(_ops.codes, name="code"))

# 注册表：把已完成的放进来；未列出的 id 会视为“未实现”
FACTOR_FUNCS: Dict[int, Callable[[Ops], Series]] = {
    1: alpha1,2: alpha2,3: alpha3,4: alpha4,
    5: alpha5,6: alpha6,7: alpha7,8: alpha8,9: alpha9,
    # 11: alpha11, 12: alpha12, 13: alpha13, 14: alpha14, 15: alpha15,
    # 16: alpha16, 17: alpha17, 18: alpha18, 19: alpha19, 20: alpha20,
    # 21: alpha21, 22: alpha22, 23: alpha23, 24: alpha24, 25: alpha25,
    # 26: alpha26, 27: alpha27, 28: alpha28, 29: alpha29, 31: alpha31,
    # 32: alpha32, 33: alpha33, 34: alpha34, 35: alpha35,

    # 37: alpha37, 38: alpha38, 39: alpha39, 40: alpha40, 41: alpha41,
    #
    # 42: alpha42, 43: alpha43, 44: alpha44, 45: alpha45, 46: alpha46,
    # 47: alpha47, 48: alpha48, 49: alpha49, 50: alpha50, 51: alpha51,
    # 52: alpha52, 53: alpha53, 54: alpha54, 55: alpha55, 56: alpha56,
    # 57: alpha57, 58: alpha58, 59: alpha59, 60: alpha60,
    #
    # 61: alpha61, 62: alpha62, 63: alpha63, 64: alpha64, 65: alpha65,
    # 66: alpha66, 67: alpha67, 68: alpha68, 69: alpha69, 70: alpha70,
    # 71: alpha71, 72: alpha72, 73: alpha73, 74: alpha74, 75: alpha75,
    # 76: alpha76, 77: alpha77, 78: alpha78, 79: alpha79, 80: alpha80,
    #
    # 81: alpha81, 82: alpha82, 83: alpha83, 84: alpha84, 85: alpha85,
    # 86: alpha86, 87: alpha87, 88: alpha88, 89: alpha89, 90: alpha90,
    # 91: alpha91, 92: alpha92, 93: alpha93, 94: alpha94, 95: alpha95,
    # 96: alpha96, 97: alpha97, 98: alpha98, 99: alpha99, 100: alpha100,
    #
    # 101: alpha101, 102: alpha102, 103: alpha103, 104: alpha104, 105: alpha105,
    # 106: alpha106, 107: alpha107, 108: alpha108, 109: alpha109, 110: alpha110,
    # 111: alpha111, 112: alpha112, 113: alpha113, 114: alpha114, 115: alpha115,
    # 116: alpha116, 117: alpha117, 118: alpha118, 119: alpha119, 120: alpha120,
    # 121: alpha121, 122: alpha122, 123: alpha123, 124: alpha124, 125: alpha125,
    # 126: alpha126,
    #
    # 128: alpha128, 129: alpha129, 130: alpha130, 131: alpha131, 132: alpha132,
    # 133: alpha133, 134: alpha134, 135: alpha135, 136: alpha136, 137: alpha137, 138: alpha138,
    # 139: alpha139, 140: alpha140, 141: alpha141, 142: alpha142, 143: alpha143, 144: alpha144,
    # 145: alpha145,
    #
    # 147: alpha147, 148: alpha148, 149: alpha149, 150: alpha150, 151: alpha151, 152: alpha152,
    # 153: alpha153, 154: alpha154, 155: alpha155, 156: alpha156, 157: alpha157, 158: alpha158,
    # 159: alpha159, 160: alpha160, 161: alpha161, 162: alpha162, 163: alpha163, 164: alpha164,
    #
    # 166: alpha166, 167: alpha167, 168: alpha168, 169: alpha169, 170: alpha170,
    # 171: alpha171, 172: alpha172, 173: alpha173, 174: alpha174, 175: alpha175,
    # 176: alpha176, 177: alpha177, 178: alpha178, 179: alpha179, 180: alpha180,
    #
    # 182: alpha182,
    # 184: alpha184, 185: alpha185, 186: alpha186, 187: alpha187, 188: alpha188,
    # 189: alpha189, 190: alpha190, 191: alpha191


}

# 若你已有“确认禁用”的 8 个编号（原 YAML 缺口），可以固定在此：
KNOWN_DISABLED = {10, 30, 36, 127, 146, 165, 181, 183}


# =========================
# Public API (契约函数)
# =========================
def factor_exposures_191(date: pd.Timestamp,
                         panel: pd.DataFrame,
                         codes: List[str]) -> pd.DataFrame:
    """
    Compute cross-sectional factor exposures f1..f191 on `date` using pre-defined alpha functions.
    No YAML. No look-ahead. No cleaning here.
    """
    t0 = time.time()
    t = pd.Timestamp(date).normalize()
    codes = list(dict.fromkeys([str(c) for c in codes]))

    log.step("[F191] (static) build ops & evaluate alpha functions ...")
    ops = Ops(t, panel, codes)

    # 固定输出列
    cols = [f"f{i}" for i in range(1, 191 + 1)]
    out = pd.DataFrame(np.nan, index=pd.Index(codes, name="code"), columns=cols, dtype=float)

    # 计算禁用/未实现集合
    implemented_ids = set(FACTOR_FUNCS.keys())
    all_ids = set(range(1, 191 + 1))
    unimpl_ids = sorted(list(all_ids - implemented_ids))
    disabled_ids = sorted(list(KNOWN_DISABLED.union(all_ids - implemented_ids)))

    if disabled_ids:
        log.step(f"[INFO] [F191] DISABLED_IDS={disabled_ids}")

    ok = fail = 0
    total = 191
    t_loop = time.time()

    for i in range(1, total + 1):
        col = f"f{i}"
        fn = FACTOR_FUNCS.get(i, _unimplemented)
        try:
            s = fn(ops)
            out[col] = pd.to_numeric(s, errors="coerce")
            ok += 1
        except Exception as e:
            fail += 1
            log.error(f"[F191] F{i} -> {type(e).__name__}: {e}")

        if i % 10 == 0 or i == total:
            t_loop = log.loop_progress("[F191] eval", i, total, start_time=t_loop, every=10)

    # 缺失字段一次性 WARN
    if ops.missing_columns:
        log.warn(f"[F191] MISSING_COLUMNS={ops.missing_columns}")

    out = clip_inf_nan(out)
    log.debug(f"[F191][ROLL] builds={ops._stat_build}, hits={ops._stat_hit}")
    log.done(f"[F191] time={time.time()-t0:.2f}s ok={ok} fail={fail} disabled={len(disabled_ids)}")
    return out


# =========================
# Smoke Test
# =========================
if __name__ == "__main__":  # pragma: no cover
    try:
        log.set_verbosity("STEP")
    except Exception:
        pass

    rng = np.random.default_rng(191)
    # 3 stocks × 60 business days random walk
    # ……前面代码不变……

    dates = pd.bdate_range("2024-01-02", periods=60)
    codes = ["600001.SH", "000002.SZ", "300003.SZ"]

    # === 新增：先在循环外合成一条基准指数序列（全市场同一条） ===
    bench_close = 1000 + rng.normal(0, 1.5, size=len(dates)).cumsum()
    bench_open = bench_close * (1 + rng.normal(0, 0.001, size=len(dates)))

    panel_list = []
    for c in codes:
        px = 10 + rng.normal(0, 0.1, size=len(dates)).cumsum()
        hi = px + rng.normal(0.02, 0.02, size=len(dates))
        lo = px - rng.normal(0.02, 0.02, size=len(dates))
        vw = px + rng.normal(0.0, 0.02, size=len(dates))
        vol = rng.integers(1_000, 5_000, size=len(dates))
        df = pd.DataFrame({
            "date": dates, "code": c,
            "open": px * (1 + rng.normal(0, 0.005, size=len(dates))),
            "high": hi, "low": lo, "close": px,
            "vwap": vw, "volume": vol,
            # === 新增：把基准列并到每只股票的行里（同一条指数，按日复制） ===
            "BANCHMARKINDEXOPEN": bench_open,
            "BANCHMARKINDEXCLOSE": bench_close,
        })
        df["amount"] = (df["vwap"] * df["volume"]).astype(float)
        panel_list.append(df)

    panel = pd.concat(panel_list, ignore_index=True)

    # ……后面代码不变……

    t = dates[-1]
    res = factor_exposures_191(t, panel, codes)





    # 基础检查
    assert res.shape == (len(codes), 191), f"shape mismatch: {res.shape}"
    assert not np.isinf(res.to_numpy()).any(), "inf detected in result"

    # 统计全 NaN 列
    all_nan_mask = res.isna().all(axis=0)
    all_nan_cols = res.columns[all_nan_mask].tolist()
    n_all_nan = len(all_nan_cols)

    # 历史长度（交易日数）
    n_days = panel["date"].nunique()

    # ——【新增】前 80 条的“手工暖机需求”映射（用于判断是否因为没暖机导致 NaN）
    # 说明：None 表示该编号删除/未用，不参与暖机判定
    WARMUP_NEED_1_80 = {
        1:7,  2:2,  3:7,  4:20, 5:11, 6:5,  7:4,  8:5,  9:8,  10:None, 11:6, 12:10,
        13:1, 14:6, 15:2, 16:9, 17:15,18:6,19:6, 20:7, 21:11, 22:20,  23:39, 24:10,
        25:250, 26:236, 27:18, 28:13, 29:7, 30:None, 31:12, 32:5, 33:240, 34:12,
        35:23, 36:None, 37:15, 38:20, 39:240, 40:27, 41:8, 42:10, 43:7, 44:27,
        45:164, 46:24, 47:14, 48:20, 49:13, 50:13, 51:13, 52:27, 53:13, 54:10,
        55:21, 56:70, 57:11, 58:21, 59:21, 60:20, 61:103, 62:5, 63:6, 64:88,
        65:6, 66:6, 67:24, 68:16, 69:21, 70:6, 71:24, 72:20, 73:35, 74:65,
        75:50, 76:21, 77:47, 78:23, 79:12, 80:6 , 81: 21 , 82: 20,83: 5,84: 20,85: 20,
        86: 20,87: 11,88: 20,89: 27,90: 5,91: 40,92: 180,93: 20,94: 30,95: 20,96: 9,97: 10,98: 200,99: 5,
        100: 20,101: 37,102: 6,103: 20,104: 20,105: 10,106: 20,107: 1,108: 120,109: 10,110: 20,111: 11,
        112: 12,113: 20,114: 5,115: 30,116: 20,117: 32,118: 20,119: 56,120: 1,121: 60,122: 13,123: 60,124: 30,
        125: 80,126: 1, 128: 14, 129: 12, 130: 40, 131: 50, 132: 20, 133: 20,
        134: 12, 135: 20, 136: 10, 137: 1,  138: 60, 139: 10,
        140: 60, 141: 15, 142: 20, #143：None
        144: 20, 145: 26,147: 12, 148: 60, 149: 252, 150: 1, 151: 20, 152: 26, 153: 24, 154: 180,
        155: 27, 156: 5, 157: 6, 158: 15, 159: 24, 160: 20, 161: 12, 162: 12,
        163: 20, 164: 13,166: 20, 167: 12, 168: 20, 169: 26, 170: 20, 171: 1, 172: 14, 173: 13, 174: 20,
        175: 6, 176: 12, 177: 20, 178: 1, 179: 50, 180: 60,182: 20, 184: 200, 185: 1, 186: 14,
        187: 20, 188: 11, 189: 6, 190: 20, 191: 20,
    }

    # ——【新增】未注册因子过滤：不在注册表内的编号不计入“前十全 NaN”
    try:
        implemented_ids = set(FACTOR_FUNCS.keys())
    except Exception:
        implemented_ids = set()  # 极端情况下 fallback

    # 分类：因为长度不够导致的全 NaN vs 其它原因（如输入列缺失、表达式错误）
    len_insufficient = []  # 仅统计 1..80 且实现了的
    other_reasons = []     # 过滤掉“未注册 & 暖机不足”的剩余 NaN
    unregistered = []      # 仅记录数量，不展示到 Top10

    for col in all_nan_cols:
        fid = int(col[1:])  # 'f16' -> 16

        # 未注册（未实现）的直接跳过 Top10，只计入统计
        if fid not in implemented_ids:
            unregistered.append(fid)
            continue

        # 仅对 1..80 应用暖机过滤
        need = WARMUP_NEED_1_80.get(fid, None)
        if (need is not None) and (n_days < need):
            len_insufficient.append((fid, need))
            continue

        # 其余进入“其它原因”池
        other_reasons.append(fid)


    # 打印“前 80 的暖机需要”清单（简版）
    pairs = ", ".join([f"f{fid}:{('—' if WARMUP_NEED_1_80.get(fid) is None else WARMUP_NEED_1_80[fid])}"
                       for fid in range(1, 81)])
    try:
        log.step(f"[SMOKE][F191] WarmupNeed f1..f80 -> {pairs}")
    except Exception:
        pass

    # 打印摘要（前十各取样）
    preview_len = [f"f{fid}(need≥{need})" for fid, need in len_insufficient[:20]]
    preview_oth = [f"f{fid}" for fid in other_reasons[:20]]
    log.done(
        f"[SMOKE][F191] OK shape={res.shape}, days={n_days}, all-NaN={n_all_nan}, "
        f"warmup_excluded={len(len_insufficient)} sample={preview_len}, "
        f"other={len(other_reasons)} sample={preview_oth}, "
        f"unregistered_excluded={len(unregistered)}"
    )
