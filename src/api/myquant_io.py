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