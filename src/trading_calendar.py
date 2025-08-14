"""
[CN] 交易日历契约：判断/推算交易日、窗口对齐。
[Purpose] Calendar utilities.

Interfaces:
- is_trade_day(date: pd.Timestamp) -> bool
- next_trade_day(date: pd.Timestamp, n: int=1) -> pd.Timestamp
- prev_trade_day(date: pd.Timestamp, n: int=1) -> pd.Timestamp
- trade_days_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]

Assumptions:
- Uses get_trade_days under the hood; behavior consistent with exchange calendar.
"""