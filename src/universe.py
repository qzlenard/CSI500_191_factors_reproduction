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