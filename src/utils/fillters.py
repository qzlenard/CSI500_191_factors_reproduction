"""
[CN] 可交易性过滤契约：剔除停牌与涨/跌停。
[Purpose] Tradability filters for daily cross-sections.

Interfaces:
- tradable_codes(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str]) -> list[str]
  * Exclude paused==1
  * Exclude close >= high_limit or close <= low_limit
  * If limits missing, use thresholds (non-ST ±9.8%, ST ±4.8%) based on code or flags.

Notes:
- Must log counts: kept vs excluded, and reasons.
"""