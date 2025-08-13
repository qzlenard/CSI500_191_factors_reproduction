"""
[CN] 取数与清洗契约：拉取一年日频 OHLCV，按股票 CSV 落地，必要字段兜底。
[Purpose] Collect & clean OHLCV; persist per-code CSV; build in-memory panel.

Interfaces:
- collect_and_store_ohlcv(codes: list[str], start: str, end: str, fq: str, out_dir: str) -> None
  Writes data/raw/ohlcv/{code}.csv (append/incremental).
- build_panel_from_csv(codes: list[str], start: str, end: str, raw_dir: str) -> pd.DataFrame
  Returns a panel-like long DataFrame indexed by date with 'code' column.

Cleaning Rules (Hard Constraints):
- Fill/derive paused, high_limit, low_limit if missing (use thresholds for non-ST ±9.8%, ST ±4.8%).
- Ensure monotonic dates; drop duplicates; clip inf/nan (utils/numerics.py).
- Log coverage and missing-field fallbacks.
"""