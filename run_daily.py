"""
[CN] 简要口径：对齐 t 的暴露与 r_{t→t+2}；剔除停牌与涨/跌停；winsorize(1%)+z-score；回归用 Ridge(α 见 config)。
[EN] Summary: Align exposures at t with forward r_{t→t+2}; exclude paused/limit-hit; winsorize(1%)+z-score; regression uses Ridge (alpha in config).
[CN] 时序文件滚动保留 WINDOW_TRADING_DAYS；仅处理新增交易日；若 end 为交易日则返回并落盘当日调仓指令。
[EN] Time series use rolling keep_last=WINDOW_TRADING_DAYS; incremental-only; if `end` is a trade day, return & write today's orders.
[Link] Cross-module constraints: see docs/contract_notes.md
"""

"""
[CN] 主入口契约：串联步骤1→4，增量处理，仅处理“新增交易日”；若 end 为交易日，最终 return 为当日调仓指令 DataFrame 并落盘 CSV。
[Purpose] Orchestrate the end-to-end daily pipeline with progress logging and rolling storage.

CLI / API:
- main(start: str, end: str) -> pd.DataFrame
  * start/end in "YYYY-MM-DD" (inclusive); processes only trade days within.
  * Uses state/manifest.json to find last processed date; performs incremental update.
Return:
- If `end` is a valid trade day: returns today's orders DataFrame (schema below).
- Otherwise: returns an empty DataFrame with correct columns.

Side Effects & Contracts:
- Writes progress logs to out/logs/.
- Appends time series with rolling window keep_last=WINDOW_TRADING_DAYS:
  out/ts/factor_returns.csv, out/ts/coverage.csv, out/ts/metrics.csv
- Writes daily cross-sections:
  out/residuals/YYYYMMDD_residuals.csv
  out/alpha/YYYYMMDD_alpha.csv
- Writes orders for trade day:
  out/orders/YYYYMMDD_orders.csv

CSV Schemas:
- data/raw/ohlcv/{code}.csv: columns = [date, code, open, high, low, close, volume, amount?, preclose?, paused?, high_limit?, low_limit?]
- out/residuals/YYYYMMDD_residuals.csv: [code, f1..f191]
- out/ts/factor_returns.csv: [date, f1..f191]
- out/alpha/YYYYMMDD_alpha.csv: [code, alpha]
- out/orders/YYYYMMDD_orders.csv: [date, code, target_weight, side, px_type, note]

Logging (utils/logging.py):
- STEP/LOOP/DEBUG levels; print R², N, coverage, ETA; warn on data gaps & skips.
"""