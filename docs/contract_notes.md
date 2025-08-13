"""
Cross-Module Hard Constraints (recap):
- Alignment: exposures at t vs forward return r_{t→t+2}.
- Filtering: remove paused & limit-hit names before regressions.
- Cleaning: winsorize(1%) + z-score on factor & style exposures.
- Regression: Ridge(alpha=RIDGE_ALPHA) with skip rule N < (p + 5).
- Rolling Storage: append_with_rolloff(..., keep_last=WINDOW_TRADING_DAYS).

[CN] 上述契约一经确认，后续实现严格遵循；任何改动需先改本契约并在宏观答疑确认一致性。
"""
