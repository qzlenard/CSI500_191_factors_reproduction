# Cross-module constraints: see docs/contract_notes.md
"""
[CN] 正交化契约（逐日截面）：对每个因子在“行业+风格”上回归，取残差作为正交化暴露。
[Purpose] Orthogonalize factor exposures daily to industry & styles.

Signature:
def orthogonalize(factors: pd.DataFrame, styles: pd.DataFrame, inds: pd.DataFrame) -> pd.DataFrame:
    '''
    Inputs:
      - factors: index=code, cols=f1..f191 (cleaned)
      - styles:  index=code, stable style columns (cleaned)
      - inds:    index=code, industry dummies (drop-one or handle intercept)
    Returns:
      - residualized factors with same index/columns as `factors`.
    Regression:
      - Per factor cross-section: y = factor; X = [inds, styles]; Ridge(alpha=RIDGE_ALPHA).
      - Skip if N < (p + 5); record skip + reason; log R² and N for successful fits.
    Filtering:
      - Exclude paused and limit-up/down names (utils/filters.tradable_codes).
    '''

Notes:
- Align on intersection of codes with valid y and X; drop all-NaN columns.
- Persist out/residuals/YYYYMMDD_residuals.csv
"""
