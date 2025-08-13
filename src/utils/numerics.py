"""
[CN] 数学与稳健回归契约：winsor、z-score、Ridge/OLS 安全封装。
[Purpose] Numerical helpers used by neutralization & regression.

Interfaces:
- winsorize(s: pd.Series, pct: float) -> pd.Series
- zscore(s: pd.Series) -> pd.Series
- clip_inf_nan(df: pd.DataFrame) -> pd.DataFrame
- ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict
  * Returns {'coef': np.ndarray, 'intercept': float, 'r2': float, 'n': int}
- safe_lstsq(X: np.ndarray, y: np.ndarray) -> dict
  * Fallback OLS with checks; same return keys.

Contracts:
- Center/standardize X if required by caller (policy in caller).
- Robust to rank deficiency; never crash—return NaNs and WARN instead.
"""