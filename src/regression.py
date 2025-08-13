# Cross-module constraints: see docs/contract_notes.md
"""
[CN] 横截面回归契约：两日收益 ~ 行业 + 风格 + 残差因子 → 因子收益率（逐日）。
[Purpose] Estimate daily factor returns (betas) from cross-sectional regression.

Signatures:
def forward_return(panel: pd.DataFrame, date: pd.Timestamp, codes: list[str], fwd_days: int=2) -> pd.Series:
    '''
    [CN] 对齐当日 t 暴露与未来 t→t+fwd_days 的简单收益：P_{t+f}/P_t - 1。
    Returns pd.Series indexed by code with forward simple returns.
    '''

def cs_factor_returns(y: pd.Series, styles: pd.DataFrame, inds: pd.DataFrame, resid_f: pd.DataFrame,
                      use_ridge: bool=True, ridge_alpha: float=1e-6) -> pd.Series:
    '''
    [CN] 逐日横截面：被解释变量 y=两日收益；解释变量 X=[inds, styles, resid_f]。
    Returns pd.Series of factor returns for columns resid_f (f1...f191) only.
    Regression:
      - Ridge with alpha=ridge_alpha (≈ OLS stabilization).
      - Skip if N < (p + 5); WARN and record metrics with NaN for that day.
    Logging:
      - Record R², N, effective columns used; write to out/ts/metrics.csv and out/ts/coverage.csv.
    Persistence:
      - Append factor returns row to out/ts/factor_returns.csv with rolling keep_last.
    '''

Hard Constraints:
- Cleaning: winsorize + z-score applied to X blocks (styles & residual factors) prior to fit.
- Sample: filter tradable universe (no paused, no limit-up/down).
"""