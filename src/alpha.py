# Cross-module constraints: see docs/contract_notes.md
"""
[CN] Alpha 生成契约：过去一年（lookback）两日因子收益率的均值作为下期预测，对当期残差截面加权聚合为 Alpha。
[Purpose] Turn factor return history into next-period alpha cross-section.

Signature:
def next_alpha_from_trailing_mean(resid_f_today: pd.DataFrame, factor_returns_path: str,
                                  lookback_days: int, codes: list[str]) -> pd.Series:
    '''
    Inputs:
      - resid_f_today: index=code, cols=f1..f191 (today’s residualized exposures)
      - factor_returns_path: CSV "out/ts/factor_returns.csv"
      - lookback_days: e.g., LOOKBACK_FOR_ALPHA (default 252)
      - codes: target codes to output
    Returns:
      - pd.Series alpha indexed by code (float), AFTER aligning columns and cleaning.
    Logging:
      - [ALPHA] saved K names, coverage %, lookback used, any NaNs dropped.
    Persistence:
      - Write out/alpha/YYYYMMDD_alpha.csv
    '''
"""