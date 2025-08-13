"""
[CN] 191 因子契约：输出 191 列暴露，允许 NaN；每个因子内部决定最小窗长；当日按 codes 与 panel 对齐。
[Purpose] Compute raw 191-factor exposures prior to orthogonalization.

Signature:
def factor_exposures_191(date: pd.Timestamp, panel: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code with columns f1..f191.
    NaNs allowed (insufficient history); columns fixed and ordered.
    Cleaning: winsorize(WINSOR_PCT) + z-score if STANDARDIZE.
    '''

Notes:
- Factor formulas follow the bank’s 191 definitions (to be implemented).
- Window requirements differ per factor; log effective sample size per day.
"""
