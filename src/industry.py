"""
[CN] 行业哑变量契约：当日对 codes 生成行业 One-Hot。
[Purpose] Provide industry dummy matrix aligned with codes.

Signature:
def industry_dummies(date: pd.Timestamp, codes: list[str]) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code, columns = ["IND_xxx" ...], 0/1 dummies.
    Must include an intercept handling policy: either drop-one or center later in regression.
    '''

Notes:
- Industry taxonomy (e.g., Shenwan, CSRC) to be fixed in config.
- Stable column naming across days; unseen industries → zero columns (or skipped).
"""