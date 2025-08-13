"""
[CN] 风格因子契约：用财报快照 + 价格代理生成稳定的风格暴露，公告日感知 + 滞后。
[Purpose] Style exposures for cross-section regression RHS.

Signature:
def style_exposures(date: pd.Timestamp, codes: list[str],
                    panel: pd.DataFrame, funda_snapshot: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns DataFrame indexed by code with stable column set, e.g.:
    ["size","bp","ep_ttm","sp_ttm","growth_rev_yoy","growth_np_yoy",
     "leverage","roe_ttm","roa_ttm","cf_yield", ...]
    Cleaning: winsorize(WINSOR_PCT) + z-score if STANDARDIZE.
    Missing: small gaps may fall back to price-based proxies; must WARN in logs.
    '''

Notes:
- size = ln(float_mktcap); bp = book_value/market_cap; etc.
- Forward-fill daily between announcement snapshots; lag controlled by STYLE_LAG_DAYS.
"""