# Cross-module constraints: see docs/contract_notes.md
"""
[CN] 组合构建契约：基于 Alpha 截面生成今日调仓指令；默认 long_only_topk=50、max_weight=5%。
[Purpose] Convert alpha cross-section into today’s target weights and orders.

Signature:
def build_orders_from_alpha(alpha: pd.Series,
                            mode: str="long_only_topk", top_k: int=50, max_weight: float=0.05,
                            neutral: bool=False, turnover_threshold: float=0.01,
                            prev_target_weights: Optional[pd.Series]=None) -> pd.DataFrame:
    '''
    Inputs:
      - alpha: pd.Series indexed by code
      - mode: portfolio construction mode (default "long_only_topk")
      - top_k: default from config (e.g., 50)
      - max_weight: per-name cap (e.g., 0.05)
      - neutral: if True, dollar/industry neutrality policy TBD (future extension)
      - turnover_threshold: skip rebalance if below (e.g., 0.01)
      - prev_target_weights: optional Series of last targets to compute turnover
    Returns:
      - orders DataFrame with columns [date, code, target_weight, side, px_type, note]
    Logging:
      - [ORDERS] K, turnover %, neutrality flags; warnings on caps & ties.
    Persistence:
      - Write out/orders/YYYYMMDD_orders.csv
    '''

Notes:
- side ∈ {"BUY","SELL","HOLD"}; px_type ∈ {"MKT","CLOSE"} (policy TBD).
"""