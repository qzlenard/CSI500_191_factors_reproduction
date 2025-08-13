"""
[CN] 配置模块契约（仅列键名与含义；数值稍后再定，一处配置全局引用）
[Purpose] Centralized configuration keys used across the pipeline. No logic here.

CONFIG KEYS (to be defined later):
- INDEX_CODE: str                      # e.g., "000905.SH" (CSI500)
- DATA_DIR: str                        # base data directory, e.g., "data/"
- OUT_DIR: str                         # base output directory, e.g., "out/"
- STATE_DIR: str                       # e.g., "state/"
- WINDOW_TRADING_DAYS: int             # rolling keep_last for time series, e.g., 252
- FORWARD_DAYS: int                    # forward return horizon (default 2)
- WINSOR_PCT: float                    # winsorize tail fraction per side, e.g., 0.01
- STANDARDIZE: bool                    # apply z-score after winsorization
- RIDGE_ALPHA: float                   # ridge regularization for cross-sectional regressions, e.g., 1e-6
- STYLE_LAG_DAYS: int                  # fundamentals snapshot lag (announcement-aware)
- LOOKBACK_FOR_ALPHA: int              # lookback for trailing mean of factor returns, e.g., 252
- PORTFOLIO_MODE: str                  # "long_only_topk" | "long_short_beta_neutral" | etc.
- PORTFOLIO_TOP_K: int                 # default 50
- PORTFOLIO_MAX_WEIGHT: float          # default 0.05 (5%)
- TURNOVER_THRESHOLD: float            # skip rebalance if turnover below this, e.g., 0.01
- LOG_VERBOSITY: str                   # "SILENT" | "STEP" | "LOOP" | "DEBUG"
- FQ: str                              # price adjustment mode, e.g., "pre" (forward-adjusted)
- RANDOM_SEED: int                     # reproducibility for any randomized ops
Assumptions:
- All paths are relative to project root unless absolute.
- China A-share trading calendar is assumed (exchange holiday rules).
"""
from __future__ import annotations

# file: config.py
"""
Project: CSI500 191-Factor Pipeline (placeholder config)
Role: Centralized constants, paths, and knobs for the whole project.
Note: This is a scaffold (placeholders). Values are provisional and may be tuned later.

All code and comments are in English by design.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import json
from typing import Optional, List, Dict, Any


# -------------------------
# Verbosity / Logging Level
# -------------------------
class Verbosity:
    """Verbosity levels used by utils/logging.py"""
    SILENT = "SILENT"
    STEP = "STEP"      # step-level logs (default)
    LOOP = "LOOP"      # show loop progress bars / ETA
    DEBUG = "DEBUG"    # very chatty


# -------------------------
# Paths & Layout
# -------------------------
@dataclass
class PathsConfig:
    """Filesystem layout. All paths are relative to project root by default."""
    # Infer project root as the directory that contains this file.
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Data areas (input-like)
    data_dir: Path = field(init=False)
    raw_ohlcv_dir: Path = field(init=False)
    ref_dir: Path = field(init=False)

    # Output areas (artifacts)
    out_dir: Path = field(init=False)
    out_residuals_dir: Path = field(init=False)
    out_alpha_dir: Path = field(init=False)
    out_orders_dir: Path = field(init=False)
    out_ts_dir: Path = field(init=False)
    out_logs_dir: Path = field(init=False)

    # State
    state_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    locks_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_ohlcv_dir = self.data_dir / "raw" / "ohlcv"
        self.ref_dir = self.data_dir / "ref"

        self.out_dir = self.project_root / "out"
        self.out_residuals_dir = self.out_dir / "residuals"
        self.out_alpha_dir = self.out_dir / "alpha"
        self.out_orders_dir = self.out_dir / "orders"
        self.out_ts_dir = self.out_dir / "ts"
        self.out_logs_dir = self.out_dir / "logs"

        self.state_dir = self.project_root / "state"
        self.manifest_path = self.state_dir / "manifest.json"
        self.locks_dir = self.state_dir / "locks"


# -------------------------
# Universe / Calendar
# -------------------------
@dataclass
class UniverseConfig:
    """Index membership and calendar settings."""
    index_code: str = "000905.SH"  # CSI 500
    exchange_calendar: str = "SSE"  # placeholder tag for Shanghai exchange calendar
    timezone: str = "Asia/Shanghai"  # trading timezone for A-share
    # Window for daily job
    lookback_trading_days: int = 252  # one trading year
    # Forward return horizon (t -> t+2)
    forward_days: int = 2


# -------------------------
# Data Fetch & Cleaning
# -------------------------
@dataclass
class FetchConfig:
    """
    Data acquisition knobs for src/api/myquant_io.py
    Note: fq denotes price adjustment. 'qfq' ~ pre-adjusted (前复权).
    """
    price_adjust: str = "qfq"   # {'none','qfq','hfq'} placeholder
    # If we need a small buffer to ensure continuity (non-trading days, etc.)
    calendar_buffer_days: int = 10
    # Per-call chunk size to avoid overloading API
    batch_size: int = 100
    # Light retry policy placeholders
    max_retries: int = 3
    retry_backoff_sec: float = 1.0

    # Limit-up / down filtering thresholds when high/low limits missing
    limit_up_nonst: float = 0.098  # ~ 9.8%
    limit_dn_nonst: float = -0.098
    limit_up_st: float = 0.048     # ~ 4.8%
    limit_dn_st: float = -0.048

    # Optional: API token env key (placeholder)
    myquant_token_env: str = "MYQUANT_TOKEN"


@dataclass
class CleaningConfig:
    """Winsorization / standardization placeholders."""
    winsor_pct: float = 0.01  # 1% two-sided
    zscore_ddof: int = 0      # standard zscore, ddof=0


# -------------------------
# Style (Fundamentals) Factors
# -------------------------
@dataclass
class StyleConfig:
    """
    Financial statement styles snapshot settings (announcement-aware lag).
    """
    lag_trading_days: int = 3         # placeholder: snapshot no later than t-3
    fallback_with_price_proxy: bool = True  # allow sparse fallback with price proxies
    required_columns: List[str] = field(default_factory=lambda: [
        "size", "bp", "ep_ttm", "sp_ttm",
        "growth_rev_yoy", "growth_np_yoy",
        "leverage", "roe_ttm", "roa_ttm", "cf_yield",
    ])


# -------------------------
# 191 Factors (Exposures)
# -------------------------
@dataclass
class FactorConfig:
    """191 factor library placeholders."""
    n_factors: int = 191
    # We will emit f1..f191 as canonical column names.
    names: List[str] = field(default_factory=lambda: [f"f{i}" for i in range(1, 192)])
    # Some factors may have minimal internal window requirements (placeholder hint).
    min_window_by_factor: Dict[str, int] = field(default_factory=dict)


# -------------------------
# Neutralization (Step 2)
# -------------------------
@dataclass
class NeutralizeConfig:
    """Regression settings for orthogonalization on styles + industry dummies."""
    use_ridge: bool = True
    ridge_alpha: float = 1e-6
    # Minimum sample requirement: N >= (p + safety_margin)
    safety_margin: int = 5
    # Debug dump toggle (placeholder, not implemented)
    dump_design_matrix: bool = False


# -------------------------
# Cross-Sectional Regression (Step 3)
# -------------------------
@dataclass
class RegressionConfig:
    """Cross-sectional regression for factor returns."""
    use_ridge: bool = True
    ridge_alpha: float = 1e-6
    safety_margin: int = 5
    record_metrics: bool = True  # R^2, N, resid variance, etc.


# -------------------------
# Alpha (Step 4)
# -------------------------
@dataclass
class AlphaConfig:
    """Alpha construction using trailing mean of factor returns."""
    trailing_days: int = 252
    # Combine rule placeholder: simple dot of (mean factor returns) with today's residual exposures
    combine_mode: str = "dot"  # placeholder for future extensions


# -------------------------
# Portfolio (Final Orders)
# -------------------------
@dataclass
class PortfolioConfig:
    """
    Portfolio building from alpha cross-section.
    """
    mode: str = "long_only_topk"  # {'long_only_topk', 'long_short_neutral', ...}
    top_k: int = 50
    max_weight: float = 0.05
    neutral: bool = False           # industry/market neutrality toggle (placeholder)
    turnover_threshold: float = 0.01  # small-change ignore threshold
    px_type: str = "close"          # placeholder for order pricing note
    note: str = "auto"              # free-form note field


# -------------------------
# Rolling Storage & IO
# -------------------------
@dataclass
class StorageConfig:
    """Rolling time-series storage policy."""
    keep_last_days: int = 252
    atomic_writes: bool = True
    # CSV schema guard toggles (placeholder only)
    strict_schema: bool = False


# -------------------------
# Logging / Runtime
# -------------------------
@dataclass
class LoggingConfig:
    """Console and file logging knobs for utils/logging.py"""
    verbosity: str = Verbosity.STEP
    log_to_console: bool = True
    log_to_file: bool = True
    progress_every: int = 50  # emit [LOOP] log every N items (placeholder)


@dataclass
class RuntimeConfig:
    """Generic runtime knobs."""
    random_seed: int = 42
    n_jobs: int = max(os.cpu_count() or 2, 2)
    fail_fast: bool = False


# -------------------------
# Top-Level Config
# -------------------------
@dataclass
class Config:
    """Top-level bag that carries all sub-configs."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    styles: StyleConfig = field(default_factory=StyleConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    neutralize: NeutralizeConfig = field(default_factory=NeutralizeConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (useful for debug dumps)."""
        d = asdict(self)
        # Convert Paths to strings for nicer printing
        def stringify_paths(node: Any):
            if isinstance(node, dict):
                return {k: stringify_paths(v) for k, v in node.items()}
            if isinstance(node, list):
                return [stringify_paths(v) for v in node]
            if isinstance(node, Path):
                return str(node)
            return node
        return stringify_paths(d)


# -------------------------
# Environment Overrides (lightweight)
# -------------------------
def _apply_env_overrides(cfg: Config, prefix: str = "PIPE_") -> None:
    """
    Apply a tiny set of environment overrides.
    - Only supports a few frequently changed fields to keep it simple.
    - Types are parsed conservatively. Non-parsable values are ignored.
    """
    env = os.environ

    # Example: project root override (useful in notebooks/CI)
    root = env.get(f"{prefix}PROJECT_ROOT")
    if root:
        cfg.paths.project_root = Path(root).resolve()
        cfg.paths.__post_init__()  # recompute dependent paths

    # Verbosity override
    v = env.get(f"{prefix}VERBOSITY")
    if v in (Verbosity.SILENT, Verbosity.STEP, Verbosity.LOOP, Verbosity.DEBUG):
        cfg.logging.verbosity = v

    # Forward days & trailing window (commonly tweaked)
    fwd = env.get(f"{prefix}FORWARD_DAYS")
    if fwd and fwd.isdigit():
        cfg.universe.forward_days = int(fwd)

    trail = env.get(f"{prefix}ALPHA_TRAILING_DAYS")
    if trail and trail.isdigit():
        cfg.alpha.trailing_days = int(trail)

    # Portfolio knobs
    topk = env.get(f"{prefix}TOP_K")
    if topk and topk.isdigit():
        cfg.portfolio.top_k = int(topk)

    maxw = env.get(f"{prefix}MAX_WEIGHT")
    if maxw:
        try:
            cfg.portfolio.max_weight = float(maxw)
        except ValueError:
            pass

    # Ridge alpha
    ridge = env.get(f"{prefix}RIDGE_ALPHA")
    if ridge:
        try:
            val = float(ridge)
            cfg.neutralize.ridge_alpha = val
            cfg.regression.ridge_alpha = val
        except ValueError:
            pass

    # Keep-last rolling days
    keep = env.get(f"{prefix}KEEP_LAST_DAYS")
    if keep and keep.isdigit():
        cfg.storage.keep_last_days = int(keep)


# -------------------------
# Factory
# -------------------------
def load_config() -> Config:
    """
    Create a Config with defaults, then apply light env overrides.
    This function should be the single import point in the codebase:
        from config import load_config
        CFG = load_config()
    """
    cfg = Config()
    _apply_env_overrides(cfg)
    return cfg


# -------------------------
# Module globals (optional convenience)
# -------------------------
CFG: Config = load_config()

__all__ = [
    "Verbosity",
    "PathsConfig",
    "UniverseConfig",
    "FetchConfig",
    "CleaningConfig",
    "StyleConfig",
    "FactorConfig",
    "NeutralizeConfig",
    "RegressionConfig",
    "AlphaConfig",
    "PortfolioConfig",
    "StorageConfig",
    "LoggingConfig",
    "RuntimeConfig",
    "Config",
    "load_config",
    "CFG",
]


# -------------------------
# CLI Preview (debug only)
# -------------------------
if __name__ == "__main__":
    # Print a compact JSON view for quick inspection
    print(json.dumps(CFG.to_dict(), indent=2, ensure_ascii=False))
