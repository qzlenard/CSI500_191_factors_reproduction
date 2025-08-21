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
    index_code: str = "SHSE.000905"          # CSI 500 (GM normalized)
    exchange_calendar: str = "SSE"           # tag for Shanghai exchange calendar
    timezone: str = "Asia/Shanghai"
    lookback_trading_days: int = 252         # one trading year
    forward_days: int = 2                    # r_{t -> t+2}

    # Adding EXCLUDE_B_PREFIX here
    EXCLUDE_B_PREFIX: set = field(default_factory=lambda: {"900", "200"})  # B股前缀


# -------------------------
# Data Fetch & Cleaning
# -------------------------
@dataclass
class FetchConfig:
    """
    Data acquisition knobs for src/api/myquant_io.py
    fq denotes price adjustment. Use {'pre','post','none'}.
    """
    price_adjust: str = "pre"   # unified with gm.api adapter mapping
    calendar_buffer_days: int = 10
    batch_size: int = 100
    max_retries: int = 3
    retry_backoff_sec: float = 1.0

    # Limit-up / down filtering thresholds when high/low limits missing
    limit_up_nonst: float = 0.098   # ~ 9.8%
    limit_dn_nonst: float = -0.098
    limit_up_st: float = 0.048      # ~ 4.8%
    limit_dn_st: float = -0.048

    # Name of the environment variable that carries GM token
    token_env_key: str = "GM_TOKEN"


@dataclass
class CleaningConfig:
    """Winsorization / standardization placeholders."""
    winsor_pct: float = 0.01  # 1% two-sided
    zscore_ddof: int = 0      # standard zscore, ddof=0
    standardize: bool = True  # apply z-score after winsorization


# -------------------------
# Style (Fundamentals) Factors
# -------------------------
@dataclass
class StyleConfig:
    """Financial statement styles snapshot settings (announcement-aware lag)."""
    lag_trading_days: int = 30
    fallback_with_price_proxy: bool = True
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
    names: List[str] = field(default_factory=lambda: [f"f{i}" for i in range(1, 192)])
    min_window_by_factor: Dict[str, int] = field(default_factory=dict)


# -------------------------
# Neutralization (Step 2)
# -------------------------
@dataclass
class NeutralizeConfig:
    """Regression settings for orthogonalization on styles + industry dummies."""
    use_ridge: bool = True
    ridge_alpha: float = 1e-6
    safety_margin: int = 5
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
    combine_mode: str = "dot"


# -------------------------
# Portfolio (Final Orders)
# -------------------------
@dataclass
class PortfolioConfig:
    """Portfolio building from alpha cross-section."""
    mode: str = "long_only_topk"  # {'long_only_topk','long_short_neutral',...}
    top_k: int = 50
    max_weight: float = 0.05
    neutral: bool = False
    turnover_threshold: float = 0.01
    px_type: str = "close"
    note: str = "auto"


# -------------------------
# Rolling Storage & IO
# -------------------------
@dataclass
class StorageConfig:
    """Rolling time-series storage policy."""
    keep_last_days: int = 252
    atomic_writes: bool = True
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

        def stringify_paths(node: Any):
            if isinstance(node, dict):
                return {k: stringify_paths(v) for k, v in node.items()}
            if isinstance(node, list):
                return [stringify_paths(v) for v in node]
            if isinstance(node, Path):
                return str(node)
            if isinstance(node, set):  # Add this line to convert set to list
                return list(node)
            return node

        return stringify_paths(d)


# -------------------------
# Environment Overrides (lightweight)
# -------------------------
def _apply_env_overrides(cfg: Config, prefix: str = "PIPE_") -> None:
    """
    Apply a tiny set of environment overrides.
    Types are parsed conservatively. Non-parsable values are ignored.
    """
    env = os.environ

    # Project root override (e.g., in CI)
    root = env.get(f"{prefix}PROJECT_ROOT")
    if root:
        cfg.paths.project_root = Path(root).resolve()
        cfg.paths.__post_init__()  # recompute dependent paths

    # Verbosity
    v = env.get(f"{prefix}VERBOSITY")
    if v in (Verbosity.SILENT, Verbosity.STEP, Verbosity.LOOP, Verbosity.DEBUG):
        cfg.logging.verbosity = v

    # Forward days & trailing window
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
# Module globals (convenience)
# -------------------------
CFG: Config = load_config()

# --- GM token exposure for adapters expecting `config.GM_TOKEN` ---
# User-provided token (overrides env). Clear this string to fall back to env.
GM_TOKEN: Optional[str] = "1d43f62303ce89bd50caa3a1bb339f463c875bd3"
if not GM_TOKEN:
    GM_TOKEN = os.environ.get(CFG.fetch.token_env_key, None)

# --- Simple alias constants (backward-compat with earlier contracts) ---
# Universe / windows
INDEX_CODE: str = CFG.universe.index_code
WINDOW_TRADING_DAYS: int = CFG.universe.lookback_trading_days
FORWARD_DAYS: int = CFG.universe.forward_days

# Cleaning / regression
WINSOR_PCT: float = CFG.cleaning.winsor_pct
STANDARDIZE: bool = CFG.cleaning.standardize
RIDGE_ALPHA: float = CFG.regression.ridge_alpha
STYLE_LAG_DAYS: int = CFG.styles.lag_trading_days
LOOKBACK_FOR_ALPHA: int = CFG.alpha.trailing_days

# Portfolio
PORTFOLIO_MODE: str = CFG.portfolio.mode
PORTFOLIO_TOP_K: int = CFG.portfolio.top_k
PORTFOLIO_MAX_WEIGHT: float = CFG.portfolio.max_weight
TURNOVER_THRESHOLD: float = CFG.portfolio.turnover_threshold

# Logging / fetch
LOG_VERBOSITY: str = CFG.logging.verbosity
FQ: str = CFG.fetch.price_adjust  # {'pre','post','none'}
RANDOM_SEED: int = CFG.runtime.random_seed

# Paths (both Path and str forms)
PROJ_ROOT: Path = CFG.paths.project_root
DATA_DIR: Path = CFG.paths.data_dir
RAW_OHLCV_DIR: Path = CFG.paths.raw_ohlcv_dir
REF_DIR: Path = CFG.paths.ref_dir

OUT_DIR: Path = CFG.paths.out_dir
OUT_RESIDUALS_DIR: Path = CFG.paths.out_residuals_dir
OUT_ALPHA_DIR: Path = CFG.paths.out_alpha_dir
OUT_ORDERS_DIR: Path = CFG.paths.out_orders_dir
OUT_TS_DIR: Path = CFG.paths.out_ts_dir
OUT_LOG_DIR: Path = CFG.paths.out_logs_dir

STATE_DIR: Path = CFG.paths.state_dir
LOCKS_DIR: Path = CFG.paths.locks_dir
MANIFEST_PATH: Path = CFG.paths.manifest_path

# String aliases (for modules that prefer str)
PROJ_ROOT_STR = str(PROJ_ROOT)
DATA_DIR_STR = str(DATA_DIR)
RAW_OHLCV_DIR_STR = str(RAW_OHLCV_DIR)
REF_DIR_STR = str(REF_DIR)
OUT_DIR_STR = str(OUT_DIR)
OUT_RESIDUALS_DIR_STR = str(OUT_RESIDUALS_DIR)
OUT_ALPHA_DIR_STR = str(OUT_ALPHA_DIR)
OUT_ORDERS_DIR_STR = str(OUT_ORDERS_DIR)
OUT_TS_DIR_STR = str(OUT_TS_DIR)
OUT_LOG_DIR_STR = str(OUT_LOG_DIR)
STATE_DIR_STR = str(STATE_DIR)
LOCKS_DIR_STR = str(LOCKS_DIR)
MANIFEST_PATH_STR = str(MANIFEST_PATH)

# Time-series CSV canonical locations/patterns
FACTOR_RETURNS_CSV: str = str(OUT_TS_DIR / "factor_returns.csv")
COVERAGE_CSV: str = str(OUT_TS_DIR / "coverage.csv")
METRICS_CSV: str = str(OUT_TS_DIR / "metrics.csv")
RESIDUALS_CSV_PATTERN: str = str(OUT_RESIDUALS_DIR / "{yyyymmdd}_residuals.csv")
ALPHA_CSV_PATTERN: str = str(OUT_ALPHA_DIR / "{yyyymmdd}_alpha.csv")
ORDERS_CSV_PATTERN: str = str(OUT_ORDERS_DIR / "{yyyymmdd}_orders.csv")

# Rolling window for time-series append+rolloff (alias for keep_last_days)
ROLLING_KEEP_DAYS: int = CFG.storage.keep_last_days


__all__ = [
    # Classes
    "Verbosity", "PathsConfig", "UniverseConfig", "FetchConfig", "CleaningConfig",
    "StyleConfig", "FactorConfig", "NeutralizeConfig", "RegressionConfig",
    "AlphaConfig", "PortfolioConfig", "StorageConfig", "LoggingConfig",
    "RuntimeConfig", "Config",
    # Factory / object
    "load_config", "CFG",
    # Aliases / constants
    "GM_TOKEN", "INDEX_CODE", "WINDOW_TRADING_DAYS", "FORWARD_DAYS",
    "WINSOR_PCT", "STANDARDIZE", "RIDGE_ALPHA", "STYLE_LAG_DAYS",
    "LOOKBACK_FOR_ALPHA", "PORTFOLIO_MODE", "PORTFOLIO_TOP_K",
    "PORTFOLIO_MAX_WEIGHT", "TURNOVER_THRESHOLD", "LOG_VERBOSITY", "FQ",
    "RANDOM_SEED", "PROJ_ROOT", "DATA_DIR", "RAW_OHLCV_DIR", "REF_DIR",
    "OUT_DIR", "OUT_RESIDUALS_DIR", "OUT_ALPHA_DIR", "OUT_ORDERS_DIR",
    "OUT_TS_DIR", "OUT_LOG_DIR", "STATE_DIR", "LOCKS_DIR", "MANIFEST_PATH",
    "PROJ_ROOT_STR", "DATA_DIR_STR", "RAW_OHLCV_DIR_STR", "REF_DIR_STR",
    "OUT_DIR_STR", "OUT_RESIDUALS_DIR_STR", "OUT_ALPHA_DIR_STR",
    "OUT_ORDERS_DIR_STR", "OUT_TS_DIR_STR", "OUT_LOG_DIR_STR",
    "STATE_DIR_STR", "LOCKS_DIR_STR", "MANIFEST_PATH_STR",
    "FACTOR_RETURNS_CSV", "COVERAGE_CSV", "METRICS_CSV",
    "RESIDUALS_CSV_PATTERN", "ALPHA_CSV_PATTERN", "ORDERS_CSV_PATTERN",
    "ROLLING_KEEP_DAYS",
]

# file: config.py
# --- APPEND-ONLY: Industry config keys for industry.py (do NOT modify existing content above) ---

# =========================
# A. Industry standard & paths
# =========================
# Industry taxonomy standard used by industry.py filtering.
# Allowed values: "SW"(default), "GICS", "CITIC", "CSRC"
IND_STANDARD: str = "SW"

# Industry hierarchy level (we use level-1 in regressions). Allowed: 1/2/3
IND_LEVEL: int = 1

# Root directory for industry mapping/snapshots (path join only, no I/O here)
IND_DIR: str = "data/ref/industry_map"

# Constituents CSV glob pattern (we'll select the most recent or <= date in industry.py)
IND_CONS_GLOB: str = "data/ref/industry_map/QT_INDUS_CONSTITUENTS_202508191251.csv"

# One-hot snapshot output path pattern; {date} should be YYYYMMDD
IND_SNAPSHOT_PATTERN: str = "data/ref/industry_map/{date}_industry.csv"

# JSON path to persist the union of all historical industry columns
# (to keep column space monotonically non-decreasing)
IND_ALLCOLUMNS_JSON: str = "data/ref/industry_map/latest_columns.json"


# =========================
# B. Source CSV column mapping (aligned with your file headers)
# File name example: QT_INDUS_CONSTITUENTS_202508191251.csv
# If 'effective_date' is absent, industry.py will fallback by priority:
# effective_date -> updatetime -> tmstamp (no look-ahead).
# =========================
IND_COLMAP = {
    "code": "STOCK_CODE",
    "industry_code": "INDUSTRY_CODE",
    "industry_name": "INDUSTRY_NAME",
    "industry_level": "INDUSTRY_LEVEL",
    "standard_code": "STANDARD_CODE",
    "standard_name": "STANDARD_NAME",
    "use_status": "USE_STATUS",
    "effective_date": "INTO_DATE",   # 有 OUT_DATE 可忽略
    "updatetime": "UPDATETIME",
    "tmstamp": "TMSTAMP",
    "entrytime": "ENTRYTIME",
}


# If 'use_status' column exists, keep rows with this ACTIVE flag
IND_USE_STATUS_ACTIVE: int = 1


# =========================
# C. Misc (non-intrusive defaults)
# =========================
# Unified date format for snapshot naming (does not affect other modules)
DATE_FMT: str = "%Y%m%d"

# Reserved for potential online source TTL (not enforced now)
IND_CACHE_TTL_DAYS: int = 3650


# =========================
# (Optional) Forecast EPS placeholders for future styles alignment
# (No I/O by default; safe to leave as is)
# =========================
FCAST_EPS_CSV = None
FCAST_EPS_COLMAP = {
    "code": "code",
    "forecast_date": "forecast_date",
    "agency": "agency",
    "forecast_eps": "forecast_eps",
    "horizon": "horizon",
}
FCAST_EPS_POLICY: str = "csv_then_estimate"


# =========================
# Export: extend __all__ without touching existing exports
# =========================
try:
    __all__.extend([
        "IND_STANDARD", "IND_LEVEL", "IND_DIR", "IND_CONS_GLOB",
        "IND_SNAPSHOT_PATTERN", "IND_ALLCOLUMNS_JSON", "IND_COLMAP",
        "IND_USE_STATUS_ACTIVE", "DATE_FMT", "IND_CACHE_TTL_DAYS",
        "FCAST_EPS_CSV", "FCAST_EPS_COLMAP", "FCAST_EPS_POLICY",
    ])
except NameError:
    # In case __all__ was not defined earlier in this module
    __all__ = [
        "IND_STANDARD", "IND_LEVEL", "IND_DIR", "IND_CONS_GLOB",
        "IND_SNAPSHOT_PATTERN", "IND_ALLCOLUMNS_JSON", "IND_COLMAP",
        "IND_USE_STATUS_ACTIVE", "DATE_FMT", "IND_CACHE_TTL_DAYS",
        "FCAST_EPS_CSV", "FCAST_EPS_COLMAP", "FCAST_EPS_POLICY",
    ]


# -------------------------
# CLI Preview (debug only)
# -------------------------
if __name__ == "__main__":
    print(json.dumps(CFG.to_dict(), indent=2, ensure_ascii=False))
