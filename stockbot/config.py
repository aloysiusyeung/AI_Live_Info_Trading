"""Configuration loaded from environment variables.

The application refuses to start unless ALPACA_PAPER is true. Credentials are
held in memory only and are never logged, serialised or written to the
database. ``Settings.__repr__`` deliberately masks them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:  # python-dotenv is optional at runtime; env vars may already be exported.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without dotenv
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Values accepted as boolean true. Anything else is false.
_TRUE_VALUES = {"1", "true", "yes", "on"}

#: Regular US session is 6.5 hours -> 39 bars of 10 minutes.
BARS_PER_REGULAR_SESSION = 39


class ConfigError(RuntimeError):
    """Raised when the environment is not safe or complete enough to start."""


class LiveTradingRefused(ConfigError):
    """Raised on any attempt to leave paper mode."""


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name) or default
    items = [part.strip().upper() for part in raw.split(",")]
    return [item for item in items if item]


@dataclass
class Settings:
    """Immutable-ish snapshot of the runtime configuration."""

    # --- credentials (never logged) ---
    alpaca_api_key: str = field(repr=False, default="")
    alpaca_secret_key: str = field(repr=False, default="")

    # --- safety ---
    paper: bool = True
    enable_paper_orders: bool = False
    kill_switch: bool = False

    # --- universe / data ---
    watchlist: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY"])
    benchmark_symbol: str = "SPY"
    data_feed: str = "iex"
    bar_minutes: int = 10
    history_days: int = 180

    # --- freshness / bar-close gating ---
    bar_settle_seconds: int = 45
    max_data_age_seconds: int = 1800

    # --- horizon / labelling ---
    prediction_horizon_bars: int = BARS_PER_REGULAR_SESSION
    estimated_spread_bps: float = 3.0
    estimated_slippage_bps: float = 2.0
    label_threshold_bps: float = 0.0

    # --- model / validation ---
    walkforward_splits: int = 5
    min_training_rows: int = 750
    min_confidence: float = 0.58
    uncertainty_max: float = 0.18
    retrain_interval_hours: int = 24

    # --- risk limits ---
    max_position_pct: float = 0.10
    max_portfolio_exposure_pct: float = 0.60
    max_daily_loss_pct: float = 0.02
    risk_per_trade_pct: float = 0.005
    max_spread_bps: float = 25.0
    min_avg_dollar_volume: float = 5_000_000.0
    min_order_notional: float = 50.0
    duplicate_order_window_minutes: int = 60

    # --- news (market-wide, not limited to the watchlist) ---
    news_enabled: bool = True
    news_backfill_days: int = 7
    news_page_limit: int = 50
    news_max_pages: int = 40
    news_include_content: bool = False
    news_exclude_contentless: bool = False

    # --- dynamic universe driven by news ---
    dynamic_universe_enabled: bool = True
    max_dynamic_symbols: int = 40
    min_news_for_candidate: int = 2
    candidate_lookback_hours: int = 48
    universe_refresh_hours: int = 12
    # Training is expensive (three candidates x walk-forward, per symbol). Cap
    # how many models one cycle may build so a freshly widened universe fills
    # in over several cycles instead of overrunning the 10-minute budget.
    max_trainings_per_cycle: int = 5

    # --- infrastructure ---
    database_path: str = str(PROJECT_ROOT / "data" / "stockbot.sqlite")
    model_dir: str = str(PROJECT_ROOT / "models_store")
    log_dir: str = str(PROJECT_ROOT / "logs")
    log_level: str = "INFO"
    timezone: str = "America/New_York"
    scheduler_interval_minutes: int = 10

    def __post_init__(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    # -- derived ---------------------------------------------------------
    @property
    def round_trip_cost_bps(self) -> float:
        """Spread + slippage applied once on entry and once on exit."""
        return 2.0 * (self.estimated_spread_bps + self.estimated_slippage_bps)

    @property
    def round_trip_cost(self) -> float:
        return self.round_trip_cost_bps / 10_000.0

    @property
    def all_symbols(self) -> List[str]:
        """The always-analysed core: the watchlist plus the benchmark.

        This is the *floor* of the analysis universe, not its ceiling. News
        ingestion is market-wide and ignores this entirely; the dynamic universe
        extends it with news-active, tradable symbols.
        """
        symbols = list(self.watchlist)
        if self.benchmark_symbol not in symbols:
            symbols.append(self.benchmark_symbol)
        return symbols

    @property
    def horizon_label(self) -> str:
        sessions = self.prediction_horizon_bars / BARS_PER_REGULAR_SESSION
        if abs(sessions - 1.0) < 1e-9:
            return "next trading day"
        if sessions >= 1:
            return f"{sessions:.1f} trading sessions"
        minutes = self.prediction_horizon_bars * self.bar_minutes
        return f"{minutes} minutes"

    def credentials(self) -> tuple[str, str]:
        return self.alpaca_api_key, self.alpaca_secret_key

    def masked(self) -> dict:
        """Config safe for logging / display: no secrets."""
        return {
            "paper": self.paper,
            "enable_paper_orders": self.enable_paper_orders,
            "kill_switch": self.kill_switch,
            "watchlist": self.watchlist,
            "benchmark_symbol": self.benchmark_symbol,
            "data_feed": self.data_feed,
            "bar_minutes": self.bar_minutes,
            "prediction_horizon_bars": self.prediction_horizon_bars,
            "horizon": self.horizon_label,
            "news_enabled": self.news_enabled,
            "news_coverage": "market-wide (all US symbols)" if self.news_enabled else "off",
            "dynamic_universe_enabled": self.dynamic_universe_enabled,
            "max_dynamic_symbols": self.max_dynamic_symbols,
            "api_key": _mask(self.alpaca_api_key),
            "secret_key": "<set>" if self.alpaca_secret_key else "<missing>",
            "database_path": self.database_path,
        }


def _mask(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def load_settings(require_credentials: bool = True, dotenv_path: str | None = None) -> Settings:
    """Build :class:`Settings` from the environment.

    Raises :class:`LiveTradingRefused` when ALPACA_PAPER is not true. This is the
    single entry point used by every executable in the project.
    """
    load_dotenv(dotenv_path or str(PROJECT_ROOT / ".env"), override=False)

    paper = env_bool("ALPACA_PAPER", default=False)
    if not paper:
        raise LiveTradingRefused(
            "Refusing to start: ALPACA_PAPER must be 'true'. "
            "This application supports paper trading only and will not connect "
            "to a live trading endpoint."
        )

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if require_credentials and not (api_key and secret_key):
        raise ConfigError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment."
        )

    settings = Settings(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        paper=True,
        enable_paper_orders=env_bool("ENABLE_PAPER_ORDERS", False),
        kill_switch=env_bool("KILL_SWITCH", False),
        watchlist=env_list("WATCHLIST", "AAPL,MSFT,NVDA,SPY"),
        benchmark_symbol=os.getenv("BENCHMARK_SYMBOL", "SPY").strip().upper(),
        data_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip().lower(),
        bar_minutes=env_int("BAR_MINUTES", 10),
        history_days=env_int("HISTORY_DAYS", 180),
        bar_settle_seconds=env_int("BAR_SETTLE_SECONDS", 45),
        max_data_age_seconds=env_int("MAX_DATA_AGE_SECONDS", 1800),
        prediction_horizon_bars=env_int("PREDICTION_HORIZON_BARS", BARS_PER_REGULAR_SESSION),
        estimated_spread_bps=env_float("ESTIMATED_SPREAD_BPS", 3.0),
        estimated_slippage_bps=env_float("ESTIMATED_SLIPPAGE_BPS", 2.0),
        label_threshold_bps=env_float("LABEL_THRESHOLD_BPS", 0.0),
        walkforward_splits=env_int("WALKFORWARD_SPLITS", 5),
        min_training_rows=env_int("MIN_TRAINING_ROWS", 750),
        min_confidence=env_float("MIN_CONFIDENCE", 0.58),
        uncertainty_max=env_float("UNCERTAINTY_MAX", 0.18),
        retrain_interval_hours=env_int("RETRAIN_INTERVAL_HOURS", 24),
        max_position_pct=env_float("MAX_POSITION_PCT", 0.10),
        max_portfolio_exposure_pct=env_float("MAX_PORTFOLIO_EXPOSURE_PCT", 0.60),
        max_daily_loss_pct=env_float("MAX_DAILY_LOSS_PCT", 0.02),
        risk_per_trade_pct=env_float("RISK_PER_TRADE_PCT", 0.005),
        max_spread_bps=env_float("MAX_SPREAD_BPS", 25.0),
        min_avg_dollar_volume=env_float("MIN_AVG_DOLLAR_VOLUME", 5_000_000.0),
        min_order_notional=env_float("MIN_ORDER_NOTIONAL", 50.0),
        duplicate_order_window_minutes=env_int("DUPLICATE_ORDER_WINDOW_MINUTES", 60),
        news_enabled=env_bool("NEWS_ENABLED", True),
        news_backfill_days=env_int("NEWS_BACKFILL_DAYS", 7),
        news_page_limit=env_int("NEWS_PAGE_LIMIT", 50),
        news_max_pages=env_int("NEWS_MAX_PAGES", 40),
        news_include_content=env_bool("NEWS_INCLUDE_CONTENT", False),
        news_exclude_contentless=env_bool("NEWS_EXCLUDE_CONTENTLESS", False),
        dynamic_universe_enabled=env_bool("DYNAMIC_UNIVERSE_ENABLED", True),
        max_dynamic_symbols=env_int("MAX_DYNAMIC_SYMBOLS", 40),
        min_news_for_candidate=env_int("MIN_NEWS_FOR_CANDIDATE", 2),
        candidate_lookback_hours=env_int("CANDIDATE_LOOKBACK_HOURS", 48),
        universe_refresh_hours=env_int("UNIVERSE_REFRESH_HOURS", 12),
        max_trainings_per_cycle=env_int("MAX_TRAININGS_PER_CYCLE", 5),
        database_path=os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "stockbot.sqlite")),
        model_dir=os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models_store")),
        log_dir=os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        timezone=os.getenv("TIMEZONE", "America/New_York"),
        scheduler_interval_minutes=env_int("SCHEDULER_INTERVAL_MINUTES", 10),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if settings.bar_minutes <= 0:
        raise ConfigError("BAR_MINUTES must be positive")
    if settings.prediction_horizon_bars <= 0:
        raise ConfigError("PREDICTION_HORIZON_BARS must be positive")
    if not 0 < settings.max_position_pct <= 1:
        raise ConfigError("MAX_POSITION_PCT must be in (0, 1]")
    if not 0 < settings.max_portfolio_exposure_pct <= 1:
        raise ConfigError("MAX_PORTFOLIO_EXPOSURE_PCT must be in (0, 1]")
    if not 0.5 <= settings.min_confidence < 1:
        raise ConfigError("MIN_CONFIDENCE must be in [0.5, 1)")
    if settings.data_feed not in {"iex", "sip", "otc", "delayed_sip"}:
        raise ConfigError(f"Unsupported ALPACA_DATA_FEED: {settings.data_feed}")
    if not settings.watchlist:
        raise ConfigError("WATCHLIST is empty")
    if settings.news_page_limit < 1 or settings.news_page_limit > 50:
        raise ConfigError("NEWS_PAGE_LIMIT must be between 1 and 50 (Alpaca's cap)")
    if settings.max_dynamic_symbols < 0:
        raise ConfigError("MAX_DYNAMIC_SYMBOLS must not be negative")
