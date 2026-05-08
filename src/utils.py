"""
utils.py – Shared utilities: logging, config loading, timing, I/O helpers.
"""
from __future__ import annotations

import functools
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    if HAS_COLORLOG:
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan", "INFO": "green",
                    "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red",
                },
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Ensure required keys exist with safe defaults – no KeyErrors ever."""
    cfg.setdefault("project", {})
    cfg["project"].setdefault("random_seed", 42)
    cfg["project"].setdefault("log_level", "INFO")

    cfg.setdefault("data", {})
    cfg["data"].setdefault("tickers", [])
    cfg["data"].setdefault("benchmark_ticker", "^NSEI")
    cfg["data"].setdefault("start_date", "2015-01-01")
    cfg["data"].setdefault("end_date", "2024-12-31")
    cfg["data"].setdefault("interval", "1d")
    cfg["data"].setdefault("cache_data", True)
    cfg["data"].setdefault("raw_data_path", "data/raw")
    cfg["data"].setdefault("features_data_path", "data/features")
    cfg["data"].setdefault("min_history_days", 252)
    cfg["data"].setdefault("sectors", {})

    cfg.setdefault("features", {})
    cfg["features"].setdefault("momentum_windows", [21, 63, 126, 252])
    cfg["features"].setdefault("ma_windows", [20, 50, 100, 200])
    cfg["features"].setdefault("vol_windows", [10, 21, 63])
    cfg["features"].setdefault("rsi_window", 14)
    cfg["features"].setdefault("macd_fast", 12)
    cfg["features"].setdefault("macd_slow", 26)
    cfg["features"].setdefault("macd_signal", 9)
    cfg["features"].setdefault("bb_window", 20)
    cfg["features"].setdefault("bb_std", 2.0)
    cfg["features"].setdefault("atr_window", 14)
    cfg["features"].setdefault("lag_periods", [1, 5, 10, 21])
    cfg["features"].setdefault("zscore_window", 252)
    cfg["features"].setdefault("feature_selection", {"enabled": False})

    cfg.setdefault("labeling", {})
    cfg["labeling"].setdefault("horizons", [21, 63, 126, 252])
    cfg["labeling"].setdefault("primary_horizon", 63)
    cfg["labeling"].setdefault("label_type", "binary")
    cfg["labeling"].setdefault("outperform_threshold", 0.0)
    cfg["labeling"].setdefault("triple_barrier", {"enabled": False})

    cfg.setdefault("models", {})
    cfg["models"].setdefault("use_models", ["xgboost", "lightgbm"])
    cfg["models"].setdefault("cv_splits", 5)
    cfg["models"].setdefault("cv_gap", 21)
    cfg["models"].setdefault("early_stopping_rounds", 50)
    cfg["models"].setdefault("feature_importance_top_n", 30)
    cfg["models"].setdefault("saved_models", "models/saved_models")
    cfg["models"].setdefault("walk_forward", {
        "train_window": 504, "test_window": 63, "gap": 21, "expanding_window": True
    })
    cfg["models"].setdefault("xgboost", {})
    cfg["models"].setdefault("lightgbm", {})
    cfg["models"].setdefault("random_forest", {})

    cfg.setdefault("ensemble", {})
    cfg["ensemble"].setdefault("method", "weighted_average")
    cfg["ensemble"].setdefault("weights", {})
    cfg["ensemble"].setdefault("ml_weight", 0.70)
    cfg["ensemble"].setdefault("alpha_weight", 0.30)

    cfg.setdefault("ranking", {})
    cfg["ranking"].setdefault("top_n", 15)
    cfg["ranking"].setdefault("score_method", "probability")
    cfg["ranking"].setdefault("min_score_threshold", 0.50)
    cfg["ranking"].setdefault("sector_diversification", False)
    cfg["ranking"].setdefault("max_per_sector", 5)

    cfg.setdefault("portfolio", {})
    cfg["portfolio"].setdefault("method", "max_sharpe")
    cfg["portfolio"].setdefault("max_weight", 0.20)
    cfg["portfolio"].setdefault("min_weight", 0.02)
    cfg["portfolio"].setdefault("risk_free_rate", 0.065)
    cfg["portfolio"].setdefault("regularization", 0.0001)
    cfg["portfolio"].setdefault("sector_max_exposure", 0.40)
    cfg["portfolio"].setdefault("turnover_limit", 1.0)

    cfg.setdefault("risk", {})
    cfg["risk"].setdefault("max_portfolio_volatility", 0.25)
    cfg["risk"].setdefault("stop_loss_pct", -0.15)
    cfg["risk"].setdefault("max_drawdown_exit", -0.25)
    cfg["risk"].setdefault("volatility_filter_window", 21)
    cfg["risk"].setdefault("volatility_filter_threshold", 0.60)
    cfg["risk"].setdefault("min_liquidity_adv", 0)

    cfg.setdefault("regime", {})
    cfg["regime"].setdefault("enabled", False)
    cfg["regime"].setdefault("lookback", 200)
    cfg["regime"].setdefault("bear_max_exposure", 0.70)
    cfg["regime"].setdefault("bear_top_n", 10)

    cfg.setdefault("backtest", {})
    cfg["backtest"].setdefault("initial_capital", 10_000_000)
    cfg["backtest"].setdefault("rebalance_frequency", "monthly")
    cfg["backtest"].setdefault("transaction_cost_bps", 20)
    cfg["backtest"].setdefault("slippage_bps", 10)
    cfg["backtest"].setdefault("min_trading_days", 126)
    cfg["backtest"].setdefault("walk_forward", {
        "train_window": 504, "test_window": 63, "gap": 21
    })

    cfg.setdefault("evaluation", {})
    cfg["evaluation"].setdefault("annualization_factor", 252)
    cfg["evaluation"].setdefault("output_dir", "results")
    cfg["evaluation"].setdefault("generate_html_report", True)

    cfg.setdefault("explainability", {})
    cfg["explainability"].setdefault("enabled", False)
    cfg["explainability"].setdefault("shap_background_samples", 100)
    cfg["explainability"].setdefault("top_n_features_per_stock", 10)


# ─── Timing decorator ─────────────────────────────────────────────────────────

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger(func.__module__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        log.info(f"{func.__name__} completed in {time.perf_counter() - t0:.2f}s")
        return result
    return wrapper


# ─── Numeric helpers ──────────────────────────────────────────────────────────

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    if window:
        mu = series.rolling(window, min_periods=window // 2).mean()
        sigma = series.rolling(window, min_periods=window // 2).std()
    else:
        mu, sigma = series.mean(), series.std()
    return (series - mu) / (sigma + 1e-10)


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (periods / n) - 1)


def sharpe_ratio(returns: pd.Series, rf: float = 0.065, periods: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    excess = returns - rf / periods
    std = returns.std()
    return float(excess.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / (roll_max + 1e-10)
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    cagr = annualized_return(returns, periods)
    equity = (1 + returns).cumprod()
    mdd = abs(max_drawdown(equity))
    return cagr / mdd if mdd > 0 else 0.0


def sortino_ratio(returns: pd.Series, rf: float = 0.065, periods: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    excess = returns - rf / periods
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods))


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path) -> None:
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=True)


def load_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ─── Data helpers ─────────────────────────────────────────────────────────────

def safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill)


def pct_rank_cross_section(series: pd.Series) -> pd.Series:
    """Percentile rank within each Date level of a MultiIndex Series."""
    return series.groupby(level="Date").rank(pct=True)