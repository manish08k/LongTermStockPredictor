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
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a coloured console logger."""
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
            logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Timing decorator ───────────────────────────────────────────────────────

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger(func.__module__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        log.info(f"{func.__name__} completed in {time.perf_counter() - t0:.2f}s")
        return result
    return wrapper


# ─── Numeric helpers ────────────────────────────────────────────────────────

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def zscore(series: pd.Series, window: int = None) -> pd.Series:
    if window:
        mu = series.rolling(window, min_periods=window // 2).mean()
        sigma = series.rolling(window, min_periods=window // 2).std()
    else:
        mu, sigma = series.mean(), series.std()
    return (series - mu) / (sigma + 1e-10)


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (periods / n) - 1) if n > 0 else 0.0


def sharpe_ratio(returns: pd.Series, rf: float = 0.065, periods: int = 252) -> float:
    excess = returns - rf / periods
    std = returns.std()
    return float(excess.mean() / std * np.sqrt(periods)) if std > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    cagr = annualized_return(returns, periods)
    equity = (1 + returns).cumprod()
    mdd = abs(max_drawdown(equity))
    return cagr / mdd if mdd > 0 else 0.0


# ─── I/O helpers ─────────────────────────────────────────────────────────────

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
