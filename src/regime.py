"""
regime.py – Market regime detection for NSE/Indian equities.
Detects: bull, bear, sideways, high-volatility regimes.
Adjusts portfolio exposure and stock selection accordingly.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"


class RegimeDetector:
    """
    Detects market regime from NIFTY index data.

    Rules:
      BULL       : price > 200dma AND 200dma trending up AND vol < 75th pct
      BEAR       : price < 200dma by > 10% OR recent drawdown > 15%
      HIGH_VOL   : realized vol > 75th percentile historically
      SIDEWAYS   : otherwise
    """

    def __init__(self, cfg: dict) -> None:
        rc = cfg.get("regime", {})
        self.enabled: bool = rc.get("enabled", True)
        self.lookback: int = rc.get("lookback", 200)
        self.vol_pct_high: float = rc.get("vol_percentile_high", 0.75)
        self.bear_threshold: float = rc.get("bear_market_threshold", -0.20)
        self.bear_max_exposure: float = rc.get("bear_max_exposure", 0.70)
        self.bear_top_n: int = rc.get("bear_top_n", 10)
        self._regime_history: Optional[pd.Series] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, benchmark_df: pd.DataFrame) -> pd.Series:
        """
        Compute per-date regime labels for the benchmark.

        Returns
        -------
        pd.Series[str] indexed by Date
        """
        if not self.enabled or benchmark_df is None:
            n = len(benchmark_df) if benchmark_df is not None else 1
            idx = benchmark_df.index if benchmark_df is not None else pd.DatetimeIndex([])
            return pd.Series(Regime.BULL.value, index=idx)

        close = benchmark_df["Close"]
        regime = pd.Series(Regime.SIDEWAYS.value, index=close.index)

        # 200-day MA
        ma200 = close.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        ma200_slope = ma200.pct_change(20)  # 20-day slope of 200dma

        # Realized vol (21d annualized)
        ret = close.pct_change()
        vol_21d = ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        vol_hist_pct = vol_21d.rank(pct=True)

        # Drawdown from rolling peak
        roll_max = close.rolling(252, min_periods=63).max()
        drawdown = (close - roll_max) / (roll_max + 1e-10)

        # Classify each date
        for date in close.index:
            c = close.loc[date]
            ma = ma200.loc[date] if not np.isnan(ma200.loc[date]) else c
            slope = ma200_slope.loc[date] if not np.isnan(ma200_slope.loc[date]) else 0
            vol_pct = vol_hist_pct.loc[date] if not np.isnan(vol_hist_pct.loc[date]) else 0.5
            dd = drawdown.loc[date] if not np.isnan(drawdown.loc[date]) else 0

            if vol_pct >= self.vol_pct_high:
                regime.loc[date] = Regime.HIGH_VOL.value
            elif dd <= self.bear_threshold or c < ma * 0.90:
                regime.loc[date] = Regime.BEAR.value
            elif c > ma and slope > 0:
                regime.loc[date] = Regime.BULL.value
            else:
                regime.loc[date] = Regime.SIDEWAYS.value

        self._regime_history = regime

        # Log distribution
        dist = regime.value_counts()
        log.info(f"Regime distribution:\n{dist.to_string()}")
        return regime

    def current_regime(self, benchmark_df: pd.DataFrame) -> Regime:
        """Return the most recent regime."""
        regimes = self.detect(benchmark_df)
        latest = regimes.iloc[-1]
        log.info(f"Current market regime: {latest}")
        return Regime(latest)

    def get_exposure_multiplier(self, regime: Regime) -> float:
        """
        Returns portfolio exposure multiplier for a given regime.
        Bull/Sideways = 1.0, High-vol = 0.85, Bear = bear_max_exposure.
        """
        if regime == Regime.BEAR:
            return self.bear_max_exposure
        elif regime == Regime.HIGH_VOL:
            return 0.85
        return 1.0

    def get_top_n_override(self, regime: Regime, default_top_n: int) -> int:
        """Return adjusted top-N for bear markets."""
        if regime == Regime.BEAR:
            return min(self.bear_top_n, default_top_n)
        return default_top_n

    def regime_at_date(
        self, benchmark_df: pd.DataFrame, date: pd.Timestamp
    ) -> Regime:
        """Regime on a specific date (or nearest prior)."""
        regimes = self._regime_history
        if regimes is None:
            regimes = self.detect(benchmark_df)
        dates = regimes.index
        prior = dates[dates <= date]
        if len(prior) == 0:
            return Regime.SIDEWAYS
        return Regime(regimes.loc[prior[-1]])
