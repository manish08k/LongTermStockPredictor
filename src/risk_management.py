"""
risk_management.py – Position sizing, stop-loss, volatility filtering,
and drawdown-based exit logic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, annualized_return

log = get_logger(__name__)


class RiskManager:
    """
    Applies risk overlays to a proposed portfolio allocation.
    """

    def __init__(self, cfg: dict) -> None:
        rc = cfg["risk"]
        self.max_portfolio_vol: float = rc.get("max_portfolio_volatility", 0.25)
        self.stop_loss_pct: float = rc.get("stop_loss_pct", -0.15)
        self.max_drawdown_exit: float = rc.get("max_drawdown_exit", -0.20)
        self.vol_filter_window: int = rc.get("volatility_filter_window", 21)
        self.vol_filter_threshold: float = rc.get("volatility_filter_threshold", 0.50)

    # ── Public API ─────────────────────────────────────────────────────────

    def filter_universe(
        self,
        tickers: List[str],
        price_history: Dict[str, pd.DataFrame],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> List[str]:
        """
        Remove stocks that fail volatility or liquidity filters.
        """
        approved = []
        for tkr in tickers:
            if tkr not in price_history:
                continue
            df = price_history[tkr]
            if as_of_date is not None:
                df = df[df.index <= as_of_date]
            if len(df) < self.vol_filter_window:
                log.debug(f"Filtered {tkr}: insufficient history")
                continue
            ret = df["Close"].pct_change().dropna()
            vol = ret.rolling(self.vol_filter_window).std().iloc[-1] * np.sqrt(252)
            if vol > self.vol_filter_threshold:
                log.debug(f"Filtered {tkr}: vol={vol:.2%} > threshold")
                continue
            approved.append(tkr)
        log.info(f"Vol filter: {len(tickers)} → {len(approved)} stocks approved")
        return approved

    def apply_stop_loss(
        self,
        weights: Dict[str, float],
        entry_prices: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Zero out positions that have hit the stop-loss.
        """
        adjusted = dict(weights)
        triggered = []
        for tkr, w in weights.items():
            if tkr not in entry_prices or tkr not in current_prices:
                continue
            entry = entry_prices[tkr]
            curr = current_prices[tkr]
            if entry <= 0:
                continue
            ret_since_entry = curr / entry - 1
            if ret_since_entry <= self.stop_loss_pct:
                adjusted[tkr] = 0.0
                triggered.append((tkr, ret_since_entry))

        if triggered:
            log.warning(f"Stop-loss triggered for: {triggered}")
            # Renormalise
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def scale_for_target_vol(
        self,
        weights: Dict[str, float],
        ret_matrix: pd.DataFrame,
        target_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Scale portfolio weights so that expected portfolio volatility
        does not exceed max_portfolio_vol.
        """
        target = target_vol or self.max_portfolio_vol
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return weights

        w = np.array([weights[t] for t in tickers])
        cov = ret_matrix[tickers].cov().values * 252
        port_vol = np.sqrt(w @ cov @ w)

        if port_vol > target:
            scale = target / port_vol
            log.info(f"Scaling weights by {scale:.3f} (port_vol={port_vol:.2%} > target={target:.2%})")
            scaled = {t: weights[t] * scale for t in weights}
            # Cash for remaining weight
            cash = 1.0 - sum(scaled.values())
            log.info(f"Cash allocation after vol scaling: {cash:.2%}")
            return scaled

        return weights

    def check_portfolio_drawdown(
        self,
        equity_curve: pd.Series,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Returns True if portfolio should be liquidated due to drawdown.
        """
        limit = threshold or self.max_drawdown_exit
        if len(equity_curve) < 2:
            return False
        peak = equity_curve.cummax().iloc[-1]
        current = equity_curve.iloc[-1]
        dd = (current - peak) / (peak + 1e-10)
        if dd <= limit:
            log.warning(f"Portfolio drawdown {dd:.2%} exceeds exit threshold {limit:.2%}")
            return True
        return False

    def kelly_position_size(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Fractional Kelly criterion for position sizing.
        fraction < 1 for risk reduction (e.g., quarter-Kelly).
        """
        if avg_loss == 0:
            return 0.0
        odds = avg_win / abs(avg_loss)
        kelly = win_prob - (1 - win_prob) / odds
        return max(0.0, fraction * kelly)

    def compute_var(
        self,
        weights: Dict[str, float],
        ret_matrix: pd.DataFrame,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Historical Value-at-Risk for the portfolio.
        """
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return 0.0
        w = np.array([weights[t] for t in tickers])
        port_rets = (ret_matrix[tickers] * w).sum(axis=1)
        var = np.percentile(port_rets, (1 - confidence) * 100)
        var_scaled = var * np.sqrt(horizon)
        log.info(f"Portfolio VaR ({confidence:.0%}, {horizon}d): {var_scaled:.2%}")
        return float(var_scaled)

    def compute_cvar(
        self,
        weights: Dict[str, float],
        ret_matrix: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall).
        """
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return 0.0
        w = np.array([weights[t] for t in tickers])
        port_rets = (ret_matrix[tickers] * w).sum(axis=1)
        var = np.percentile(port_rets, (1 - confidence) * 100)
        cvar = port_rets[port_rets <= var].mean()
        log.info(f"Portfolio CVaR ({confidence:.0%}): {cvar:.2%}")
        return float(cvar)
