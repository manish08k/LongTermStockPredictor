"""
risk_management.py – Position sizing, volatility/liquidity filtering,
stop-loss, VaR, CVaR, Kelly criterion.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, annualized_return

log = get_logger(__name__)


class RiskManager:
    def __init__(self, cfg: dict) -> None:
        rc = cfg.get("risk", {})
        self.max_portfolio_vol: float = rc.get("max_portfolio_volatility", 0.25)
        self.stop_loss_pct: float = rc.get("stop_loss_pct", -0.15)
        self.max_drawdown_exit: float = rc.get("max_drawdown_exit", -0.25)
        self.vol_window: int = rc.get("volatility_filter_window", 21)
        self.vol_threshold: float = rc.get("volatility_filter_threshold", 0.60)
        self.min_adv: float = rc.get("min_liquidity_adv", 0)

    def filter_universe(
        self,
        tickers: List[str],
        price_history: Dict[str, pd.DataFrame],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> List[str]:
        """Filter stocks by volatility and liquidity thresholds."""
        approved = []
        for tkr in tickers:
            if tkr not in price_history:
                continue
            df = price_history[tkr]
            if as_of_date is not None:
                df = df[df.index <= as_of_date]
            if len(df) < self.vol_window:
                continue
            ret = df["Close"].pct_change().dropna()
            vol = ret.rolling(self.vol_window).std().iloc[-1] * np.sqrt(252)
            if np.isnan(vol) or vol > self.vol_threshold:
                log.debug(f"Vol-filtered {tkr}: {vol:.2%}")
                continue
            # Liquidity filter (ADV in rupees)
            if self.min_adv > 0 and "Volume" in df.columns:
                adv = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
                if np.isnan(adv) or adv < self.min_adv:
                    log.debug(f"Liquidity-filtered {tkr}: ADV={adv:,.0f}")
                    continue
            approved.append(tkr)
        log.info(f"Risk filter: {len(tickers)} → {len(approved)} stocks")
        return approved

    def apply_stop_loss(
        self,
        weights: Dict[str, float],
        entry_prices: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> Dict[str, float]:
        adjusted = dict(weights)
        triggered = []
        for tkr, w in weights.items():
            if tkr not in entry_prices or tkr not in current_prices:
                continue
            entry = entry_prices[tkr]
            curr = current_prices[tkr]
            if entry <= 0:
                continue
            ret = curr / entry - 1
            if ret <= self.stop_loss_pct:
                adjusted[tkr] = 0.0
                triggered.append((tkr, f"{ret:.2%}"))
        if triggered:
            log.warning(f"Stop-loss: {triggered}")
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
        target = target_vol or self.max_portfolio_vol
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return weights
        w = np.array([weights[t] for t in tickers])
        cov = ret_matrix[tickers].cov().values * 252
        port_vol = float(np.sqrt(w @ cov @ w))
        if port_vol > target:
            scale = target / port_vol
            log.info(f"Scaling portfolio by {scale:.3f} (vol={port_vol:.2%} > target={target:.2%})")
            scaled = {t: weights[t] * scale for t in tickers}
            for t in weights:
                if t not in scaled:
                    scaled[t] = weights[t] * scale
            return scaled
        return weights

    def check_portfolio_drawdown(
        self, equity_curve: pd.Series, threshold: Optional[float] = None
    ) -> bool:
        limit = threshold or self.max_drawdown_exit
        if len(equity_curve) < 2:
            return False
        peak = equity_curve.cummax().iloc[-1]
        curr = equity_curve.iloc[-1]
        dd = (curr - peak) / (peak + 1e-10)
        if dd <= limit:
            log.warning(f"Portfolio DD {dd:.2%} exceeds exit limit {limit:.2%}")
            return True
        return False

    def kelly_position_size(
        self, win_prob: float, avg_win: float, avg_loss: float, fraction: float = 0.25
    ) -> float:
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
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return 0.0
        w = np.array([weights[t] for t in tickers])
        port_rets = (ret_matrix[tickers] * w).sum(axis=1)
        var = np.percentile(port_rets.dropna(), (1 - confidence) * 100)
        return float(var * np.sqrt(horizon))

    def compute_cvar(
        self, weights: Dict[str, float], ret_matrix: pd.DataFrame, confidence: float = 0.95
    ) -> float:
        tickers = [t for t in weights if t in ret_matrix.columns]
        if not tickers:
            return 0.0
        w = np.array([weights[t] for t in tickers])
        port_rets = (ret_matrix[tickers] * w).sum(axis=1).dropna()
        var = np.percentile(port_rets, (1 - confidence) * 100)
        return float(port_rets[port_rets <= var].mean())