"""
portfolio.py – Portfolio construction via mean-variance, max Sharpe,
risk parity, and equal-weight optimisation.
Uses PyPortfolioOpt under the hood.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# Try PyPortfolioOpt; fall back to scipy if unavailable
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.risk_models import CovarianceShrinkage
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False
    log.warning("PyPortfolioOpt not available; using scipy fallback")

from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Constructs optimal portfolios from a list of selected tickers.
    """

    def __init__(self, cfg: dict) -> None:
        pc = cfg["portfolio"]
        self.method: str = pc.get("method", "max_sharpe")
        self.max_weight: float = pc.get("max_weight", 0.20)
        self.min_weight: float = pc.get("min_weight", 0.02)
        self.risk_free: float = pc.get("risk_free_rate", 0.065)
        self.regularization: float = pc.get("regularization", 0.0001)

    # ── Public API ─────────────────────────────────────────────────────────

    def optimize(
        self,
        tickers: List[str],
        price_history: Dict[str, pd.DataFrame],
        scores: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio weights.

        Parameters
        ----------
        tickers       : list of selected tickers
        price_history : dict[ticker -> OHLCV DataFrame]
        scores        : optional ensemble scores for score-tilted weighting

        Returns
        -------
        dict[ticker -> weight]
        """
        if not tickers:
            log.warning("No tickers for portfolio optimization")
            return {}

        # Build return matrix
        ret_matrix = self._build_return_matrix(tickers, price_history)
        if ret_matrix is None or ret_matrix.empty:
            return self._equal_weight(tickers)

        # Need at least 30 observations
        if len(ret_matrix) < 30:
            log.warning("Insufficient return history; using equal weights")
            return self._equal_weight(tickers)

        # Optimise
        try:
            if self.method == "max_sharpe" and HAS_PYPFOPT:
                weights = self._max_sharpe_pypfopt(ret_matrix)
            elif self.method == "min_volatility" and HAS_PYPFOPT:
                weights = self._min_vol_pypfopt(ret_matrix)
            elif self.method == "risk_parity":
                weights = self._risk_parity(ret_matrix)
            elif self.method == "equal_weight":
                weights = self._equal_weight(tickers)
            else:
                weights = self._max_sharpe_scipy(ret_matrix)

            # Score-tilt (optional)
            if scores is not None:
                weights = self._score_tilt(weights, scores, tilt_strength=0.2)

            weights = self._apply_constraints(weights)
            log.info(f"Portfolio [{self.method}]: {len(weights)} stocks | "
                     f"max={max(weights.values()):.2%} min={min(weights.values()):.2%}")
            return weights

        except Exception as e:
            log.error(f"Optimisation failed ({e}); falling back to equal weight")
            return self._equal_weight(tickers)

    # ── Optimization methods ───────────────────────────────────────────────

    def _max_sharpe_pypfopt(self, ret_matrix: pd.DataFrame) -> Dict[str, float]:
        mu = expected_returns.mean_historical_return(ret_matrix, returns_data=True)
        S = CovarianceShrinkage(ret_matrix, returns_data=True).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        ef.add_objective(lambda w: self.regularization * (w ** 2).sum())
        ef.max_sharpe(risk_free_rate=self.risk_free / 252)
        return dict(ef.clean_weights())

    def _min_vol_pypfopt(self, ret_matrix: pd.DataFrame) -> Dict[str, float]:
        mu = expected_returns.mean_historical_return(ret_matrix, returns_data=True)
        S = CovarianceShrinkage(ret_matrix, returns_data=True).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        ef.min_volatility()
        return dict(ef.clean_weights())

    def _max_sharpe_scipy(self, ret_matrix: pd.DataFrame) -> Dict[str, float]:
        """Scipy-based mean-variance optimisation (fallback)."""
        n = ret_matrix.shape[1]
        mu = ret_matrix.mean().values * 252
        cov = ret_matrix.cov().values * 252

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_ret - self.risk_free) / (port_vol + 1e-10)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        weights = result.x
        return dict(zip(ret_matrix.columns, weights))

    def _risk_parity(self, ret_matrix: pd.DataFrame) -> Dict[str, float]:
        """Equal Risk Contribution portfolio."""
        n = ret_matrix.shape[1]
        cov = ret_matrix.cov().values * 252

        def risk_budget_obj(w):
            port_var = w @ cov @ w
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk / (port_var + 1e-10)
            target = np.ones(n) / n
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_budget_obj, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000},
        )
        return dict(zip(ret_matrix.columns, result.x))

    @staticmethod
    def _equal_weight(tickers: List[str]) -> Dict[str, float]:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_return_matrix(
        tickers: List[str], price_history: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        frames = {}
        for t in tickers:
            if t in price_history:
                frames[t] = price_history[t]["Close"].pct_change()
        if not frames:
            return None
        df = pd.DataFrame(frames).dropna(how="all")
        df = df.ffill().dropna(how="any")
        return df

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clip weights to [min, max] and renormalise."""
        w = {k: np.clip(v, self.min_weight, self.max_weight) for k, v in weights.items()}
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    @staticmethod
    def _score_tilt(
        weights: Dict[str, float],
        scores: pd.Series,
        tilt_strength: float = 0.2,
    ) -> Dict[str, float]:
        """Tilt portfolio weights towards higher-scored stocks."""
        available = {k: v for k, v in weights.items() if k in scores.index}
        if not available:
            return weights
        score_arr = pd.Series(available).index.map(lambda t: scores.get(t, 0))
        score_pct = pd.Series(score_arr.values, index=list(available.keys()))
        score_pct = (score_pct - score_pct.min()) / (score_pct.max() - score_pct.min() + 1e-10)
        tilted = {
            k: v * (1 + tilt_strength * score_pct.get(k, 0))
            for k, v in weights.items()
        }
        total = sum(tilted.values())
        return {k: v / total for k, v in tilted.items()}

    def get_portfolio_metrics(
        self,
        weights: Dict[str, float],
        ret_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute ex-ante portfolio statistics."""
        tickers = [t for t in weights if t in ret_matrix.columns]
        w = np.array([weights[t] for t in tickers])
        ret_sub = ret_matrix[tickers]
        mu = ret_sub.mean().values * 252
        cov = ret_sub.cov().values * 252

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (port_ret - self.risk_free) / (port_vol + 1e-10)

        return {
            "expected_return": port_ret,
            "expected_volatility": port_vol,
            "expected_sharpe": sharpe,
        }
