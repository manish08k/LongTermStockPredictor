"""
portfolio.py – Portfolio optimization: max Sharpe, min vol, risk parity, HRP.
Adds sector exposure limits, turnover constraints, and score tilting.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.utils import get_logger

log = get_logger(__name__)

try:
    from pypfopt import EfficientFrontier, expected_returns
    from pypfopt.risk_models import CovarianceShrinkage
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False
    log.debug("PyPortfolioOpt not installed; using scipy fallback")


class PortfolioOptimizer:
    """
    Constructs optimal portfolios from a universe of selected tickers.
    Supports: max_sharpe, min_volatility, risk_parity, hrp, equal_weight.
    """

    def __init__(self, cfg: dict) -> None:
        pc = cfg.get("portfolio", {})
        self.method: str = pc.get("method", "max_sharpe")
        self.max_weight: float = pc.get("max_weight", 0.12)
        self.min_weight: float = pc.get("min_weight", 0.03)
        self.risk_free: float = pc.get("risk_free_rate", 0.065)
        self.regularization: float = pc.get("regularization", 0.0001)
        self.sector_max: float = pc.get("sector_max_exposure", 0.40)
        self.turnover_limit: float = pc.get("turnover_limit", 1.0)
        self.sectors: Dict[str, str] = cfg.get("data", {}).get("sectors", {})
        self._prev_weights: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def optimize(
        self,
        tickers: List[str],
        price_history: Dict[str, pd.DataFrame],
        scores: Optional[pd.Series] = None,
        lookback_days: int = 252,
    ) -> Dict[str, float]:
        """
        Compute portfolio weights for the given tickers.

        Parameters
        ----------
        tickers       : pre-selected stock universe
        price_history : dict[ticker -> OHLCV DataFrame]
        scores        : ensemble scores for score-tilted weighting
        lookback_days : return history for covariance estimation

        Returns
        -------
        dict[ticker -> weight], weights sum to 1
        """
        if not tickers:
            log.warning("No tickers provided to optimizer")
            return {}

        # Build return matrix
        ret_matrix = self._build_return_matrix(tickers, price_history, lookback_days)
        if ret_matrix is None or len(ret_matrix) < 30:
            log.warning("Insufficient history; using equal weights")
            return self._normalize(self._equal_weight(tickers))

        # Remove tickers with all-NaN returns
        valid_tickers = [t for t in tickers if t in ret_matrix.columns]
        if not valid_tickers:
            return self._normalize(self._equal_weight(tickers))
        ret_matrix = ret_matrix[valid_tickers]

        # Optimize
        try:
            weights = self._dispatch_method(ret_matrix)
        except Exception as e:
            log.warning(f"Optimization [{self.method}] failed: {e}; using equal weights")
            weights = self._equal_weight(valid_tickers)

        # Score tilt
        if scores is not None:
            weights = self._score_tilt(weights, scores, tilt_strength=0.20)

        # Apply constraints
        weights = self._apply_weight_bounds(weights)
        weights = self._apply_sector_constraints(weights)
        weights = self._apply_turnover_constraint(weights)
        weights = self._normalize(weights)

        log.info(
            f"Portfolio [{self.method}]: {len(weights)} stocks | "
            f"max={max(weights.values()):.2%} min={min(weights.values()):.2%}"
        )
        self._prev_weights = dict(weights)
        return weights

    # ── Dispatch ───────────────────────────────────────────────────────────

    def _dispatch_method(self, ret_matrix: pd.DataFrame) -> Dict[str, float]:
        if self.method == "max_sharpe" and HAS_PYPFOPT:
            return self._max_sharpe_pypfopt(ret_matrix)
        elif self.method == "min_volatility" and HAS_PYPFOPT:
            return self._min_vol_pypfopt(ret_matrix)
        elif self.method == "risk_parity":
            return self._risk_parity(ret_matrix)
        elif self.method == "hrp":
            return self._hrp(ret_matrix)
        elif self.method == "equal_weight":
            return self._equal_weight(list(ret_matrix.columns))
        else:
            return self._max_sharpe_scipy(ret_matrix)

    # ── Optimization methods ───────────────────────────────────────────────

    def _max_sharpe_pypfopt(self, ret: pd.DataFrame) -> Dict[str, float]:
        mu = expected_returns.mean_historical_return(ret, returns_data=True)
        S = CovarianceShrinkage(ret, returns_data=True).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        ef.max_sharpe(risk_free_rate=self.risk_free / 252)
        return {k: v for k, v in ef.clean_weights().items() if v > 1e-4}

    def _min_vol_pypfopt(self, ret: pd.DataFrame) -> Dict[str, float]:
        mu = expected_returns.mean_historical_return(ret, returns_data=True)
        S = CovarianceShrinkage(ret, returns_data=True).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        ef.min_volatility()
        return {k: v for k, v in ef.clean_weights().items() if v > 1e-4}

    def _max_sharpe_scipy(self, ret: pd.DataFrame) -> Dict[str, float]:
        n = ret.shape[1]
        mu = ret.mean().values * 252
        cov = ret.cov().values * 252 + np.eye(n) * self.regularization

        def neg_sharpe(w):
            r = w @ mu
            v = np.sqrt(w @ cov @ w)
            return -(r - self.risk_free) / (v + 1e-10)

        result = minimize(
            neg_sharpe, np.ones(n) / n, method="SLSQP",
            bounds=[(self.min_weight, self.max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        return dict(zip(ret.columns, np.clip(result.x, 0, None)))

    def _risk_parity(self, ret: pd.DataFrame) -> Dict[str, float]:
        """Equal Risk Contribution (ERC)."""
        n = ret.shape[1]
        cov = ret.cov().values * 252 + np.eye(n) * 1e-6

        def erc_obj(w):
            pv = w @ cov @ w
            mrc = cov @ w
            rc = w * mrc / (pv + 1e-10)
            target = np.ones(n) / n
            return float(np.sum((rc - target) ** 2))

        result = minimize(
            erc_obj, np.ones(n) / n, method="SLSQP",
            bounds=[(self.min_weight, self.max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 1000},
        )
        return dict(zip(ret.columns, np.clip(result.x, 0, None)))

    def _hrp(self, ret: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (Lopez de Prado 2016).
        Does not require mean return estimation – more robust.
        """
        cov = ret.cov()
        corr = ret.corr()

        # Correlation-based distance matrix
        dist = ((1 - corr) / 2) ** 0.5
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method="single")
        sorted_tickers = [corr.columns[i] for i in leaves_list(link)]

        # Recursive bisection
        weights = self._hrp_recursive(cov, sorted_tickers)
        return dict(zip(sorted_tickers, [weights[t] for t in sorted_tickers]))

    def _hrp_recursive(
        self, cov: pd.DataFrame, tickers: List[str]
    ) -> Dict[str, float]:
        """Recursive bisection for HRP."""
        if len(tickers) == 1:
            return {tickers[0]: 1.0}
        mid = len(tickers) // 2
        left = tickers[:mid]
        right = tickers[mid:]

        w_left = self._hrp_recursive(cov, left)
        w_right = self._hrp_recursive(cov, right)

        # Variance of each cluster
        def cluster_var(tkrs, w_dict):
            w = np.array([w_dict[t] for t in tkrs])
            sub_cov = cov.loc[tkrs, tkrs].values
            return float(w @ sub_cov @ w)

        var_l = cluster_var(left, w_left)
        var_r = cluster_var(right, w_right)
        alpha = 1 - var_l / (var_l + var_r + 1e-10)

        result = {}
        for t, w in w_left.items():
            result[t] = alpha * w
        for t, w in w_right.items():
            result[t] = (1 - alpha) * w
        return result

    @staticmethod
    def _equal_weight(tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    # ── Constraints ────────────────────────────────────────────────────────

    def _apply_weight_bounds(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clip each weight to [min_weight, max_weight]."""
        return {
            k: float(np.clip(v, self.min_weight, self.max_weight))
            for k, v in weights.items()
            if v > 0
        }

    def _apply_sector_constraints(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Reduce sector over-weight iteratively."""
        if not self.sectors:
            return weights

        w = dict(weights)
        total = sum(w.values()) or 1.0

        # Compute sector totals
        sector_sums: Dict[str, float] = {}
        for tkr, wt in w.items():
            sec = self.sectors.get(tkr, "Unknown")
            sector_sums[sec] = sector_sums.get(sec, 0) + wt / total

        for sec, sec_weight in sector_sums.items():
            if sec_weight > self.sector_max:
                # Scale down all stocks in this sector
                factor = self.sector_max / sec_weight
                for tkr in w:
                    if self.sectors.get(tkr, "Unknown") == sec:
                        w[tkr] *= factor

        return w

    def _apply_turnover_constraint(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Limit portfolio turnover vs previous weights."""
        if not self._prev_weights or self.turnover_limit >= 1.0:
            return weights

        # Compute proposed turnover
        all_tickers = set(weights) | set(self._prev_weights)
        turnover = sum(
            abs(weights.get(t, 0) - self._prev_weights.get(t, 0))
            for t in all_tickers
        )
        if turnover <= self.turnover_limit:
            return weights

        # Blend with previous weights
        blend = self.turnover_limit / (turnover + 1e-10)
        blended = {}
        for t in all_tickers:
            new_w = weights.get(t, 0)
            old_w = self._prev_weights.get(t, 0)
            blended[t] = old_w + blend * (new_w - old_w)

        return {t: w for t, w in blended.items() if w > 1e-4}

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(abs(v) for v in weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _score_tilt(
        weights: Dict[str, float],
        scores: pd.Series,
        tilt_strength: float = 0.20,
    ) -> Dict[str, float]:
        """Tilt weights towards higher-scored stocks."""
        score_dict = scores.to_dict() if isinstance(scores, pd.Series) else scores
        available = {k: v for k, v in weights.items() if k in score_dict}
        if not available:
            return weights

        sc = np.array([score_dict.get(t, 0) for t in available])
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max > sc_min:
            sc_norm = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_norm = np.ones(len(sc)) * 0.5

        tilted = {}
        for i, (t, w) in enumerate(available.items()):
            tilted[t] = w * (1 + tilt_strength * sc_norm[i])

        # Add back tickers not in scores
        for t, w in weights.items():
            if t not in tilted:
                tilted[t] = w

        total = sum(tilted.values())
        return {k: v / total for k, v in tilted.items()}

    @staticmethod
    def _build_return_matrix(
        tickers: List[str],
        price_history: Dict[str, pd.DataFrame],
        lookback: int = 252,
    ) -> Optional[pd.DataFrame]:
        frames = {}
        for t in tickers:
            if t not in price_history:
                continue
            close = price_history[t]["Close"]
            if len(close) < lookback // 2:
                continue
            ret = close.pct_change().tail(lookback)
            frames[t] = ret

        if not frames:
            return None

        df = pd.DataFrame(frames).dropna(how="all").ffill().dropna(how="any")
        return df if len(df) >= 30 else None

    def get_portfolio_stats(
        self,
        weights: Dict[str, float],
        price_history: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Ex-ante portfolio stats."""
        ret = self._build_return_matrix(list(weights.keys()), price_history)
        if ret is None:
            return {}
        tickers = [t for t in weights if t in ret.columns]
        w = np.array([weights[t] for t in tickers])
        mu = ret[tickers].mean().values * 252
        cov = ret[tickers].cov().values * 252
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w + 1e-10))
        return {
            "expected_return": port_ret,
            "expected_volatility": port_vol,
            "expected_sharpe": (port_ret - self.risk_free) / (port_vol + 1e-10),
        }