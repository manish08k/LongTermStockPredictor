"""
backtest.py – Walk-forward backtesting engine with realistic assumptions:
transaction costs, slippage, monthly/quarterly rebalancing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, annualized_return, sharpe_ratio, max_drawdown

log = get_logger(__name__)


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    direction: str          # "buy" | "sell"
    shares: float
    price: float
    cost: float             # transaction cost in currency
    slippage: float

    @property
    def value(self) -> float:
        return self.shares * self.price


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    trades: List[Trade] = field(default_factory=list)
    portfolio_weights: Dict[pd.Timestamp, Dict[str, float]] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Walk-forward backtesting with realistic market impact model.
    """

    def __init__(self, cfg: dict) -> None:
        bc = cfg["backtest"]
        self.initial_capital: float = bc.get("initial_capital", 10_000_000)
        self.rebalance_freq: str = bc.get("rebalance_frequency", "monthly")
        self.tc_bps: float = bc.get("transaction_cost_bps", 20) / 10_000
        self.slippage_bps: float = bc.get("slippage_bps", 10) / 10_000
        self.min_trading_days: int = bc.get("min_trading_days", 126)
        self.wf_cfg: dict = bc.get("walk_forward", {})
        self.risk_free: float = cfg.get("portfolio", {}).get("risk_free_rate", 0.065)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        portfolio_weights_by_date: Dict[pd.Timestamp, Dict[str, float]],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Simulate portfolio over time.

        Parameters
        ----------
        portfolio_weights_by_date : {rebalance_date -> {ticker: weight}}
        price_data : {ticker -> OHLCV DataFrame}
        benchmark_data : OHLCV DataFrame for benchmark index

        Returns
        -------
        BacktestResult
        """
        log.info(f"Starting backtest | Capital: ₹{self.initial_capital:,.0f} | "
                 f"Rebalance: {self.rebalance_freq}")

        # Build common date index
        all_dates = self._build_date_index(price_data)
        if len(all_dates) < self.min_trading_days:
            raise ValueError(f"Insufficient trading days: {len(all_dates)}")

        # Build close price matrix
        close_matrix = self._build_close_matrix(price_data, all_dates)
        bm_close = benchmark_data["Close"].reindex(all_dates, method="ffill")

        # Simulation state
        cash = self.initial_capital
        positions: Dict[str, float] = {}      # ticker -> shares
        entry_prices: Dict[str, float] = {}   # ticker -> entry price
        equity_history: List[float] = []
        trades: List[Trade] = []

        rebal_dates = sorted(portfolio_weights_by_date.keys())
        next_rebal_idx = 0

        for date in all_dates:
            # ── Rebalance if needed ────────────────────────────────────────
            if next_rebal_idx < len(rebal_dates) and date >= rebal_dates[next_rebal_idx]:
                target_weights = portfolio_weights_by_date[rebal_dates[next_rebal_idx]]
                cash, new_trades = self._rebalance(
                    date, cash, positions, entry_prices,
                    target_weights, close_matrix,
                )
                trades.extend(new_trades)
                next_rebal_idx += 1

            # ── Mark-to-market ─────────────────────────────────────────────
            port_value = cash
            for tkr, shares in positions.items():
                price = close_matrix.loc[date, tkr] if tkr in close_matrix.columns else 0
                port_value += shares * price
            equity_history.append(port_value)

        equity = pd.Series(equity_history, index=all_dates, name="portfolio")
        returns = equity.pct_change().dropna()
        bm_returns = bm_close.pct_change().dropna()

        # Align
        common_idx = returns.index.intersection(bm_returns.index)
        returns = returns.loc[common_idx]
        bm_returns = bm_returns.loc[common_idx]
        bm_equity = (1 + bm_returns).cumprod() * self.initial_capital

        result = BacktestResult(
            equity_curve=equity,
            returns=returns,
            benchmark_equity=bm_equity,
            benchmark_returns=bm_returns,
            trades=trades,
            portfolio_weights=portfolio_weights_by_date,
        )
        result.metrics = self._compute_metrics(returns, bm_returns, equity, trades)
        self._log_metrics(result.metrics)
        return result

    # ── Rebalancing ────────────────────────────────────────────────────────

    def _rebalance(
        self,
        date: pd.Timestamp,
        cash: float,
        positions: Dict[str, float],
        entry_prices: Dict[str, float],
        target_weights: Dict[str, float],
        close_matrix: pd.DataFrame,
    ) -> Tuple[float, List[Trade]]:
        """
        Execute rebalance to target_weights. Returns (new_cash, trades_executed).
        """
        trades: List[Trade] = []

        # Compute current portfolio value
        port_value = cash
        for tkr, shares in positions.items():
            price = close_matrix.loc[date, tkr] if tkr in close_matrix.columns else 0
            port_value += shares * price

        if port_value <= 0:
            log.warning(f"{date}: Portfolio value ≤ 0; skipping rebalance")
            return cash, trades

        # Compute target dollar values
        target_dollars = {
            tkr: w * port_value
            for tkr, w in target_weights.items()
            if w > 0
        }

        # Sell positions not in target or reduced
        for tkr in list(positions.keys()):
            if tkr not in close_matrix.columns:
                continue
            price = close_matrix.loc[date, tkr]
            target_val = target_dollars.get(tkr, 0.0)
            current_val = positions[tkr] * price
            delta = target_val - current_val

            if abs(delta) < port_value * 0.001:  # ignore tiny trades
                continue

            if delta < 0:  # sell
                shares_to_sell = abs(delta) / (price * (1 + self.slippage_bps))
                shares_to_sell = min(shares_to_sell, positions.get(tkr, 0))
                exec_price = price * (1 - self.slippage_bps)
                tc = shares_to_sell * exec_price * self.tc_bps
                cash += shares_to_sell * exec_price - tc
                positions[tkr] = positions.get(tkr, 0) - shares_to_sell
                if positions[tkr] <= 0:
                    positions.pop(tkr, None)
                    entry_prices.pop(tkr, None)
                trades.append(Trade(date, tkr, "sell", shares_to_sell, exec_price, tc, self.slippage_bps))

        # Buy new/increased positions
        for tkr, target_val in target_dollars.items():
            if tkr not in close_matrix.columns:
                log.debug(f"Skipping {tkr}: not in price matrix")
                continue
            price = close_matrix.loc[date, tkr]
            if price <= 0:
                continue
            current_val = positions.get(tkr, 0) * price
            delta = target_val - current_val
            if delta > 0 and cash >= delta:
                exec_price = price * (1 + self.slippage_bps)
                shares_to_buy = delta / exec_price
                tc = shares_to_buy * exec_price * self.tc_bps
                if cash >= shares_to_buy * exec_price + tc:
                    cash -= shares_to_buy * exec_price + tc
                    positions[tkr] = positions.get(tkr, 0) + shares_to_buy
                    entry_prices[tkr] = exec_price
                    trades.append(Trade(date, tkr, "buy", shares_to_buy, exec_price, tc, self.slippage_bps))

        return cash, trades

    # ── Performance metrics ────────────────────────────────────────────────

    def _compute_metrics(
        self,
        returns: pd.Series,
        bm_returns: pd.Series,
        equity: pd.Series,
        trades: List[Trade],
    ) -> Dict[str, float]:
        n_years = len(returns) / 252
        port_cagr = annualized_return(returns)
        bm_cagr = annualized_return(bm_returns)
        port_sharpe = sharpe_ratio(returns, rf=self.risk_free)
        bm_sharpe = sharpe_ratio(bm_returns, rf=self.risk_free)
        port_mdd = max_drawdown(equity)
        port_vol = returns.std() * np.sqrt(252)

        # Alpha / Beta
        cov_matrix = np.cov(returns.values, bm_returns.values)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)
        alpha = port_cagr - (self.risk_free + beta * (bm_cagr - self.risk_free))

        # Win rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0

        # Calmar
        calmar = port_cagr / abs(port_mdd) if port_mdd != 0 else 0.0

        # Sortino
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = (port_cagr - self.risk_free) / (downside + 1e-10)

        # Trade statistics
        n_trades = len(trades)
        total_cost = sum(t.cost for t in trades)

        return {
            "cagr": port_cagr,
            "benchmark_cagr": bm_cagr,
            "excess_return": port_cagr - bm_cagr,
            "sharpe": port_sharpe,
            "benchmark_sharpe": bm_sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": port_mdd,
            "volatility": port_vol,
            "beta": beta,
            "alpha": alpha,
            "win_rate": win_rate,
            "n_trades": n_trades,
            "total_transaction_cost": total_cost,
            "years": n_years,
            "final_value": float(equity.iloc[-1]) if len(equity) > 0 else 0,
        }

    @staticmethod
    def _log_metrics(metrics: Dict[str, float]) -> None:
        log.info("=" * 55)
        log.info("BACKTEST RESULTS")
        log.info("=" * 55)
        log.info(f"  CAGR              : {metrics['cagr']:.2%}")
        log.info(f"  Benchmark CAGR    : {metrics['benchmark_cagr']:.2%}")
        log.info(f"  Excess Return     : {metrics['excess_return']:.2%}")
        log.info(f"  Sharpe Ratio      : {metrics['sharpe']:.3f}")
        log.info(f"  Sortino Ratio     : {metrics['sortino']:.3f}")
        log.info(f"  Calmar Ratio      : {metrics['calmar']:.3f}")
        log.info(f"  Max Drawdown      : {metrics['max_drawdown']:.2%}")
        log.info(f"  Ann. Volatility   : {metrics['volatility']:.2%}")
        log.info(f"  Beta              : {metrics['beta']:.3f}")
        log.info(f"  Alpha             : {metrics['alpha']:.2%}")
        log.info(f"  Win Rate          : {metrics['win_rate']:.2%}")
        log.info(f"  Total Trades      : {metrics['n_trades']}")
        log.info(f"  Transaction Costs : ₹{metrics['total_transaction_cost']:,.0f}")
        log.info(f"  Final Value       : ₹{metrics['final_value']:,.0f}")
        log.info("=" * 55)

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_date_index(price_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        all_idx = set()
        for df in price_data.values():
            all_idx.update(df.index)
        return pd.DatetimeIndex(sorted(all_idx))

    @staticmethod
    def _build_close_matrix(
        price_data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        frames = {}
        for tkr, df in price_data.items():
            frames[tkr] = df["Close"].reindex(dates, method="ffill")
        return pd.DataFrame(frames)

    def get_rebalance_dates(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[pd.Timestamp]:
        """Generate rebalance dates based on configured frequency."""
        freq_map = {
            "daily": "B",
            "weekly": "W-FRI",
            "monthly": "BMS",
            "quarterly": "QS",
        }
        freq = freq_map.get(self.rebalance_freq, "BMS")
        return list(pd.date_range(start=start, end=end, freq=freq))
