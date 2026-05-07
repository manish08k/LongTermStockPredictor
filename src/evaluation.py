"""
evaluation.py – Comprehensive strategy evaluation and report generation.
Produces equity curves, performance tables, and drawdown charts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.backtest import BacktestResult
from src.utils import get_logger, annualized_return, sharpe_ratio, max_drawdown

log = get_logger(__name__)


class Evaluator:
    """
    Generates performance analysis reports and charts.
    """

    def __init__(self, cfg: dict) -> None:
        self.annualization = cfg.get("evaluation", {}).get("annualization_factor", 252)
        self.output_dir = Path(cfg.get("evaluation", {}).get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, result: BacktestResult) -> Dict[str, float]:
        """Run all evaluations and save reports."""
        metrics = result.metrics

        # Monthly returns analysis
        monthly = self._monthly_returns(result.returns)

        # Save charts
        self._plot_equity_curve(result)
        self._plot_drawdown(result)
        self._plot_rolling_metrics(result)
        self._plot_monthly_heatmap(result.returns)

        # Print summary
        self._print_summary(metrics, monthly)

        return metrics

    # ── Charts ─────────────────────────────────────────────────────────────

    def _plot_equity_curve(self, result: BacktestResult) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Portfolio Performance vs Benchmark", fontsize=14, fontweight="bold")

        # Equity curves
        ax = axes[0]
        port_norm = result.equity_curve / result.equity_curve.iloc[0]
        bm_norm = result.benchmark_equity / result.benchmark_equity.iloc[0]
        ax.plot(port_norm, label="Portfolio", color="#2196F3", linewidth=2)
        ax.plot(bm_norm, label="Benchmark (Nifty)", color="#FF5722", linewidth=1.5, linestyle="--")
        ax.set_ylabel("Growth of ₹1", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Relative performance
        ax2 = axes[1]
        relative = (port_norm / bm_norm - 1) * 100
        ax2.fill_between(relative.index, relative, 0,
                         where=relative >= 0, alpha=0.4, color="green", label="Outperform")
        ax2.fill_between(relative.index, relative, 0,
                         where=relative < 0, alpha=0.4, color="red", label="Underperform")
        ax2.plot(relative, color="#333333", linewidth=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Relative Return (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "equity_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved equity curve → {path}")

    def _plot_drawdown(self, result: BacktestResult) -> None:
        fig, ax = plt.subplots(figsize=(14, 4))
        equity = result.equity_curve
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max * 100

        ax.fill_between(dd.index, dd, 0, alpha=0.6, color="#F44336", label="Portfolio Drawdown")

        # Benchmark drawdown
        bm_eq = result.benchmark_equity
        bm_roll_max = bm_eq.cummax()
        bm_dd = (bm_eq - bm_roll_max) / bm_roll_max * 100
        ax.plot(bm_dd, color="#FF9800", linewidth=1, linestyle="--", label="Benchmark Drawdown")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_title("Portfolio Drawdown Analysis", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "drawdown.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved drawdown chart → {path}")

    def _plot_rolling_metrics(self, result: BacktestResult) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Rolling Performance Metrics (252-day window)", fontsize=13, fontweight="bold")
        ret = result.returns
        bm_ret = result.benchmark_returns
        window = 252

        # Rolling Sharpe
        ax = axes[0]
        roll_sharpe = (ret.rolling(window).mean() / ret.rolling(window).std()) * np.sqrt(252)
        bm_roll_sharpe = (bm_ret.rolling(window).mean() / bm_ret.rolling(window).std()) * np.sqrt(252)
        ax.plot(roll_sharpe, label="Portfolio", color="#2196F3", linewidth=1.5)
        ax.plot(bm_roll_sharpe, label="Benchmark", color="#FF5722", linewidth=1, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(1, color="green", linewidth=0.5, linestyle=":")
        ax.set_ylabel("Rolling Sharpe", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Rolling Volatility
        ax2 = axes[1]
        roll_vol = ret.rolling(window).std() * np.sqrt(252) * 100
        bm_roll_vol = bm_ret.rolling(window).std() * np.sqrt(252) * 100
        ax2.plot(roll_vol, label="Portfolio Vol", color="#9C27B0", linewidth=1.5)
        ax2.plot(bm_roll_vol, label="Benchmark Vol", color="#FF9800", linewidth=1, linestyle="--")
        ax2.set_ylabel("Rolling Vol (%)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Rolling Alpha
        ax3 = axes[2]
        roll_alpha = (ret.rolling(window).mean() - bm_ret.rolling(window).mean()) * 252 * 100
        ax3.fill_between(roll_alpha.index, roll_alpha, 0,
                         where=roll_alpha >= 0, alpha=0.5, color="green")
        ax3.fill_between(roll_alpha.index, roll_alpha, 0,
                         where=roll_alpha < 0, alpha=0.5, color="red")
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("Rolling Alpha (%)", fontsize=10)
        ax3.set_xlabel("Date", fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "rolling_metrics.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved rolling metrics → {path}")

    def _plot_monthly_heatmap(self, returns: pd.Series) -> None:
        """Seaborn-style monthly return heatmap."""
        try:
            import seaborn as sns
        except ImportError:
            log.debug("seaborn not available; skipping heatmap")
            return

        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_df = monthly.to_frame("ret")
        monthly_df["Year"] = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.month
        heatmap_data = monthly_df.pivot(index="Year", columns="Month", values="ret")
        heatmap_data.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                                 "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig, ax = plt.subplots(figsize=(14, max(4, len(heatmap_data) * 0.5)))
        sns.heatmap(
            heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"label": "Return (%)"}
        )
        ax.set_title("Monthly Returns Heatmap (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Year", fontsize=10)

        plt.tight_layout()
        path = self.output_dir / "monthly_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved monthly heatmap → {path}")

    # ── Text Summary ───────────────────────────────────────────────────────

    @staticmethod
    def _monthly_returns(returns: pd.Series) -> pd.Series:
        return returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    @staticmethod
    def _print_summary(metrics: Dict[str, float], monthly: pd.Series) -> None:
        print("\n" + "=" * 60)
        print("  QUANTITATIVE STRATEGY PERFORMANCE REPORT")
        print("=" * 60)
        rows = [
            ("CAGR",                  f"{metrics.get('cagr', 0):.2%}"),
            ("Benchmark CAGR",        f"{metrics.get('benchmark_cagr', 0):.2%}"),
            ("Annualised Alpha",      f"{metrics.get('alpha', 0):.2%}"),
            ("Sharpe Ratio",          f"{metrics.get('sharpe', 0):.3f}"),
            ("Sortino Ratio",         f"{metrics.get('sortino', 0):.3f}"),
            ("Calmar Ratio",          f"{metrics.get('calmar', 0):.3f}"),
            ("Max Drawdown",          f"{metrics.get('max_drawdown', 0):.2%}"),
            ("Annualised Volatility", f"{metrics.get('volatility', 0):.2%}"),
            ("Beta",                  f"{metrics.get('beta', 0):.3f}"),
            ("Win Rate (daily)",      f"{metrics.get('win_rate', 0):.2%}"),
            ("Total Trades",          f"{int(metrics.get('n_trades', 0))}"),
            ("Transaction Costs",     f"₹{metrics.get('total_transaction_cost', 0):,.0f}"),
            ("Final Portfolio Value", f"₹{metrics.get('final_value', 0):,.0f}"),
            ("Simulation Years",      f"{metrics.get('years', 0):.1f}"),
        ]
        for label, value in rows:
            print(f"  {label:<28} {value:>15}")
        print("=" * 60)

        # Monthly stats
        if len(monthly) > 0:
            print(f"\n  Monthly Return Stats:")
            print(f"    Mean   : {monthly.mean():.2%}")
            print(f"    Median : {monthly.median():.2%}")
            print(f"    Std    : {monthly.std():.2%}")
            print(f"    Best   : {monthly.max():.2%}")
            print(f"    Worst  : {monthly.min():.2%}")
        print()

    def save_metrics_csv(self, metrics: Dict[str, float], filename: str = "metrics.csv") -> None:
        df = pd.DataFrame([metrics])
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        log.info(f"Metrics saved → {path}")
