"""
labeling.py – Forward-return labels for multi-horizon prediction.
Supports: binary (outperform), regression (raw return), meta-labeling.
Strictly avoids look-ahead bias: labels use only future prices from t.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, timeit

log = get_logger(__name__)


class Labeler:
    """
    Attaches forward-return labels to the feature panel.

    Parameters
    ----------
    cfg : dict
        Full config dict. Reads labeling section.
    """

    def __init__(self, cfg: dict) -> None:
        lc = cfg["labeling"]
        self.horizons: list[int] = lc.get("horizons", [21,63,126,252])
        self.primary_horizon: int = lc.get("primary_horizon", 63)
        self.label_type: str = lc.get("label_type", "binary")
        self.outperform_threshold: float = lc.get("outperform_threshold", 0.0)
        self.triple_barrier: dict = lc.get("triple_barrier", {"enabled": False})
        self.benchmark_ticker: str = cfg["data"].get("benchmark_ticker", cfg["data"].get("benchmark", "^NSEI"))

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def label(
        self,
        panel: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute forward-return labels and attach to panel.

        Returns
        -------
        panel with additional columns:
          - fwd_ret_{h}d         : raw forward return for horizon h
          - label_{h}d           : binary 0/1 outperform label
          - label_primary        : primary horizon binary label
        """
        log.info(f"Labeling horizons: {self.horizons} days | type={self.label_type}")
        result = panel.copy()

        # Compute benchmark forward returns (if available)
        bm_fwd: Dict[int, pd.Series] = {}
        if benchmark_df is not None:
            bm_close = benchmark_df["Close"]
            for h in self.horizons:
                bm_fwd[h] = self._compute_forward_return(bm_close, h)

        # Compute stock forward returns
        close = result["Close"]
        for h in self.horizons:
            fwd_ret = self._compute_forward_return_panel(close, h)
            result[f"fwd_ret_{h}d"] = fwd_ret

            # Binary: outperform benchmark
            if self.label_type == "binary":
                if h in bm_fwd:
                    # Reindex benchmark to match panel dates
                    bm_h = bm_fwd[h].reindex(result.index.get_level_values("Date"))
                    bm_h.index = result.index
                    excess = fwd_ret - bm_h
                else:
                    excess = fwd_ret  # vs 0

                result[f"label_{h}d"] = (excess > self.outperform_threshold).astype(int)

            elif self.label_type == "regression":
                result[f"label_{h}d"] = fwd_ret

            elif self.label_type == "multi_class":
                result[f"label_{h}d"] = self._multi_class_label(fwd_ret)

        # Primary horizon label
        result["label_primary"] = result[f"label_{self.primary_horizon}d"]
        result["fwd_ret_primary"] = result[f"fwd_ret_{self.primary_horizon}d"]

        # Triple barrier labeling (optional)
        if self.triple_barrier.get("enabled", False):
            tb_labels = self._triple_barrier_label(close)
            result["label_tb"] = tb_labels

        # Annualized return
        for h in self.horizons:
            result[f"ann_ret_{h}d"] = (1 + result[f"fwd_ret_{h}d"]) ** (252 / h) - 1

        # Meta-labeling: are we confident enough?
        result["meta_label"] = self._meta_label(result)

        # Drop rows where label is NaN (end of sample, no future available)
        n_before = len(result)
        result = result.dropna(subset=["label_primary"])
        log.info(
            f"Labels attached: {len(result)} rows (dropped {n_before - len(result)} "
            f"NaN-label rows from end of sample)"
        )
        log.info(
            f"Primary label balance: "
            f"{result['label_primary'].value_counts().to_dict()}"
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compute_forward_return_panel(close: pd.Series, horizon: int) -> pd.Series:
        """
        For each (Date, Ticker), compute return from t to t+horizon.
        Uses groupby Ticker + shift(-horizon).
        No look-ahead: shift is applied forward, labels are at time t
        but describe what happens between t and t+horizon.
        """
        return close.groupby(level="Ticker").transform(
            lambda x: x.shift(-horizon) / x - 1
        )

    @staticmethod
    def _compute_forward_return(close: pd.Series, horizon: int) -> pd.Series:
        """Single-series forward return (for benchmark)."""
        return close.shift(-horizon) / close - 1

    @staticmethod
    def _multi_class_label(fwd_ret: pd.Series) -> pd.Series:
        """
        3-class label:
          2 = strong outperform (top quartile)
          1 = moderate (middle 50%)
          0 = underperform (bottom quartile)
        """
        q25 = fwd_ret.groupby(level="Date").transform(lambda x: x.quantile(0.25))
        q75 = fwd_ret.groupby(level="Date").transform(lambda x: x.quantile(0.75))
        label = pd.Series(1, index=fwd_ret.index)
        label[fwd_ret <= q25] = 0
        label[fwd_ret >= q75] = 2
        return label

    def _triple_barrier_label(self, close: pd.Series) -> pd.Series:
        """
        Labels: +1 = hit profit take first, -1 = hit stop first, 0 = timeout.
        """
        pt = self.triple_barrier["profit_taking"]
        sl = self.triple_barrier["stop_loss"]
        max_hold = self.triple_barrier["max_holding"]

        log.info(f"Triple barrier: PT={pt}, SL={sl}, max_hold={max_hold}")

        labels = pd.Series(np.nan, index=close.index)

        def _label_ticker(prices: pd.Series) -> pd.Series:
            result = pd.Series(0, index=prices.index)
            arr = prices.values
            for i in range(len(arr)):
                entry = arr[i]
                if np.isnan(entry) or entry <= 0:
                    continue
                end = min(i + max_hold, len(arr))
                for j in range(i + 1, end):
                    ret = arr[j] / entry - 1
                    if ret >= pt:
                        result.iloc[i] = 1
                        break
                    elif ret <= sl:
                        result.iloc[i] = -1
                        break
            return result

        for ticker, grp in close.groupby(level="Ticker"):
            grp_result = _label_ticker(grp.droplevel("Ticker"))
            labels.loc[grp.index] = grp_result.values

        return labels

    @staticmethod
    def _meta_label(panel: pd.DataFrame) -> pd.Series:
        """
        Meta-label: 1 if the primary forward return is in the top/bottom 30%
        cross-sectionally (high conviction signal), else 0.
        """
        fwd = panel.get("fwd_ret_primary")
        if fwd is None:
            return pd.Series(1, index=panel.index)

        top_30 = fwd.groupby(level="Date").transform(lambda x: x.quantile(0.70))
        bot_30 = fwd.groupby(level="Date").transform(lambda x: x.quantile(0.30))
        meta = ((fwd >= top_30) | (fwd <= bot_30)).astype(int)
        return meta
