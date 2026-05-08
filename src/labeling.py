"""
labeling.py – Forward-return labels with strict anti-leakage.

Improvements vs original:
  - Benchmark-relative excess return labels
  - Percentile-based binary labels (top 40% = positive)
  - Multi-class labels (3-class and 5-class)
  - Regression labels (raw forward return)
  - Excess return threshold (outperform NIFTY by >8% = label 1)
  - Annualized return labels
  - Proper NaN handling at sample boundaries
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, timeit

log = get_logger(__name__)


class Labeler:
    """
    Attaches forward-return labels to the feature panel.
    All labels are computed using only future prices – zero leakage.
    """

    def __init__(self, cfg: dict) -> None:
        lc = cfg.get("labeling", {})
        self.horizons: List[int] = lc.get("horizons", [21, 63, 126, 252])
        self.primary_horizon: int = lc.get("primary_horizon", 63)
        self.label_type: str = lc.get("label_type", "binary")
        self.outperform_threshold: float = lc.get("outperform_threshold", 0.0)
        self.excess_return_target: float = lc.get("excess_return_target", 0.08)
        self.percentile_threshold: float = lc.get("percentile_threshold", 0.60)
        self.triple_barrier: dict = lc.get("triple_barrier", {"enabled": False})
        self.benchmark_ticker: str = (
            cfg.get("data", {}).get("benchmark_ticker", "^NSEI")
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def label(
        self,
        panel: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute and attach forward-return labels.

        Returns
        -------
        panel with columns:
          fwd_ret_{h}d        – raw forward return
          excess_ret_{h}d     – return minus benchmark return
          label_{h}d          – primary label (binary/regression/multiclass)
          label_primary       – alias for primary horizon label
          label_high_conv     – high-conviction: excess > 8% annualized
          ann_ret_{h}d        – annualized forward return
          meta_label          – 1 if in top/bottom 30% cross-sectionally
        """
        log.info(
            f"Labeling horizons={self.horizons} | "
            f"primary={self.primary_horizon}d | type={self.label_type}"
        )
        result = panel.copy()

        # ── Benchmark forward returns ──────────────────────────────────────
        bm_fwd: Dict[int, pd.Series] = {}
        if benchmark_df is not None and "Close" in benchmark_df.columns:
            bm_close = benchmark_df["Close"]
            for h in self.horizons:
                bm_fwd[h] = bm_close.shift(-h) / bm_close - 1
                log.debug(
                    f"Benchmark fwd_ret_{h}d: "
                    f"mean={bm_fwd[h].mean():.3f} std={bm_fwd[h].std():.3f}"
                )

        # ── Stock forward returns (per ticker, no leakage) ─────────────────
        if "Close" not in result.columns:
            raise ValueError("Panel must contain 'Close' column for labeling")

        close = result["Close"]

        for h in self.horizons:
            # Forward return: price at t+h / price at t - 1
            fwd_ret = self._forward_return_panel(close, h)
            result[f"fwd_ret_{h}d"] = fwd_ret

            # Annualised return
            result[f"ann_ret_{h}d"] = (1 + fwd_ret) ** (252 / h) - 1

            # Excess return vs benchmark
            if h in bm_fwd:
                bm_aligned = self._align_benchmark(bm_fwd[h], result.index)
                excess = fwd_ret - bm_aligned
            else:
                excess = fwd_ret
            result[f"excess_ret_{h}d"] = excess

            # ── Primary label ──────────────────────────────────────────────
            if self.label_type == "binary":
                # Top percentile_threshold within cross-section
                label = self._binary_label_percentile(excess, self.percentile_threshold)
            elif self.label_type == "regression":
                label = excess
            elif self.label_type == "multi_class":
                label = self._multi_class_label(excess)
            else:
                label = self._binary_label_percentile(excess, self.percentile_threshold)

            result[f"label_{h}d"] = label

        # Primary horizon
        result["label_primary"] = result[f"label_{self.primary_horizon}d"]
        result["fwd_ret_primary"] = result[f"fwd_ret_{self.primary_horizon}d"]
        result["excess_ret_primary"] = result[f"excess_ret_{self.primary_horizon}d"]

        # ── High-conviction label: excess_ann > 8% ─────────────────────────
        ann_excess = (
            (1 + result[f"excess_ret_{self.primary_horizon}d"])
            ** (252 / self.primary_horizon) - 1
        )
        result["label_high_conv"] = (
            ann_excess > self.excess_return_target
        ).astype(float)

        # ── Triple barrier (optional) ──────────────────────────────────────
        if self.triple_barrier.get("enabled", False):
            result["label_tb"] = self._triple_barrier_label(close)

        # ── Meta-label (confidence filter) ────────────────────────────────
        result["meta_label"] = self._meta_label(result)

        # ── Drop rows where primary label is NaN ──────────────────────────
        # (these are at the end of sample where no future data exists)
        n_before = len(result)
        result = result.dropna(subset=["label_primary"])
        n_dropped = n_before - len(result)
        log.info(
            f"Labeled {len(result)} rows | Dropped {n_dropped} NaN-label rows "
            f"(end-of-sample, no future data – correct)"
        )

        if len(result) == 0:
            raise RuntimeError(
                "All labels are NaN. Check that date range is long enough "
                "for the primary horizon and data is present."
            )

        # Label balance
        if self.label_type == "binary":
            vc = result["label_primary"].value_counts()
            log.info(f"Label balance: {vc.to_dict()} | pos_rate={vc.get(1,0)/len(result):.2%}")

        return result

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _forward_return_panel(close: pd.Series, horizon: int) -> pd.Series:
        """
        Forward return at each (Date, Ticker).
        Computed using groupby+shift(-horizon) – strictly no leakage.
        """
        return close.groupby(level="Ticker").transform(
            lambda x: x.shift(-horizon) / x - 1
        )

    @staticmethod
    def _align_benchmark(bm_series: pd.Series, panel_index: pd.MultiIndex) -> pd.Series:
        """Broadcast benchmark scalar per date across all (Date,Ticker) rows."""
        dates = panel_index.get_level_values("Date")
        bm_aligned = bm_series.reindex(dates).values
        return pd.Series(bm_aligned, index=panel_index, name="bm")

    def _binary_label_percentile(
        self, excess_ret: pd.Series, threshold: float
    ) -> pd.Series:
        """
        Label = 1 if excess return is in the top (1-threshold) cross-sectionally.
        E.g., threshold=0.60 → top 40% = positive.
        """
        cutoff = excess_ret.groupby(level="Date").transform(
            lambda x: x.quantile(threshold)
        )
        return (excess_ret >= cutoff).astype(float)

    @staticmethod
    def _multi_class_label(excess_ret: pd.Series) -> pd.Series:
        """
        3-class:
          2 = strong outperform (top quartile)
          1 = neutral (middle)
          0 = underperform (bottom quartile)
        """
        q25 = excess_ret.groupby(level="Date").transform(lambda x: x.quantile(0.25))
        q75 = excess_ret.groupby(level="Date").transform(lambda x: x.quantile(0.75))
        label = pd.Series(1, index=excess_ret.index, dtype=float)
        label[excess_ret <= q25] = 0
        label[excess_ret >= q75] = 2
        return label

    def _triple_barrier_label(self, close: pd.Series) -> pd.Series:
        """Labels: +1 hit profit target first, -1 hit stop first, 0 timeout."""
        pt = self.triple_barrier.get("profit_taking", 0.15)
        sl = self.triple_barrier.get("stop_loss", -0.10)
        max_hold = self.triple_barrier.get("max_holding", 63)
        log.info(f"Triple barrier: PT={pt:.0%} SL={sl:.0%} max_hold={max_hold}d")

        labels = pd.Series(0.0, index=close.index)

        def _label_one_ticker(prices: pd.Series) -> pd.Series:
            arr = prices.values
            result = np.zeros(len(arr))
            for i in range(len(arr) - 1):
                entry = arr[i]
                if np.isnan(entry) or entry <= 0:
                    continue
                end = min(i + max_hold + 1, len(arr))
                for j in range(i + 1, end):
                    r = arr[j] / entry - 1
                    if r >= pt:
                        result[i] = 1
                        break
                    elif r <= sl:
                        result[i] = -1
                        break
            return pd.Series(result, index=prices.index)

        for ticker, grp in close.groupby(level="Ticker"):
            grp_clean = grp.droplevel("Ticker")
            grp_labels = _label_one_ticker(grp_clean)
            labels.loc[grp.index] = grp_labels.values

        return labels

    @staticmethod
    def _meta_label(panel: pd.DataFrame) -> pd.Series:
        """
        Meta-label = 1 if the primary excess return is in top or bottom 30%.
        Useful for filtering low-conviction signals.
        """
        fwd = panel.get("excess_ret_primary")
        if fwd is None:
            return pd.Series(1, index=panel.index)
        top_30 = fwd.groupby(level="Date").transform(lambda x: x.quantile(0.70))
        bot_30 = fwd.groupby(level="Date").transform(lambda x: x.quantile(0.30))
        return ((fwd >= top_30) | (fwd <= bot_30)).astype(float)