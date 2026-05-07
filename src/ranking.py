"""
ranking.py – Rank stocks by ensemble score and select top-N.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


class StockRanker:
    """
    Converts ensemble scores into a ranked list of stocks per date.
    """

    def __init__(self, cfg: dict) -> None:
        rc = cfg["ranking"]
        self.top_n: int = rc.get("top_n", 10)
        self.score_method: str = rc.get("score_method", "probability")
        self.min_score_threshold: float = rc.get("min_score_threshold", 0.50)

    # ── Public API ─────────────────────────────────────────────────────────

    def rank(
        self,
        scores: pd.Series,             # index = (Date, Ticker)
        return_dates: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Produce a ranked DataFrame with columns [rank, score, selected].

        Parameters
        ----------
        scores : pd.Series with MultiIndex (Date, Ticker)
        return_dates : subset of dates to rank (for walk-forward)

        Returns
        -------
        pd.DataFrame with columns [score, rank, percentile, selected]
        """
        result = scores.to_frame("score")

        # Normalise score to percentile within each date
        result["percentile"] = result.groupby(level="Date")["score"].rank(pct=True)

        # Rank within each date (1 = best)
        result["rank"] = result.groupby(level="Date")["score"].rank(
            ascending=False, method="first"
        )

        # Select top-N per date with minimum threshold
        result["selected"] = (
            (result["rank"] <= self.top_n)
            & (result["percentile"] >= self.min_score_threshold)
        ).astype(int)

        n_selected = result["selected"].sum()
        n_dates = result.index.get_level_values("Date").nunique()
        log.info(
            f"Ranked {result.shape[0]} (Date,Ticker) pairs across {n_dates} dates"
            f" | {n_selected} selected ({n_selected/n_dates:.1f}/date avg)"
        )
        return result

    def top_stocks_at_date(
        self,
        scores: pd.Series,
        date: pd.Timestamp,
    ) -> pd.Series:
        """
        Return the top-N ticker scores for a specific date.
        """
        try:
            day_scores = scores.loc[date]
        except KeyError:
            # Nearest date
            dates = scores.index.get_level_values("Date").unique()
            nearest = dates[dates <= date][-1]
            day_scores = scores.loc[nearest]
            log.debug(f"Date {date} not found; using nearest {nearest}")

        top = (
            day_scores
            .sort_values(ascending=False)
            .head(self.top_n)
        )
        top = top[top >= self.min_score_threshold]
        return top

    def latest_top_stocks(self, scores: pd.Series) -> pd.DataFrame:
        """
        Return top-N stocks based on the most recent date in scores.
        """
        latest_date = scores.index.get_level_values("Date").max()
        top = self.top_stocks_at_date(scores, latest_date)
        result = top.reset_index()
        result.columns = ["Ticker", "Score"]
        result["Rank"] = range(1, len(result) + 1)
        result["Date"] = latest_date
        log.info(f"\nTop {len(result)} stocks as of {latest_date.date()}:\n{result.to_string(index=False)}")
        return result

    def score_summary(self, ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ranking stats across all dates.
        """
        summary = (
            ranked.groupby(level="Ticker")
            .agg(
                avg_score=("score", "mean"),
                avg_rank=("rank", "mean"),
                times_selected=("selected", "sum"),
                times_ranked=("rank", "count"),
            )
        )
        summary["selection_rate"] = summary["times_selected"] / summary["times_ranked"]
        summary = summary.sort_values("avg_score", ascending=False)
        return summary
