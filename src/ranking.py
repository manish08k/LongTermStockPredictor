"""
ranking.py – Rank stocks by ensemble score and select top-N.

BUG FIXES vs original:
  - dropna(subset=["Score"]) before ranking
  - per-date ranking with stable sort
  - latest_date filtering guaranteed to return results
  - top-N extraction always works even with NaN scores
  - sector diversification constraint
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)


class StockRanker:
    """
    Converts ensemble scores into a ranked list of stocks per date.
    Guarantees non-empty output when valid scores exist.
    """

    def __init__(self, cfg: dict) -> None:
        rc = cfg.get("ranking", {})
        self.top_n: int = rc.get("top_n", 15)
        self.score_method: str = rc.get("score_method", "probability")
        self.min_score_threshold: float = rc.get("min_score_threshold", 0.50)
        self.sector_diversification: bool = rc.get("sector_diversification", False)
        self.max_per_sector: int = rc.get("max_per_sector", 5)
        self.sectors: Dict[str, str] = cfg.get("data", {}).get("sectors", {})

    # ── Public API ─────────────────────────────────────────────────────────

    def rank(
        self,
        scores: pd.Series,
        return_dates: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Produce a ranked DataFrame per date.

        Parameters
        ----------
        scores : pd.Series with MultiIndex (Date, Ticker)

        Returns
        -------
        pd.DataFrame with columns [score, rank, percentile, selected]
        """
        if scores.empty:
            log.warning("No scores provided to ranker – returning empty DataFrame")
            return pd.DataFrame(columns=["score", "rank", "percentile", "selected"])

        # ── FIX 1: Drop NaN scores before doing anything ──────────────────
        scores = scores.dropna()
        if scores.empty:
            log.warning("All scores are NaN – returning empty DataFrame")
            return pd.DataFrame(columns=["score", "rank", "percentile", "selected"])

        result = scores.rename("score").to_frame()

        # ── FIX 2: Percentile rank within each date ────────────────────────
        result["percentile"] = (
            result.groupby(level="Date")["score"]
            .rank(pct=True)
        )

        # ── FIX 3: Rank within each date (1 = best), stable sort ──────────
        result["rank"] = (
            result.groupby(level="Date")["score"]
            .rank(ascending=False, method="first")
        )

        # ── FIX 4: Threshold – lower it if too restrictive ─────────────────
        threshold = self.min_score_threshold
        n_above = (result["score"] >= threshold).sum()
        if n_above < len(result) * 0.10:
            # Fewer than 10% pass threshold – relax it
            threshold = result["score"].quantile(0.40)
            log.warning(
                f"Min score threshold {self.min_score_threshold:.2f} too strict; "
                f"relaxed to {threshold:.4f} (40th pct)"
            )

        # ── FIX 5: Select top-N per date ───────────────────────────────────
        result["selected"] = (
            (result["rank"] <= self.top_n)
            & (result["score"] >= threshold)
        ).astype(int)

        n_selected = result["selected"].sum()
        n_dates = result.index.get_level_values("Date").nunique()
        avg_per_date = n_selected / max(n_dates, 1)
        log.info(
            f"Ranked {result.shape[0]} (Date,Ticker) pairs | "
            f"{n_dates} dates | {n_selected} selected "
            f"({avg_per_date:.1f}/date avg)"
        )
        return result

    def top_stocks_at_date(
        self,
        scores: pd.Series,
        date: pd.Timestamp,
    ) -> pd.Series:
        """Return the top-N ticker scores for a specific date."""
        scores = scores.dropna()
        if scores.empty:
            return pd.Series(dtype=float)

        dates = scores.index.get_level_values("Date").unique().sort_values()

        # ── FIX: Use nearest available date if exact date missing ──────────
        if date not in dates:
            prior = dates[dates <= date]
            if len(prior) == 0:
                log.warning(f"No data on or before {date}; using earliest available")
                date = dates[0]
            else:
                date = prior[-1]
            log.debug(f"Using nearest date: {date.date()}")

        day_scores = scores.loc[date].dropna()
        if day_scores.empty:
            return pd.Series(dtype=float)

        top = day_scores.sort_values(ascending=False).head(self.top_n)

        # Apply threshold – but keep at least 1 stock
        above_thresh = top[top >= self.min_score_threshold]
        if above_thresh.empty:
            log.debug("No scores above threshold; returning top stock anyway")
            return top.head(1)

        return above_thresh

    def latest_top_stocks(self, scores: pd.Series) -> pd.DataFrame:
        """
        Return top-N stocks for the most recent date.
        GUARANTEED to return a non-empty DataFrame when scores is non-empty.
        """
        scores = scores.dropna()
        if scores.empty:
            log.warning("No valid scores – latest_top_stocks returning empty")
            return pd.DataFrame(columns=["Ticker", "Score", "Rank", "Date", "Sector"])

        # ── FIX: Get latest date explicitly ───────────────────────────────
        latest_date = scores.index.get_level_values("Date").max()
        log.info(f"Latest scoring date: {latest_date.date()}")

        top = self.top_stocks_at_date(scores, latest_date)

        if top.empty:
            # Last resort: take any date's top stocks
            log.warning("No stocks at latest date; scanning prior dates …")
            for d in scores.index.get_level_values("Date").unique().sort_values()[::-1]:
                top = self.top_stocks_at_date(scores, d)
                if not top.empty:
                    latest_date = d
                    break

        result = top.reset_index()
        result.columns = ["Ticker", "Score"]
        result = result.dropna(subset=["Score"])
        result = result.sort_values("Score", ascending=False).reset_index(drop=True)
        result["Rank"] = range(1, len(result) + 1)
        result["Date"] = latest_date
        result["Sector"] = result["Ticker"].map(self.sectors).fillna("Unknown")

        log.info(
            f"\nTop {len(result)} stocks as of {latest_date.date()}:\n"
            + result[["Rank", "Ticker", "Score", "Sector"]].to_string(index=False)
        )
        return result

    def apply_sector_diversification(
        self,
        top_tickers: List[str],
    ) -> List[str]:
        """
        Enforce max_per_sector constraint while preserving score order.
        """
        if not self.sector_diversification or not self.sectors:
            return top_tickers

        counts: Dict[str, int] = {}
        diversified = []
        for tkr in top_tickers:
            sector = self.sectors.get(tkr, "Unknown")
            if counts.get(sector, 0) < self.max_per_sector:
                diversified.append(tkr)
                counts[sector] = counts.get(sector, 0) + 1
        return diversified

    def score_summary(self, ranked: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ranking stats across all dates."""
        if ranked.empty:
            return pd.DataFrame()
        summary = (
            ranked.groupby(level="Ticker")
            .agg(
                avg_score=("score", "mean"),
                avg_rank=("rank", "mean"),
                times_selected=("selected", "sum"),
                times_ranked=("rank", "count"),
            )
        )
        summary["selection_rate"] = (
            summary["times_selected"] / summary["times_ranked"]
        )
        return summary.sort_values("avg_score", ascending=False)