"""
data_ingestion.py – Fetches OHLCV + benchmark data via yfinance,
caches results, and returns a clean multi-ticker panel.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils import ensure_dir, get_logger, load_parquet, save_parquet, timeit

log = get_logger(__name__)


class DataIngestion:
    """
    Downloads and caches price data for a list of tickers.

    Attributes
    ----------
    tickers : list[str]
    benchmark : str
    start_date : str
    end_date : str
    interval : str
    cache : bool
    raw_dir : Path
    """

    def __init__(self, cfg: dict) -> None:
        dc = cfg["data"]
        self.tickers: List[str] = dc["tickers"]
        self.benchmark: str = dc.get("benchmark_ticker", dc.get("benchmark", "^NSEI"))
        self.start_date: str = dc["start_date"]
        self.end_date: str = dc["end_date"]
        self.interval: str = dc.get("interval", "1d")
        self.cache: bool = dc.get("cache_data", True)
        self.raw_dir: Path = ensure_dir(dc.get("raw_data_path", "data/raw"))
        self.min_history: int = dc.get("min_history_days", 252)

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Returns
        -------
        dict[ticker -> OHLCV DataFrame]
        """
        all_tickers = self.tickers + [self.benchmark]
        data: Dict[str, pd.DataFrame] = {}
        for tkr in all_tickers:
            df = self._fetch_ticker(tkr)
            if df is not None and len(df) >= self.min_history:
                data[tkr] = df
            else:
                log.warning(f"Skipping {tkr}: insufficient history ({len(df) if df is not None else 0} rows)")
        log.info(f"Fetched {len(data)} tickers (including benchmark)")
        return data

    def fetch_panel(self) -> pd.DataFrame:
        """
        Returns a stacked panel DataFrame with columns:
        [Open, High, Low, Close, Volume, Adj Close] and index (Date, Ticker).
        """
        raw = self.fetch_all()
        frames = []
        for tkr, df in raw.items():
            df = df.copy()
            df["Ticker"] = tkr
            frames.append(df)
        if not frames:
            raise ValueError("No data fetched. Check tickers and date range.")
        panel = pd.concat(frames)
        panel.index.name = "Date"
        panel = panel.reset_index().set_index(["Date", "Ticker"])
        log.info(f"Panel shape: {panel.shape}")
        return panel

    # ── Private helpers ────────────────────────────────────────────────────

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("^", "IDX_").replace(".", "_")
        return self.raw_dir / f"{safe}_{self.start_date}_{self.end_date}.parquet"

    def _fetch_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        cache_path = self._cache_path(ticker)
        if self.cache and cache_path.exists():
            log.debug(f"Loading {ticker} from cache")
            return load_parquet(cache_path)

        log.info(f"Downloading {ticker} ...")
        try:
            tkr_obj = yf.Ticker(ticker)
            df = tkr_obj.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=True,
            )
            if df.empty:
                log.warning(f"{ticker}: empty download result")
                return None
            df = self._clean_ohlcv(df, ticker)
            if self.cache:
                save_parquet(df, cache_path)
            return df
        except Exception as exc:
            log.error(f"Failed to download {ticker}: {exc}")
            return None

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardise column names and basic cleaning."""
        df = df.copy()
        # yfinance returns timezone-aware index; strip tz
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        # Keep standard columns
        keep = ["Open", "High", "Low", "Close", "Volume"]
        for col in keep:
            if col not in df.columns:
                # Try case-insensitive match
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df.rename(columns={matches[0]: col}, inplace=True)
        df = df[[c for c in keep if c in df.columns]]

        # Remove rows with all NaN prices
        df = df.dropna(subset=["Close"])

        # Forward-fill minor gaps (≤3 trading days)
        df = df.ffill(limit=3)

        # Remove zero/negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        df = df.sort_index()
        return df
