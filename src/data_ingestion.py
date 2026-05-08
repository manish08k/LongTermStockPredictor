"""
data_ingestion.py – Fetches OHLCV + benchmark data via yfinance.
Handles NSE/BSE tickers, caching, retries, and async batch downloads.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils import ensure_dir, get_logger, load_parquet, save_parquet, timeit

log = get_logger(__name__)

# ── Indian stock universe definitions ─────────────────────────────────────────

NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "NESTLEIND.NS", "BAJAJFINSV.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "BPCL.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "APOLLOHOSP.NS", "TATACONSUM.NS", "GRASIM.NS", "M&M.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "UPL.NS", "INDUSINDBK.NS", "VEDL.NS",
]

NIFTY_NEXT50 = [
    "PIDILITIND.NS", "HAVELLS.NS", "SIEMENS.NS", "ICICIGI.NS", "COLPAL.NS",
    "MARICO.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "LUPIN.NS", "TORNTPHARM.NS",
    "DABUR.NS", "INDHOTEL.NS", "GODREJCP.NS", "AMBUJACEM.NS", "ACC.NS",
    "BALKRISIND.NS", "COROMANDEL.NS", "VOLTAS.NS", "TRENT.NS", "DMART.NS",
    "CHOLAFIN.NS", "AUBANK.NS", "FEDERALBNK.NS", "IPCALAB.NS", "ALKEM.NS",
]

MIDCAP_SELECTED = [
    "PERSISTENT.NS", "LTIM.NS", "MPHASIS.NS", "COFORGE.NS",
    "METROPOLIS.NS", "SOLARINDS.NS", "LALPATHLAB.NS", "DEEPAKNTR.NS",
    "ABCAPITAL.NS", "MAXHEALTH.NS", "POLICYBZR.NS", "NAUKRI.NS",
]

NSE_FULL_UNIVERSE = list(dict.fromkeys(NIFTY50 + NIFTY_NEXT50 + MIDCAP_SELECTED))


class DataIngestion:
    """
    Downloads and caches OHLCV data for NSE/BSE tickers.
    Uses yfinance batch downloads for efficiency with retry logic.
    """

    def __init__(self, cfg: dict) -> None:
        dc = cfg["data"]
        # Use tickers from config if provided, else fall back to NSE universe
        self.tickers: List[str] = dc.get("tickers") or NSE_FULL_UNIVERSE
        # Remove benchmark from stock list
        self.benchmark: str = dc.get("benchmark_ticker", "^NSEI")
        self.tickers = [t for t in self.tickers if t != self.benchmark]

        self.start_date: str = dc.get("start_date", "2015-01-01")
        self.end_date: str = dc.get("end_date", "2024-12-31")
        self.interval: str = dc.get("interval", "1d")
        self.cache: bool = dc.get("cache_data", True)
        self.raw_dir: Path = ensure_dir(dc.get("raw_data_path", "data/raw"))
        self.min_history: int = dc.get("min_history_days", 252)
        self._retry_delay: float = 2.0
        self._max_retries: int = 3

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Returns dict[ticker -> OHLCV DataFrame] for stocks + benchmark.
        Uses batch download for speed, then validates each.
        """
        all_tickers = self.tickers + [self.benchmark]
        data: Dict[str, pd.DataFrame] = {}

        # Try batch download first (much faster)
        batch_data = self._batch_download(all_tickers)
        data.update(batch_data)

        # Fall back to individual downloads for any missing
        missing = [t for t in all_tickers if t not in data]
        if missing:
            log.info(f"Fetching {len(missing)} tickers individually …")
            for tkr in missing:
                df = self._fetch_ticker_with_retry(tkr)
                if df is not None and len(df) >= self.min_history:
                    data[tkr] = df

        log.info(f"Fetched {len(data)} tickers total (including benchmark)")
        return data

    def fetch_panel(self) -> pd.DataFrame:
        """Returns stacked panel with MultiIndex (Date, Ticker)."""
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
        return panel

    # ── Private helpers ────────────────────────────────────────────────────

    def _batch_download(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Use yfinance batch download for efficiency."""
        # Check which are cached
        cached = {}
        to_download = []
        for tkr in tickers:
            cp = self._cache_path(tkr)
            if self.cache and cp.exists():
                try:
                    df = load_parquet(cp)
                    if len(df) >= self.min_history:
                        cached[tkr] = df
                        continue
                except Exception:
                    pass
            to_download.append(tkr)

        log.info(f"Cache hits: {len(cached)} | To download: {len(to_download)}")
        if not to_download:
            return cached

        # Batch download in chunks of 50
        downloaded = {}
        chunk_size = 50
        for i in range(0, len(to_download), chunk_size):
            chunk = to_download[i: i + chunk_size]
            try:
                raw = yf.download(
                    tickers=chunk,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if raw.empty:
                    continue

                for tkr in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw.copy()
                        else:
                            df = raw[tkr].copy()
                        df = self._clean_ohlcv(df, tkr)
                        if len(df) >= self.min_history:
                            downloaded[tkr] = df
                            if self.cache:
                                save_parquet(df, self._cache_path(tkr))
                        else:
                            log.debug(f"Skipping {tkr}: only {len(df)} rows")
                    except Exception as e:
                        log.debug(f"Parse failed for {tkr}: {e}")

            except Exception as e:
                log.warning(f"Batch download failed for chunk {i}: {e}")
                time.sleep(self._retry_delay)

        result = {**cached, **downloaded}
        return result

    def _fetch_ticker_with_retry(self, ticker: str) -> Optional[pd.DataFrame]:
        cache_path = self._cache_path(ticker)
        if self.cache and cache_path.exists():
            try:
                df = load_parquet(cache_path)
                if len(df) >= self.min_history:
                    return df
            except Exception:
                pass

        for attempt in range(self._max_retries):
            try:
                tkr_obj = yf.Ticker(ticker)
                df = tkr_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    auto_adjust=True,
                )
                if df.empty:
                    return None
                df = self._clean_ohlcv(df, ticker)
                if self.cache and len(df) >= self.min_history:
                    save_parquet(df, cache_path)
                return df
            except Exception as e:
                log.warning(f"{ticker} attempt {attempt+1} failed: {e}")
                time.sleep(self._retry_delay * (attempt + 1))

        log.error(f"Failed to download {ticker} after {self._max_retries} attempts")
        return None

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("^", "IDX_").replace(".", "_").replace("&", "AND")
        return self.raw_dir / f"{safe}_{self.start_date}_{self.end_date}.parquet"

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardise columns and clean data."""
        df = df.copy()
        # Strip timezone
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        # Normalise column names
        df.columns = [str(c).strip().title() for c in df.columns]
        rename_map = {c: c.title() for c in df.columns}
        df = df.rename(columns=rename_map)

        keep = ["Open", "High", "Low", "Close", "Volume"]
        # Try to find matching cols case-insensitively
        col_map = {c.lower(): c for c in df.columns}
        for want in keep:
            if want not in df.columns and want.lower() in col_map:
                df.rename(columns={col_map[want.lower()]: want}, inplace=True)

        df = df[[c for c in keep if c in df.columns]]

        # Remove rows where Close is missing/zero
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]

        # Forward fill minor gaps (≤3 days)
        df = df.ffill(limit=3)

        # Remove extreme single-day returns (>100%) – likely data errors
        if "Close" in df.columns:
            ret = df["Close"].pct_change().abs()
            df = df[ret <= 1.0]

        df = df.sort_index()
        return df