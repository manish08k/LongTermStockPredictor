"""
src/utils/ticker_validator.py – Validates NSE/BSE tickers before download.

Provides:
  - Known invalid ticker removal
  - Symbol correction (renamed tickers)
  - Live validation via yfinance
  - Batch validation with parallel processing
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Dict, List, Optional, Set, Tuple

import yfinance as yf

from src.utils import get_logger

log = get_logger(__name__)

# ── Known corrections: old_symbol -> new_symbol ───────────────────────────
SYMBOL_CORRECTIONS: Dict[str, str] = {
    # Corporate actions / rebranding
    "LTIM.NS":           "LTIMINDTREE.NS",
    "MINDTREE.NS":       "LTIMINDTREE.NS",
    "HDFC.NS":           "HDFCBANK.NS",        # merged into HDFCBANK
    "HDFCAMC.NS":        "HDFCAMC.NS",          # still valid, no change
    "L&TFH.NS":          "L&TFH.NS",
    "INFRATEL.NS":       "INDUSINDBK.NS",        # placeholder – verify
    "ZEEL.NS":           "ZEEL.NS",
    "PCJEWELLER.NS":     None,                   # delisted
    "YESBANK.NS":        "YESBANK.NS",
    "RCOM.NS":           None,                   # delisted
    "JETAIRWAYS.NS":     None,                   # delisted / suspended
    "DHFL.NS":           None,                   # delisted
    "IL&FSTRANS.NS":     None,                   # delisted
    "SUZLON.NS":         "SUZLON.NS",
    # Non-NSE tickers (US stocks mistakenly added)
    "AAPL.NS":           None,
    "MSFT.NS":           None,
    "GOOGL.NS":          None,
    "AMZN.NS":           None,
    "TSLA.NS":           None,
    "NVDA.NS":           None,
    "META.NS":           None,
}

# ── Definitively known invalid / non-tradeable tickers ────────────────────
KNOWN_INVALID: Set[str] = {
    t for t, v in SYMBOL_CORRECTIONS.items() if v is None
}


class TickerValidator:
    """
    Validates and corrects NSE/BSE ticker symbols.

    Usage
    -----
    validator = TickerValidator()
    clean_tickers = validator.validate(raw_tickers)
    """

    def __init__(
        self,
        live_check: bool = True,
        max_workers: int = 10,
        timeout: float = 8.0,
        min_history_rows: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        live_check      : Run yfinance download check for unknown tickers
        max_workers     : Parallel workers for live validation
        timeout         : Per-ticker download timeout in seconds
        min_history_rows: Minimum rows to consider a ticker valid
        """
        self.live_check = live_check
        self.max_workers = max_workers
        self.timeout = timeout
        self.min_history_rows = min_history_rows
        self._valid_cache: Set[str] = set()
        self._invalid_cache: Set[str] = set()

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, tickers: List[str]) -> List[str]:
        """
        Returns cleaned, validated list of tickers.

        Steps:
          1. Apply known corrections / removals
          2. Deduplicate
          3. Live-validate remaining (optional)
        """
        corrected = self._apply_corrections(tickers)
        deduped = list(dict.fromkeys(corrected))         # preserve order

        if self.live_check:
            deduped = self._live_validate(deduped)

        log.info(
            f"Ticker validation: {len(tickers)} in → {len(deduped)} valid out"
        )
        return deduped

    def correct(self, ticker: str) -> Optional[str]:
        """
        Return corrected ticker, or None if ticker is definitively invalid.
        """
        if ticker in SYMBOL_CORRECTIONS:
            corrected = SYMBOL_CORRECTIONS[ticker]
            if corrected is None:
                log.debug(f"Ticker {ticker} is delisted/invalid – removed")
            elif corrected != ticker:
                log.info(f"Ticker correction: {ticker} → {corrected}")
            return corrected
        return ticker

    # ── Correction pass ────────────────────────────────────────────────────

    def _apply_corrections(self, tickers: List[str]) -> List[str]:
        """Apply all symbol corrections and filter out None entries."""
        result: List[str] = []
        for t in tickers:
            corrected = self.correct(t)
            if corrected is not None:
                result.append(corrected)
        return result

    # ── Live validation ────────────────────────────────────────────────────

    def _live_validate(self, tickers: List[str]) -> List[str]:
        """
        Validate tickers via yfinance in parallel.
        Uses cache to avoid re-checking already validated tickers.
        """
        to_check = [
            t for t in tickers
            if t not in self._valid_cache and t not in self._invalid_cache
        ]

        if to_check:
            log.info(f"Live-validating {len(to_check)} tickers …")
            results = self._batch_live_check(to_check)
            for tkr, is_valid in results.items():
                if is_valid:
                    self._valid_cache.add(tkr)
                else:
                    self._invalid_cache.add(tkr)
                    log.warning(f"Live check failed: {tkr} – will be excluded")

        valid = [t for t in tickers if t in self._valid_cache]
        return valid

    def _batch_live_check(self, tickers: List[str]) -> Dict[str, bool]:
        """Parallel yfinance check for a batch of tickers."""
        results: Dict[str, bool] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_map = {
                executor.submit(self._check_single, t): t for t in tickers
            }
            for future in concurrent.futures.as_completed(future_map):
                tkr = future_map[future]
                try:
                    results[tkr] = future.result(timeout=self.timeout)
                except Exception as exc:
                    log.debug(f"Validation timeout/error for {tkr}: {exc}")
                    results[tkr] = False
        return results

    def _check_single(self, ticker: str, retries: int = 2) -> bool:
        """
        Check if a single ticker downloads ≥ min_history_rows of data.
        Returns True if valid, False otherwise.
        """
        for attempt in range(retries):
            try:
                df = yf.download(
                    ticker,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if df is not None and len(df) >= self.min_history_rows:
                    return True
                return False
            except Exception as e:
                log.debug(f"{ticker} attempt {attempt + 1} error: {e}")
                if attempt < retries - 1:
                    time.sleep(1.0)
        return False


# ── Convenience function ───────────────────────────────────────────────────

def validate_tickers(
    tickers: List[str],
    live_check: bool = True,
    max_workers: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Validate a list of tickers and return (valid, invalid) tuple.

    Parameters
    ----------
    tickers     : Raw ticker list to validate
    live_check  : Run yfinance live check (slower but thorough)
    max_workers : Parallel download workers

    Returns
    -------
    (valid_tickers, invalid_tickers)
    """
    validator = TickerValidator(live_check=live_check, max_workers=max_workers)
    valid = validator.validate(tickers)
    invalid = [t for t in tickers if t not in valid]
    return valid, invalid
