"""
data_validation.py – Data quality checks, outlier detection, and alignment.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, winsorize

log = get_logger(__name__)


class DataValidator:
    """
    Validates and cleans the multi-ticker price panel.
    """

    def __init__(self, cfg: dict) -> None:
        dc = cfg["data"]
        self.min_history = dc.get("min_history_days", 252)
        self.tickers = dc["tickers"]
        self.benchmark = dc.get("benchmark_ticker", dc.get("benchmark", "^NSEI"))

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Run all validation checks and return cleaned data dict.
        """
        cleaned = {}
        report: List[dict] = []

        for tkr, df in raw_data.items():
            result, issues = self._validate_ticker(tkr, df)
            if result is not None:
                cleaned[tkr] = result
            report.append({"ticker": tkr, "rows": len(df), "issues": issues, "ok": result is not None})

        self._log_report(report)
        return cleaned

    def align_panel(
        self,
        data: Dict[str, pd.DataFrame],
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Align all tickers to a common date index (union of dates).
        Returns a multi-level DataFrame (Date × Ticker, columns=OHLCV).
        """
        # Use Close price to determine common dates
        close_dict = {tkr: df["Close"] for tkr, df in data.items()}
        close_df = pd.DataFrame(close_dict)

        if fill_method == "ffill":
            close_df = close_df.ffill().dropna(how="all")
        elif fill_method == "drop":
            close_df = close_df.dropna(how="any")

        common_dates = close_df.index

        aligned = {}
        for tkr, df in data.items():
            df_aligned = df.reindex(common_dates, method="ffill")
            aligned[tkr] = df_aligned

        log.info(f"Aligned panel: {len(common_dates)} dates × {len(aligned)} tickers")
        return aligned

    # ── Private helpers ────────────────────────────────────────────────────

    def _validate_ticker(
        self, ticker: str, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame | None, List[str]]:
        issues: List[str] = []
        df = df.copy()

        # 1. Minimum history
        if len(df) < self.min_history:
            issues.append(f"Insufficient history: {len(df)} < {self.min_history}")
            return None, issues

        # 2. Missing values in Close
        null_pct = df["Close"].isna().mean()
        if null_pct > 0.05:
            issues.append(f"High null rate in Close: {null_pct:.1%}")
            return None, issues
        df["Close"] = df["Close"].ffill().bfill()

        # 3. Stale prices (no change for >5 consecutive days)
        returns = df["Close"].pct_change()
        zero_streak = (returns == 0).rolling(5).sum()
        stale_pct = (zero_streak >= 5).mean()
        if stale_pct > 0.10:
            issues.append(f"Stale price streaks: {stale_pct:.1%}")

        # 4. Extreme returns (>100% daily) – likely data errors
        extreme_mask = returns.abs() > 1.0
        n_extreme = extreme_mask.sum()
        if n_extreme > 0:
            issues.append(f"Extreme daily returns (>100%): {n_extreme} rows → winsorized")
            df.loc[extreme_mask, "Close"] = np.nan
            df["Close"] = df["Close"].ffill()

        # 5. Volume check
        if "Volume" in df.columns:
            zero_vol = (df["Volume"] == 0).mean()
            if zero_vol > 0.20:
                issues.append(f"High zero-volume rate: {zero_vol:.1%}")

        # 6. Fill remaining OHLCV
        for col in ["Open", "High", "Low", "Volume"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        return df, issues

    @staticmethod
    def _log_report(report: List[dict]) -> None:
        ok = [r for r in report if r["ok"]]
        bad = [r for r in report if not r["ok"]]
        log.info(f"Validation: {len(ok)} passed / {len(bad)} failed")
        for r in bad:
            log.warning(f"  FAILED {r['ticker']}: {r['issues']}")
        for r in ok:
            if r["issues"]:
                log.debug(f"  WARN {r['ticker']}: {r['issues']}")
