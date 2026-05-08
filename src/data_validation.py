"""
data_validation.py – Data quality checks, outlier detection, alignment.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, winsorize

log = get_logger(__name__)


class DataValidator:
    def __init__(self, cfg: dict) -> None:
        dc = cfg.get("data", {})
        self.min_history: int = dc.get("min_history_days", 252)
        self.benchmark: str = dc.get("benchmark_ticker", "^NSEI")

    def validate(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        cleaned = {}
        report: List[dict] = []
        for tkr, df in raw_data.items():
            result, issues = self._validate_ticker(tkr, df)
            if result is not None:
                cleaned[tkr] = result
            report.append({"ticker": tkr, "ok": result is not None, "issues": issues})
        self._log_report(report)
        return cleaned

    def align_panel(
        self, data: Dict[str, pd.DataFrame], fill_method: str = "ffill"
    ) -> Dict[str, pd.DataFrame]:
        """Align all tickers to a common date index."""
        close_dict = {tkr: df["Close"] for tkr, df in data.items()}
        close_df = pd.DataFrame(close_dict).ffill().dropna(how="all")
        common_dates = close_df.index
        aligned = {}
        for tkr, df in data.items():
            aligned[tkr] = df.reindex(common_dates, method="ffill")
        log.info(f"Aligned: {len(common_dates)} dates × {len(aligned)} tickers")
        return aligned

    def _validate_ticker(
        self, ticker: str, df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        issues: List[str] = []
        df = df.copy()

        if len(df) < self.min_history:
            issues.append(f"Insufficient history: {len(df)} < {self.min_history}")
            return None, issues

        null_pct = df["Close"].isna().mean()
        if null_pct > 0.05:
            issues.append(f"High null rate: {null_pct:.1%}")
            return None, issues

        df["Close"] = df["Close"].ffill().bfill()
        ret = df["Close"].pct_change()

        # Detect stale prices
        zero_streak = (ret == 0).rolling(5).sum()
        stale_pct = (zero_streak >= 5).mean()
        if stale_pct > 0.10:
            issues.append(f"Stale prices: {stale_pct:.1%}")

        # Winsorize extreme returns
        extreme = ret.abs() > 1.0
        if extreme.sum() > 0:
            issues.append(f"Extreme returns clipped: {extreme.sum()} rows")
            df.loc[extreme, "Close"] = np.nan
            df["Close"] = df["Close"].ffill()

        # Volume check
        if "Volume" in df.columns:
            zero_vol = (df["Volume"] == 0).mean()
            if zero_vol > 0.25:
                issues.append(f"High zero-vol rate: {zero_vol:.1%}")

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