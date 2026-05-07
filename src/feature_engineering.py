"""
feature_engineering.py – Technical & statistical features for each ticker.
Produces a feature matrix indexed by (Date, Ticker).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import ta

from src.utils import get_logger, timeit, winsorize, zscore

log = get_logger(__name__)


class FeatureEngineer:
    """
    Generates a rich feature set from OHLCV data.
    All features are computed in-sample per ticker to avoid look-ahead bias.
    """

    def __init__(self, cfg: dict) -> None:
        fc = cfg["features"]
        self.momentum_windows: List[int] = fc.get("momentum_windows", [21,63,126,252])
        self.ma_windows: List[int] = fc.get("ma_windows", [20,50,100,200])
        self.vol_windows: List[int] = fc.get("vol_windows", [10,21,63])
        self.rsi_window: int = fc.get("rsi_window", fc.get("rsi_period", 14))
        self.macd_fast: int = fc.get("macd_fast", 12)
        self.macd_slow: int = fc.get("macd_slow", 26)
        self.macd_signal: int = fc.get("macd_signal", 9)
        self.bb_window: int = fc.get("bb_window", fc.get("bb_period", 20))
        self.bb_std: float = fc.get("bb_std", 2)
        self.atr_window: int = fc.get("atr_window", fc.get("atr_period", 14))
        self.lag_periods: List[int] = fc.get("lag_periods", [1,5,10,21])
        self.zscore_window: int = fc.get("zscore_window", 252)

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def build_features(
        self, aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compute features for every ticker and concatenate into a panel.

        Returns
        -------
        pd.DataFrame with MultiIndex (Date, Ticker)
        """
        frames = []
        for ticker, df in aligned_data.items():
            feat = self._compute_features(df, ticker)
            feat["Ticker"] = ticker
            frames.append(feat)

        panel = pd.concat(frames)
        panel = panel.reset_index().rename(columns={"index": "Date"})
        panel = panel.set_index(["Date", "Ticker"])
        panel = panel.sort_index()

        log.info(f"Feature panel: {panel.shape[0]} rows × {panel.shape[1]} features")
        return panel

    # ── Core feature computation ───────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = df["Close"]
        high = df["High"] if "High" in df.columns else close
        low = df["Low"] if "Low" in df.columns else close
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

        # 1. Returns
        feat["ret_1d"] = close.pct_change(1)
        feat["ret_5d"] = close.pct_change(5)

        for w in self.momentum_windows:
            feat[f"mom_{w}d"] = close.pct_change(w)
            # Skip-1 momentum (avoid microstructure reversal)
            feat[f"mom_{w}d_skip1"] = close.shift(1).pct_change(w)

        # 2. Moving averages & price/MA ratios
        for w in self.ma_windows:
            ma = close.rolling(w, min_periods=w // 2).mean()
            feat[f"ma_{w}"] = ma
            feat[f"price_ma_{w}_ratio"] = close / (ma + 1e-10)
            feat[f"ma_{w}_slope"] = ma.pct_change(5)

        # 3. MA crossovers
        if len(self.ma_windows) >= 2:
            s, l = self.ma_windows[0], self.ma_windows[1]
            ma_s = close.rolling(s).mean()
            ma_l = close.rolling(l).mean()
            feat[f"ma_cross_{s}_{l}"] = (ma_s - ma_l) / (ma_l + 1e-10)

        # 4. RSI
        feat["rsi"] = ta.momentum.RSIIndicator(close=close, window=self.rsi_window).rsi()
        feat["rsi_z"] = zscore(feat["rsi"], window=252)

        # 5. MACD
        macd_obj = ta.trend.MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal,
        )
        feat["macd"] = macd_obj.macd()
        feat["macd_signal"] = macd_obj.macd_signal()
        feat["macd_diff"] = macd_obj.macd_diff()
        feat["macd_norm"] = feat["macd"] / (close + 1e-10)

        # 6. Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=close, window=self.bb_window, window_dev=self.bb_std
        )
        feat["bb_pct"] = bb.bollinger_pband()
        feat["bb_width"] = bb.bollinger_wband()
        feat["bb_upper"] = bb.bollinger_hband()
        feat["bb_lower"] = bb.bollinger_lband()

        # 7. ATR & volatility
        atr_obj = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=self.atr_window
        )
        feat["atr"] = atr_obj.average_true_range()
        feat["atr_pct"] = feat["atr"] / (close + 1e-10)

        for w in self.vol_windows:
            ret = close.pct_change()
            feat[f"vol_{w}d"] = ret.rolling(w, min_periods=w // 2).std() * np.sqrt(252)
            feat[f"vol_{w}d_log"] = np.log(feat[f"vol_{w}d"] + 1e-10)

        # Vol regime: short/long ratio
        if len(self.vol_windows) >= 2:
            v_s = feat[f"vol_{self.vol_windows[0]}d"]
            v_l = feat[f"vol_{self.vol_windows[-1]}d"]
            feat["vol_ratio"] = v_s / (v_l + 1e-10)

        # 8. Volume features
        feat["vol_ma20"] = volume.rolling(20, min_periods=10).mean()
        feat["vol_ratio_20d"] = volume / (feat["vol_ma20"] + 1e-10)
        feat["vol_trend_5d"] = volume.pct_change(5)
        feat["dollar_volume"] = close * volume
        feat["log_dollar_vol"] = np.log(feat["dollar_volume"] + 1)

        # 9. Price range features
        feat["high_low_pct"] = (high - low) / (close + 1e-10)
        feat["close_high_pct"] = (high - close) / (high + 1e-10)
        feat["close_low_pct"] = (close - low) / (close + 1e-10)

        # 10. Rolling statistics
        ret = close.pct_change()
        for w in [21, 63, 126]:
            feat[f"ret_mean_{w}d"] = ret.rolling(w, min_periods=w // 2).mean()
            feat[f"ret_skew_{w}d"] = ret.rolling(w, min_periods=w // 2).skew()
            feat[f"ret_kurt_{w}d"] = ret.rolling(w, min_periods=w // 2).kurt()
            feat[f"ret_min_{w}d"] = ret.rolling(w, min_periods=w // 2).min()
            feat[f"ret_max_{w}d"] = ret.rolling(w, min_periods=w // 2).max()

        # 11. Lag features (of returns)
        for lag in self.lag_periods:
            feat[f"ret_lag_{lag}"] = ret.shift(lag)
            feat[f"vol_lag_{lag}"] = feat[f"vol_{self.vol_windows[0]}d"].shift(lag)

        # 12. 52-week high/low proximity
        high_52w = high.rolling(252, min_periods=126).max()
        low_52w = low.rolling(252, min_periods=126).min()
        feat["pct_from_52w_high"] = (close - high_52w) / (high_52w + 1e-10)
        feat["pct_from_52w_low"] = (close - low_52w) / (low_52w + 1e-10)

        # 13. Z-score of price
        feat["price_zscore"] = zscore(close, window=self.zscore_window)

        # 14. Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        feat["stoch_k"] = stoch.stoch()
        feat["stoch_d"] = stoch.stoch_signal()

        # 15. CCI
        feat["cci"] = ta.trend.CCIIndicator(
            high=high, low=low, close=close, window=20
        ).cci()

        # 16. Williams %R
        feat["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14
        ).williams_r()

        # 17. ADX (trend strength)
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        feat["adx"] = adx.adx()
        feat["adx_pos"] = adx.adx_pos()
        feat["adx_neg"] = adx.adx_neg()
        feat["adx_diff"] = feat["adx_pos"] - feat["adx_neg"]

        # 18. OBV
        feat["obv"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        feat["obv_ma20"] = feat["obv"].rolling(20, min_periods=10).mean()
        feat["obv_trend"] = (feat["obv"] - feat["obv_ma20"]) / (feat["obv_ma20"].abs() + 1e-10)

        # 19. VWAP approximation
        feat["vwap_approx"] = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
        feat["price_vwap_ratio"] = close / (feat["vwap_approx"] + 1e-10)

        # 20. Log returns
        feat["log_ret_1d"] = np.log(close / close.shift(1))
        for w in self.momentum_windows:
            feat[f"log_mom_{w}d"] = np.log(close / close.shift(w))

        # Drop helper columns not needed downstream
        drop_cols = ["ma_20", "ma_50", "ma_100", "ma_200", "vol_ma20",
                     "bb_upper", "bb_lower", "obv_ma20", "vwap_approx"]
        feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns])

        return feat

    # ── Post-processing ────────────────────────────────────────────────────

    @staticmethod
    def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Add per-date cross-sectional rank features (require full panel).
        """
        # Identify return-like columns for ranking
        mom_cols = [c for c in panel.columns if c.startswith("mom_") or c.startswith("log_mom_")]

        log.info(f"Adding cross-sectional ranks for {len(mom_cols)} momentum columns")
        panel = panel.copy()
        for col in mom_cols:
            panel[f"cs_rank_{col}"] = (
                panel[col]
                .groupby(level="Date")
                .rank(pct=True)
            )
        return panel

    @staticmethod
    def normalize(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorise and z-score numeric features cross-sectionally.
        """
        log.info("Normalizing feature panel …")
        result = panel.copy()
        num_cols = result.select_dtypes(include=[np.number]).columns

        # Winsorize per column
        for col in num_cols:
            try:
                result[col] = winsorize(result[col], lower=0.01, upper=0.99)
            except Exception:
                pass

        return result
