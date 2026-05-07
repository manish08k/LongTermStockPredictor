"""
alpha_factors.py – 100+ WorldQuant-style alpha factors.
All factors are computed per-ticker, then cross-sectionally ranked.
Strictly avoids look-ahead bias (only uses data up to time t).
References: WorldQuant Alpha101, Kakushadze (2016).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger, timeit, winsorize, zscore

log = get_logger(__name__)


# ── Low-level helpers ────────────────────────────────────────────────────────

def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank at each date (grouped by Date level)."""
    return s.groupby(level="Date").rank(pct=True)


def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """Time-series rank of the last value within a rolling window."""
    return s.groupby(level="Ticker").transform(
        lambda x: x.rolling(window, min_periods=window // 2).apply(
            lambda v: stats.rankdata(v)[-1] / len(v), raw=True
        )
    )


def _delay(s: pd.Series, d: int) -> pd.Series:
    return s.groupby(level="Ticker").shift(d)


def _delta(s: pd.Series, d: int) -> pd.Series:
    return s.groupby(level="Ticker").diff(d)


def _ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level="Ticker").transform(lambda x: x.rolling(w, min_periods=w // 2).mean())


def _ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level="Ticker").transform(lambda x: x.rolling(w, min_periods=w // 2).std())


def _ts_min(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level="Ticker").transform(lambda x: x.rolling(w, min_periods=w // 2).min())


def _ts_max(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level="Ticker").transform(lambda x: x.rolling(w, min_periods=w // 2).max())


def _ts_sum(s: pd.Series, w: int) -> pd.Series:
    return s.groupby(level="Ticker").transform(lambda x: x.rolling(w, min_periods=w // 2).sum())


def _ts_corr(s1: pd.Series, s2: pd.Series, w: int) -> pd.Series:
    combined = pd.concat([s1.rename("a"), s2.rename("b")], axis=1)
    return combined.groupby(level="Ticker").apply(
        lambda df: df["a"].rolling(w, min_periods=w // 2).corr(df["b"])
    ).reset_index(level=0, drop=True).sort_index()


def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-10))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan) + 1e-10)


# ── Alpha Factor Library ─────────────────────────────────────────────────────

class AlphaFactors:
    """
    Compute 100+ alpha factors on a multi-ticker panel DataFrame
    with MultiIndex (Date, Ticker).

    The panel must contain columns: Open, High, Low, Close, Volume.
    Returns the same DataFrame with alpha_* columns appended.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @timeit
    def compute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all alpha factors and append to panel.
        """
        log.info("Computing alpha factors …")
        p = panel.copy()

        # Extract base series
        c = p["Close"] if "Close" in p.columns else p.get("close")
        o = p["Open"] if "Open" in p.columns else c
        h = p["High"] if "High" in p.columns else c
        l = p["Low"] if "Low" in p.columns else c
        v = p["Volume"] if "Volume" in p.columns else pd.Series(1.0, index=p.index)

        # Derived
        ret = _delta(c, 1) / (_delay(c, 1) + 1e-10)
        log_ret = _log(c / (_delay(c, 1) + 1e-10))
        vwap = _safe_div(_ts_sum(c * v, 10), _ts_sum(v, 10))
        adv20 = _ts_mean(v, 20)

        alphas: Dict[str, pd.Series] = {}

        # ── Momentum Alphas ──────────────────────────────────────────────

        # A001: 12-1 month momentum (skip most recent month)
        alphas["a001_mom_12_1"] = _delay(c, 21) / (_delay(c, 252) + 1e-10) - 1

        # A002: 6-1 month momentum
        alphas["a002_mom_6_1"] = _delay(c, 21) / (_delay(c, 126) + 1e-10) - 1

        # A003: 3-month momentum rank
        alphas["a003_mom_3m_rank"] = _rank(_delta(c, 63))

        # A004: 1-month return rank
        alphas["a004_mom_1m_rank"] = _rank(_delta(c, 21))

        # A005: Price acceleration (momentum of momentum)
        mom_3m = c / (_delay(c, 63) + 1e-10) - 1
        mom_6m = c / (_delay(c, 126) + 1e-10) - 1
        alphas["a005_mom_accel"] = mom_3m - mom_6m

        # A006: 52-week high ratio
        alphas["a006_52w_high"] = c / (_ts_max(h, 252) + 1e-10)

        # A007: 52-week low ratio
        alphas["a007_52w_low"] = c / (_ts_min(l, 252) + 1e-10)

        # A008: Consecutive up days
        up_day = (ret > 0).astype(float)
        alphas["a008_consec_up"] = _ts_sum(up_day, 20)

        # A009: Weighted momentum (more weight to recent)
        weights_10 = np.arange(1, 11)
        def _weighted_ret(x):
            if len(x) < 10:
                return np.nan
            w = weights_10 / weights_10.sum()
            return np.dot(x[-10:], w)
        alphas["a009_weighted_mom_10"] = ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(10).apply(_weighted_ret, raw=True)
        )

        # A010: Momentum reversal (short-term)
        alphas["a010_reversal_5d"] = -_rank(_delta(c, 5))

        # A011: Reversal 1-month
        alphas["a011_reversal_21d"] = -_rank(_delta(c, 21))

        # A012: Long-term reversal (3-5 year not available, use 1Y reversal)
        alphas["a012_lt_reversal"] = -(_delay(c, 252) / (_delay(c, 504) + 1e-10) - 1)

        # ── Mean-Reversion Alphas ────────────────────────────────────────

        # A013: Deviation from 200-day MA
        ma200 = _ts_mean(c, 200)
        alphas["a013_dev_ma200"] = -(c - ma200) / (ma200 + 1e-10)

        # A014: Bollinger %B (inverted for mean-reversion)
        ma20 = _ts_mean(c, 20)
        std20 = _ts_std(c, 20)
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        alphas["a014_bb_rev"] = -(c - ma20) / (std20 + 1e-10)

        # A015: RSI mean-reversion
        delta_c = _delta(c, 1)
        gain = delta_c.clip(lower=0)
        loss = (-delta_c).clip(lower=0)
        avg_gain = _ts_mean(gain, 14)
        avg_loss = _ts_mean(loss, 14)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        alphas["a015_rsi_rev"] = -(rsi - 50) / 50  # negative: buy oversold

        # A016: Stochastic oscillator reversion
        stoch_low = _ts_min(l, 14)
        stoch_high = _ts_max(h, 14)
        alphas["a016_stoch_rev"] = -((c - stoch_low) / (stoch_high - stoch_low + 1e-10) - 0.5)

        # A017: Z-score reversion
        alphas["a017_zscore_rev"] = -zscore(c.groupby(level="Ticker").transform(lambda x: x), window=63)

        # ── Volatility Alphas ────────────────────────────────────────────

        # A018: Low volatility factor
        alphas["a018_low_vol_21d"] = -_ts_std(ret, 21)

        # A019: Low volatility 63d
        alphas["a019_low_vol_63d"] = -_ts_std(ret, 63)

        # A020: Volatility ratio (short vs long)
        vol_10 = _ts_std(ret, 10)
        vol_63 = _ts_std(ret, 63)
        alphas["a020_vol_ratio"] = -(vol_10 / (vol_63 + 1e-10))

        # A021: Realized vol change
        alphas["a021_vol_change"] = _delta(_ts_std(ret, 21), 21)

        # A022: Idiosyncratic volatility proxy (std of residuals – simplified)
        # Using absolute deviation from cross-sectional mean return
        cross_mean_ret = ret.groupby(level="Date").transform("mean")
        idio = (ret - cross_mean_ret) ** 2
        alphas["a022_idio_vol"] = -_ts_mean(idio, 63) ** 0.5

        # A023: GARCH-like (EWMA vol)
        alphas["a023_ewma_vol"] = -ret.groupby(level="Ticker").transform(
            lambda x: x.ewm(span=21).std()
        )

        # A024: Downside deviation
        downside = ret.clip(upper=0)
        alphas["a024_downside_dev"] = -_ts_std(downside, 63)

        # A025: Max drawdown proxy (rolling)
        roll_max = _ts_max(c, 63)
        alphas["a025_drawdown_63d"] = -((roll_max - c) / (roll_max + 1e-10))

        # ── Value-like / Fundamental Proxies ────────────────────────────

        # A026: Price-to-52w-high (cheap stocks near high = momentum)
        alphas["a026_p2high"] = c / (_ts_max(h, 252) + 1e-10)

        # A027: Price-to-volume (liquidity-adjusted price)
        alphas["a027_p2vol"] = _log(c) - _log(adv20 + 1)

        # A028: Earnings-yield proxy: 1/PE ≈ earnings / price
        # Without fundamentals, use rolling earnings proxy = price acceleration
        alphas["a028_ep_proxy"] = _rank(-c / (_delay(c, 252) + 1e-10))

        # ── Volume / Liquidity Alphas ────────────────────────────────────

        # A029: Volume momentum
        alphas["a029_vol_mom_5d"] = _delta(v, 5) / (_delay(v, 5) + 1e-10)

        # A030: Volume surge
        alphas["a030_vol_surge"] = v / (adv20 + 1e-10)

        # A031: Price × volume trend
        pv = c * v
        alphas["a031_pv_trend"] = _delta(_ts_mean(pv, 5), 5) / (_delay(_ts_mean(pv, 5), 5) + 1e-10)

        # A032: OBV trend
        obv = _ts_sum(_sign(ret) * v, 20)
        alphas["a032_obv_trend"] = _rank(obv)

        # A033: Volume-weighted return
        alphas["a033_vw_ret"] = _safe_div(
            _ts_sum(ret * v, 10), _ts_sum(v, 10)
        )

        # A034: Turnover (volume/ADV ratio change)
        turnover = v / (adv20 + 1e-10)
        alphas["a034_turnover_change"] = -_delta(turnover, 5)

        # A035: Amihud illiquidity (inverted = liquidity)
        amihud = _ts_mean(ret.abs() / (v * c + 1e-10), 21)
        alphas["a035_liquidity"] = -amihud

        # A036: Dollar volume trend
        dv = c * v
        alphas["a036_dv_trend"] = _delta(_ts_mean(dv, 21), 21) / (_ts_mean(dv, 21).abs() + 1e-10)

        # A037: Volume concentration (are volume spikes persistent?)
        alphas["a037_vol_kurt"] = v.groupby(level="Ticker").transform(
            lambda x: x.rolling(21, min_periods=10).kurt()
        )

        # ── Price-Range / Intraday Alphas ────────────────────────────────

        # A038: High-Low range
        hl_range = (h - l) / (c + 1e-10)
        alphas["a038_hl_range"] = -_ts_mean(hl_range, 21)

        # A039: Close-to-High (selling pressure)
        alphas["a039_close_high"] = -(h - c) / (h + 1e-10)

        # A040: Close-to-Low (buying pressure)
        alphas["a040_close_low"] = (c - l) / (c + 1e-10)

        # A041: Upper shadow / lower shadow ratio
        upper_shadow = h - c.combine(o, max)
        lower_shadow = c.combine(o, min) - l
        alphas["a041_shadow_ratio"] = lower_shadow / (upper_shadow + 1e-10)

        # A042: Open-close return (intraday)
        alphas["a042_intraday_ret"] = (c - o) / (o + 1e-10)

        # A043: Gap (open vs prev close)
        alphas["a043_gap"] = (o - _delay(c, 1)) / (_delay(c, 1) + 1e-10)

        # A044: Range expansion
        prev_range = _delay(h - l, 1)
        alphas["a044_range_expansion"] = (h - l) / (prev_range + 1e-10)

        # ── Trend / Pattern Alphas ───────────────────────────────────────

        # A045: Trend consistency (% days positive in 63d)
        alphas["a045_trend_consistency"] = _ts_mean((ret > 0).astype(float), 63)

        # A046: MACD signal
        ema_fast = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=12).mean())
        ema_slow = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=26).mean())
        macd_line = ema_fast - ema_slow
        macd_sig = macd_line.groupby(level="Ticker").transform(lambda x: x.ewm(span=9).mean())
        alphas["a046_macd_hist"] = macd_line - macd_sig

        # A047: ADX (trend strength)
        tr = pd.concat([h - l, (h - _delay(c, 1)).abs(), (l - _delay(c, 1)).abs()], axis=1).max(axis=1)
        dm_plus = ((h - _delay(h, 1)) > (_delay(l, 1) - l)).astype(float) * (h - _delay(h, 1)).clip(lower=0)
        dm_minus = ((_delay(l, 1) - l) > (h - _delay(h, 1))).astype(float) * (_delay(l, 1) - l).clip(lower=0)
        atr14 = _ts_mean(tr, 14)
        di_plus = 100 * _ts_mean(dm_plus, 14) / (atr14 + 1e-10)
        di_minus = 100 * _ts_mean(dm_minus, 14) / (atr14 + 1e-10)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        alphas["a047_adx"] = _ts_mean(dx, 14)

        # A048: Hurst exponent proxy (R/S)
        def _rs_proxy(x):
            if len(x) < 20:
                return np.nan
            r = np.log(x).diff().dropna()
            if r.std() < 1e-10:
                return 0.5
            rs = (r.cumsum() - r.cumsum().mean()).abs().max() / r.std()
            return np.log(rs + 1e-10) / np.log(len(r) + 1e-10)
        alphas["a048_hurst"] = c.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=30).apply(_rs_proxy, raw=False)
        )

        # A049: Linear trend R² (how well does price fit a linear trend)
        def _r2(x):
            if len(x) < 5:
                return np.nan
            t = np.arange(len(x))
            slope, intercept, r, p, se = stats.linregress(t, x)
            return r ** 2 * np.sign(slope)
        alphas["a049_trend_r2"] = c.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).apply(_r2, raw=True)
        )

        # A050: Price channel breakout
        upper_ch = _ts_max(h, 20)
        lower_ch = _ts_min(l, 20)
        channel_mid = (upper_ch + lower_ch) / 2
        alphas["a050_channel_pos"] = (c - channel_mid) / ((upper_ch - lower_ch) / 2 + 1e-10)

        # ── Cross-Sectional Alphas ───────────────────────────────────────

        # A051: Cross-sectional momentum rank
        alphas["a051_cs_mom_3m"] = _rank(c / (_delay(c, 63) + 1e-10) - 1)

        # A052: Cross-sectional volume rank
        alphas["a052_cs_volume"] = _rank(v)

        # A053: Cross-sectional volatility rank (low vol = high rank)
        alphas["a053_cs_low_vol"] = -_rank(_ts_std(ret, 21))

        # A054: Cross-sectional return rank in past week
        alphas["a054_cs_ret_5d"] = _rank(_delta(c, 5))

        # A055: Cross-sectional dollar volume rank
        alphas["a055_cs_dv"] = _rank(c * v)

        # A056: Relative strength vs universe mean
        cs_mean_ret = ret.groupby(level="Date").transform("mean")
        alphas["a056_rel_strength"] = _ts_mean(ret - cs_mean_ret, 63)

        # ── Statistical / Time-Series Alphas ────────────────────────────

        # A057: Autocorrelation of returns
        def _autocorr(x, lag=1):
            if len(x) < lag + 2:
                return np.nan
            return pd.Series(x).autocorr(lag=lag)
        alphas["a057_autocorr_1"] = ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).apply(lambda v: _autocorr(v, 1), raw=True)
        )

        # A058: Autocorrelation lag 5
        alphas["a058_autocorr_5"] = ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).apply(lambda v: _autocorr(v, 5), raw=True)
        )

        # A059: Variance ratio (random walk test)
        def _var_ratio(x, q=5):
            if len(x) < q + 2:
                return np.nan
            r = pd.Series(x).pct_change().dropna()
            if len(r) < q:
                return np.nan
            var1 = r.var()
            varq = r.rolling(q).sum().var() / q
            return varq / (var1 + 1e-10)
        alphas["a059_var_ratio"] = c.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).apply(_var_ratio, raw=True)
        )

        # A060: Kurtosis of returns (tail risk)
        alphas["a060_ret_kurt"] = -ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).kurt()
        )

        # A061: Skewness of returns
        alphas["a061_ret_skew"] = ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).skew()
        )

        # A062: Tail ratio
        def _tail_ratio(x):
            s = pd.Series(x).dropna()
            if len(s) < 10:
                return np.nan
            return abs(s.quantile(0.95)) / (abs(s.quantile(0.05)) + 1e-10)
        alphas["a062_tail_ratio"] = ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(63, min_periods=20).apply(_tail_ratio, raw=True)
        )

        # ── Regression-based Alphas ──────────────────────────────────────

        # A063: Beta (market sensitivity) – proxy using cross-sectional correlation
        cs_ret = ret.groupby(level="Date").transform("mean")
        def _beta(x):
            # x is paired (stock_ret, market_ret) via rolling
            return np.nan  # placeholder (requires pairs)
        # Simplified: rank by correlation with market proxy
        alphas["a063_low_beta"] = -ret.groupby(level="Ticker").transform(
            lambda x: x.rolling(252, min_periods=63).std()
        )  # use vol as beta proxy

        # A064: Alpha (Jensen) proxy
        mkt_ret = ret.groupby(level="Date").transform("mean")
        alphas["a064_alpha_proxy"] = _ts_mean(ret - mkt_ret, 63)

        # A065: Information ratio proxy
        excess = ret - mkt_ret
        alphas["a065_ir_proxy"] = _safe_div(
            _ts_mean(excess, 63),
            _ts_std(excess, 63)
        )

        # ── Composite / Interaction Alphas ───────────────────────────────

        # A066: Momentum + low vol combo
        mom_3m_z = _rank(mom_3m)
        low_vol_z = _rank(-_ts_std(ret, 63))
        alphas["a066_mom_lowvol"] = 0.5 * mom_3m_z + 0.5 * low_vol_z

        # A067: Momentum + volume confirmation
        vol_surge = v / (adv20 + 1e-10)
        alphas["a067_mom_vol_confirm"] = _rank(mom_3m) * _rank(vol_surge)

        # A068: Trend + breakout
        alphas["a068_trend_breakout"] = _rank(alphas.get("a049_trend_r2", pd.Series(0, index=c.index))) * \
                                         _rank(alphas.get("a050_channel_pos", pd.Series(0, index=c.index)))

        # A069: Quality proxy (low vol + positive momentum)
        quality = (
            _rank(-_ts_std(ret, 63))
            + _rank(mom_3m)
            + _rank(_ts_mean(ret, 21))
        ) / 3
        alphas["a069_quality"] = quality

        # A070: Size factor proxy (log market cap ~ log(price × ADV))
        alphas["a070_size_inv"] = -_log(c * adv20 + 1)

        # ── Additional WorldQuant-inspired Alphas ────────────────────────

        # A071: (-1 * delta(vwap, 7))
        alphas["a071_dvwap_7"] = -_delta(vwap, 7)

        # A072: Correlation(rank(close), rank(volume), 5)
        try:
            alphas["a072_close_vol_corr5"] = _ts_corr(_rank(c), _rank(v), 5)
        except Exception:
            alphas["a072_close_vol_corr5"] = pd.Series(np.nan, index=c.index)

        # A073: (-1 * rank(decay_linear(delta(close, 10), 10)))
        def _decay_linear(x, d):
            weights = np.arange(1, d + 1, dtype=float)
            weights /= weights.sum()
            return x.groupby(level="Ticker").transform(
                lambda s: s.rolling(d, min_periods=d // 2).apply(
                    lambda v: np.dot(v, weights[-len(v):] / weights[-len(v):].sum()), raw=True
                )
            )
        alphas["a073_decay_delta_10"] = -_rank(_decay_linear(_delta(c, 10), 10))

        # A074: correlation(high, mean(volume,5), 5)
        try:
            alphas["a074_high_vol_corr"] = _ts_corr(h, _ts_mean(v, 5), 5)
        except Exception:
            alphas["a074_high_vol_corr"] = pd.Series(np.nan, index=c.index)

        # A075: rank(sign(delta(vwap, 5)))
        alphas["a075_vwap_sign"] = _rank(_sign(_delta(vwap, 5)))

        # A076: (-1 * max(rank(corr(rank(volume), rank(vwap), 5)), 3))
        try:
            corr_vv = _ts_corr(_rank(v), _rank(vwap), 5)
            alphas["a076_vol_vwap_corr"] = -_ts_max(_rank(corr_vv), 3)
        except Exception:
            alphas["a076_vol_vwap_corr"] = pd.Series(np.nan, index=c.index)

        # A077: Rank(ADX * momentum)
        adx_ser = alphas.get("a047_adx", pd.Series(0, index=c.index))
        alphas["a077_adx_mom"] = _rank(adx_ser * (c / (_delay(c, 63) + 1e-10) - 1))

        # A078: Volume-weighted momentum
        alphas["a078_vw_mom"] = _safe_div(
            _ts_sum(v * ret, 21),
            _ts_sum(v, 21)
        )

        # A079: Momentum consistency score
        alphas["a079_mom_consistency"] = (
            _rank(c / (_delay(c, 21) + 1e-10) - 1)
            + _rank(c / (_delay(c, 63) + 1e-10) - 1)
            + _rank(c / (_delay(c, 126) + 1e-10) - 1)
            + _rank(c / (_delay(c, 252) + 1e-10) - 1)
        ) / 4

        # A080: Short-term reversal adjusted for volume
        alphas["a080_vol_adj_rev"] = -_rank(_delta(c, 5)) * _rank(vol_surge)

        # A081-A090: More technical alphas
        # A081: Relative Volume Oscillator
        vol_ma5 = _ts_mean(v, 5)
        vol_ma20_ser = _ts_mean(v, 20)
        alphas["a081_rvo"] = (vol_ma5 - vol_ma20_ser) / (vol_ma20_ser + 1e-10)

        # A082: Chaikin Money Flow
        mf_mult = ((c - l) - (h - c)) / (h - l + 1e-10)
        mf_vol = mf_mult * v
        alphas["a082_cmf"] = _safe_div(_ts_sum(mf_vol, 20), _ts_sum(v, 20))

        # A083: Force Index
        alphas["a083_force_index"] = _delta(c, 1) * v

        # A084: Elder's Bull Power / Bear Power
        ema13 = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=13).mean())
        alphas["a084_bull_power"] = h - ema13
        alphas["a085_bear_power"] = l - ema13

        # A086: Price Oscillator
        ema10 = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=10).mean())
        ema20 = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=20).mean())
        alphas["a086_price_osc"] = (ema10 - ema20) / (ema20 + 1e-10)

        # A087: Detrended Price Oscillator
        ma_10 = _ts_mean(c, 10)
        alphas["a087_dpo"] = c - _delay(ma_10, 6)

        # A088: Williams Accumulation/Distribution
        true_high = pd.concat([h, _delay(c, 1)], axis=1).max(axis=1)
        true_low = pd.concat([l, _delay(c, 1)], axis=1).min(axis=1)
        alphas["a088_williams_ad"] = _ts_sum(
            ((c - true_low) - (true_high - c)) / (true_high - true_low + 1e-10) * v, 14
        )

        # A089: Ease of Movement
        dist_moved = ((h + l) / 2) - ((_delay(h, 1) + _delay(l, 1)) / 2)
        box_ratio = (v / 1e6) / (h - l + 1e-10)
        alphas["a089_eom"] = dist_moved / (box_ratio + 1e-10)

        # A090: Mass Index (range expansion indicator)
        ema_hl = (h - l).groupby(level="Ticker").transform(lambda x: x.ewm(span=9).mean())
        ema_hl2 = ema_hl.groupby(level="Ticker").transform(lambda x: x.ewm(span=9).mean())
        alphas["a090_mass_index"] = _ts_sum(ema_hl / (ema_hl2 + 1e-10), 25)

        # A091-A101: Final batch
        # A091: Commodity Channel Index
        tp = (h + l + c) / 3
        alphas["a091_cci"] = (tp - _ts_mean(tp, 20)) / (0.015 * _ts_std(tp, 20) + 1e-10)

        # A092: Ulcer Index (downside risk)
        def _ulcer(x):
            if len(x) < 5:
                return np.nan
            maxp = np.maximum.accumulate(x)
            pct_dd = 100 * (x - maxp) / (maxp + 1e-10)
            return np.sqrt(np.mean(pct_dd ** 2))
        alphas["a092_ulcer_index"] = -c.groupby(level="Ticker").transform(
            lambda x: x.rolling(14, min_periods=7).apply(_ulcer, raw=True)
        )

        # A093: Coppock Curve
        roc14 = c / (_delay(c, 14) + 1e-10) - 1
        roc11 = c / (_delay(c, 11) + 1e-10) - 1
        alphas["a093_coppock"] = (roc14 + roc11).groupby(level="Ticker").transform(
            lambda x: x.ewm(span=10).mean()
        )

        # A094: Psychological Line (% days up in 12)
        alphas["a094_psych_line"] = _ts_mean((ret > 0).astype(float), 12)

        # A095: Rate of Change momentum
        alphas["a095_roc_20"] = c / (_delay(c, 20) + 1e-10) - 1
        alphas["a096_roc_60"] = c / (_delay(c, 60) + 1e-10) - 1

        # A097: Triple exponential (TRIX)
        ema1 = c.groupby(level="Ticker").transform(lambda x: x.ewm(span=15).mean())
        ema2 = ema1.groupby(level="Ticker").transform(lambda x: x.ewm(span=15).mean())
        ema3 = ema2.groupby(level="Ticker").transform(lambda x: x.ewm(span=15).mean())
        alphas["a097_trix"] = _delta(ema3, 1) / (ema3 + 1e-10)

        # A098: Price relative to VWAP
        alphas["a098_price_vwap"] = (c - vwap) / (vwap + 1e-10)

        # A099: Volume-adjusted price change
        alphas["a099_vol_adj_chg"] = ret * _rank(v / (adv20 + 1e-10))

        # A100: Multi-timeframe momentum score
        alphas["a100_mtf_mom"] = (
            _sign(c / (_delay(c, 5) + 1e-10) - 1)
            + _sign(c / (_delay(c, 21) + 1e-10) - 1)
            + _sign(c / (_delay(c, 63) + 1e-10) - 1)
            + _sign(c / (_delay(c, 126) + 1e-10) - 1)
            + _sign(c / (_delay(c, 252) + 1e-10) - 1)
        )

        # A101: Mean-reversion after large moves
        large_move = ret.abs() > _ts_std(ret, 63) * 2
        alphas["a101_post_spike_rev"] = -large_move.astype(float) * _sign(ret)

        # ── Append to panel ─────────────────────────────────────────────
        clean_alphas = {}
        for name, series in alphas.items():
            try:
                if isinstance(series, pd.Series):
                    series = winsorize(series.replace([np.inf, -np.inf], np.nan).fillna(0))
                    clean_alphas[name] = series
            except Exception as e:
                log.debug(f"Alpha {name} failed: {e}")

        if clean_alphas:
            alpha_df = pd.DataFrame(clean_alphas, index=p.index)
            p = pd.concat([p, alpha_df], axis=1)

        log.info(f"Computed {len(alphas)} alpha factors → panel now {p.shape[1]} columns")
        return p
