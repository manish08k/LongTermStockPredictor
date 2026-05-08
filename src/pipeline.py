"""
pipeline.py – End-to-end orchestration of the quant-longterm-ai system.

Steps:
  1.  Data ingestion (NSE/BSE universe)
  2.  Validation & alignment
  3.  Feature engineering (200+ features)
  4.  Alpha factors (100+ WorldQuant-style)
  5.  Labeling (benchmark-relative, multi-horizon)
  6.  Walk-forward model training (XGBoost, LightGBM, RF, etc.)
  7.  Ensemble scoring (ML + alpha combo)
  8.  Regime detection
  9.  Stock ranking (top-N with sector diversification)
  10. Portfolio optimization (max Sharpe / HRP / risk parity)
  11. Backtesting with transaction costs
  12. Evaluation & HTML report
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.alpha_factors import AlphaFactors
from src.backtest import BacktestEngine, BacktestResult
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidator
from src.ensemble import EnsemblePredictor
from src.evaluation import Evaluator
from src.feature_engineering import FeatureEngineer
from src.labeling import Labeler
from src.model import ModelTrainer
from src.portfolio import PortfolioOptimizer
from src.ranking import StockRanker
from src.regime import RegimeDetector, Regime
from src.risk_management import RiskManager
from src.utils import (
    ensure_dir, get_logger, load_config, save_parquet,
    set_random_seed, timeit,
)

log = get_logger(__name__)

# Labels / OHLCV columns to exclude from feature matrix
_EXCLUDE_PREFIXES = (
    "fwd_ret_", "excess_ret_", "ann_ret_",
    "label_", "meta_label",
)
_EXCLUDE_EXACT = {"Open", "High", "Low", "Close", "Volume", "Ticker"}


class QuantPipeline:
    """Master pipeline wiring all quant components."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = load_config(config_path)
        set_random_seed(self.cfg["project"].get("random_seed", 42))

        # ── Component initialization ───────────────────────────────────────
        self.ingestion = DataIngestion(self.cfg)
        self.validator = DataValidator(self.cfg)
        self.feature_eng = FeatureEngineer(self.cfg)
        self.alpha_comp = AlphaFactors(self.cfg)
        self.labeler = Labeler(self.cfg)
        self.trainer = ModelTrainer(self.cfg)
        self.ensemble = EnsemblePredictor(self.cfg)
        self.ranker = StockRanker(self.cfg)
        self.optimizer = PortfolioOptimizer(self.cfg)
        self.risk_mgr = RiskManager(self.cfg)
        self.regime_detector = RegimeDetector(self.cfg)
        self.backtest_engine = BacktestEngine(self.cfg)
        self.evaluator = Evaluator(self.cfg)

        # State
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.clean_data: Dict[str, pd.DataFrame] = {}
        self.labeled_panel: Optional[pd.DataFrame] = None

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def run(self) -> BacktestResult:
        """Execute the full pipeline end-to-end."""
        log.info("=" * 65)
        log.info("  QUANT LONGTERM AI — NSE INDIAN EQUITIES PLATFORM v2.0")
        log.info("=" * 65)

        # 1. Data ingestion
        log.info("[1/12] Ingesting data …")
        self.raw_data = self.ingestion.fetch_all()

        # 2. Validation & alignment
        log.info("[2/12] Validating data …")
        self.clean_data = self.validator.validate(self.raw_data)
        aligned = self.validator.align_panel(self.clean_data)

        benchmark_ticker = self.cfg["data"].get("benchmark_ticker", "^NSEI")
        benchmark_df = self.clean_data.get(benchmark_ticker)
        stock_data = {k: v for k, v in aligned.items() if k != benchmark_ticker}

        if not stock_data:
            raise RuntimeError("No stock data after validation. Check tickers/dates.")

        log.info(f"Universe: {len(stock_data)} stocks | Benchmark: {benchmark_ticker}")

        # 3. Feature engineering
        log.info("[3/12] Engineering features …")
        feature_panel = self.feature_eng.build_features(stock_data)
        feature_panel = self.feature_eng.add_cross_sectional_features(feature_panel)
        feature_panel = self.feature_eng.normalize(feature_panel)

        # 4. Merge OHLCV for alpha factor computation
        log.info("[4/12] Merging OHLCV into panel …")
        feature_panel = _merge_ohlcv(feature_panel, stock_data)

        # 5. Alpha factors
        log.info("[5/12] Computing alpha factors …")
        feature_panel = self.alpha_comp.compute(feature_panel)

        # 6. Labeling
        log.info("[6/12] Attaching labels …")
        labeled_panel = self.labeler.label(feature_panel, benchmark_df)
        self.labeled_panel = labeled_panel

        # Save feature panel
        feat_path = (
            Path(self.cfg["data"]["features_data_path"]) / "panel.parquet"
        )
        ensure_dir(feat_path.parent)
        save_parquet(labeled_panel, feat_path)
        log.info(f"Feature panel saved → {feat_path} | shape={labeled_panel.shape}")

        # 7. ML dataset preparation
        log.info("[7/12] Preparing ML dataset …")
        X, y, feature_cols = _prepare_ml_data(labeled_panel)
        pos_rate = y.mean() if len(y) > 0 else 0
        log.info(
            f"ML dataset: {X.shape[0]} rows × {X.shape[1]} features | "
            f"pos_rate={pos_rate:.2%}"
        )

        # 8. Walk-forward model training
        log.info("[8/12] Training models (walk-forward) …")
        model_names = self.cfg["models"].get("use_models", ["xgboost", "lightgbm"])
        trained_models = self.trainer.train(X, y, model_names)

        # 9. Regime detection
        log.info("[9/12] Detecting market regimes …")
        regime_series = (
            self.regime_detector.detect(benchmark_df)
            if benchmark_df is not None
            else None
        )

        # 10. Ensemble scoring (walk-forward predict)
        log.info("[10/12] Computing ensemble scores …")
        ml_scores = _walk_forward_score(
            labeled_panel, feature_cols, trained_models, self.trainer, self.ensemble
        )

        # Combine ML + alpha scores
        alpha_cols = [c for c in labeled_panel.columns if c.startswith("a0") or c.startswith("a1")]
        if alpha_cols:
            alpha_composite = labeled_panel[alpha_cols].mean(axis=1).reindex(ml_scores.index)
            ensemble_scores = self.ensemble.combine_ml_alpha(ml_scores, alpha_composite)
        else:
            ensemble_scores = ml_scores

        # 11. Ranking
        log.info("[11/12] Ranking stocks …")
        ranked = self.ranker.rank(ensemble_scores)
        latest_top = self.ranker.latest_top_stocks(ensemble_scores)
        _print_top_stocks(latest_top)

        # 12. Portfolio & backtest
        log.info("[12/12] Portfolio optimization & backtesting …")
        result = self._run_backtest(
            ensemble_scores, stock_data, benchmark_df, regime_series
        )

        # Evaluation
        metrics = self.evaluator.evaluate(result)
        self.evaluator.save_metrics_csv(metrics)

        log.info("=" * 65)
        log.info("  PIPELINE COMPLETE ✓")
        log.info("=" * 65)
        return result

    def rank_stocks_now(self) -> pd.DataFrame:
        """
        Convenience method: just rank the latest stocks without full backtest.
        """
        raw = self.ingestion.fetch_all()
        clean = self.validator.validate(raw)
        aligned = self.validator.align_panel(clean)
        benchmark_ticker = self.cfg["data"].get("benchmark_ticker", "^NSEI")
        benchmark_df = clean.get(benchmark_ticker)
        stock_data = {k: v for k, v in aligned.items() if k != benchmark_ticker}

        fp = self.feature_eng.build_features(stock_data)
        fp = self.feature_eng.add_cross_sectional_features(fp)
        fp = self.feature_eng.normalize(fp)
        fp = _merge_ohlcv(fp, stock_data)
        fp = self.alpha_comp.compute(fp)
        labeled = self.labeler.label(fp, benchmark_df)

        X, y, feature_cols = _prepare_ml_data(labeled)
        trained = self.trainer.train(X, y)
        ml_scores = _walk_forward_score(
            labeled, feature_cols, trained, self.trainer, self.ensemble
        )
        return self.ranker.latest_top_stocks(ml_scores)

    # ── Backtest orchestration ─────────────────────────────────────────────

    def _run_backtest(
        self,
        ensemble_scores: pd.Series,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
        regime_series: Optional[pd.Series],
    ) -> BacktestResult:
        """Build rebalance weights and run the backtest engine."""
        all_dates = (
            ensemble_scores.index.get_level_values("Date").unique().sort_values()
        )
        rebal_dates = _get_rebalance_dates(
            all_dates, self.cfg["backtest"]["rebalance_frequency"]
        )

        portfolio_weights: Dict[pd.Timestamp, Dict[str, float]] = {}
        top_n = self.cfg["ranking"]["top_n"]

        for rdate in rebal_dates:
            try:
                # Get latest scores at or before this date
                avail = all_dates[all_dates <= rdate]
                if len(avail) == 0:
                    continue
                score_date = avail[-1]

                # Scores at this date
                day_scores = ensemble_scores.loc[score_date].dropna()
                if day_scores.empty:
                    continue

                # Regime adjustment
                regime = Regime.BULL
                if regime_series is not None:
                    regime = self.regime_detector.regime_at_date(
                        benchmark_df, rdate
                    )
                adj_top_n = self.regime_detector.get_top_n_override(regime, top_n)
                exposure = self.regime_detector.get_exposure_multiplier(regime)

                # Top-N candidates
                top_tickers = (
                    day_scores.sort_values(ascending=False)
                    .head(adj_top_n * 3)  # over-select then filter
                    .index.tolist()
                )

                # Risk filter
                filtered = self.risk_mgr.filter_universe(
                    top_tickers, stock_data, as_of_date=rdate
                )
                if not filtered:
                    filtered = top_tickers

                # Sector diversification
                filtered = self.ranker.apply_sector_diversification(filtered)
                filtered = filtered[:adj_top_n]

                if not filtered:
                    continue

                # Portfolio optimization
                weights = self.optimizer.optimize(
                    filtered, stock_data,
                    scores=day_scores.loc[[t for t in filtered if t in day_scores.index]],
                )

                # Scale down in bear/high-vol regimes
                if exposure < 1.0:
                    weights = {t: w * exposure for t, w in weights.items()}
                    # remainder = cash (not modelled explicitly)

                if weights:
                    portfolio_weights[rdate] = weights
                    log.debug(
                        f"Rebal {rdate.date()} [{regime.value}] "
                        f"{len(weights)} stocks"
                    )

            except Exception as e:
                log.warning(f"Portfolio construction failed @ {rdate}: {e}")

        # Fallback if no weights generated
        if not portfolio_weights:
            log.error("No portfolio weights generated; using equal-weight fallback")
            all_tickers = list(stock_data.keys())[:top_n]
            w = 1.0 / len(all_tickers)
            portfolio_weights[all_dates[0]] = {t: w for t in all_tickers}

        log.info(f"Rebalance dates with weights: {len(portfolio_weights)}")

        # Use benchmark as benchmark, or dummy
        bm = benchmark_df if benchmark_df is not None else next(iter(stock_data.values()))

        return self.backtest_engine.run(
            portfolio_weights_by_date=portfolio_weights,
            price_data=stock_data,
            benchmark_data=bm,
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

def _merge_ohlcv(
    feature_panel: pd.DataFrame,
    stock_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    ohlcv_frames = []
    for tkr, df in stock_data.items():
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        sub = df[cols].copy()
        sub["Ticker"] = tkr
        sub.index.name = "Date"
        ohlcv_frames.append(sub.reset_index().set_index(["Date", "Ticker"]))

    if not ohlcv_frames:
        return feature_panel

    ohlcv = pd.concat(ohlcv_frames)
    # Only add columns not already present
    new_cols = [c for c in ohlcv.columns if c not in feature_panel.columns]
    if new_cols:
        return feature_panel.join(ohlcv[new_cols], how="left")
    return feature_panel


def _prepare_ml_data(
    labeled_panel: pd.DataFrame,
    label_col: str = "label_primary",
):
    """Split panel into X (features) and y (labels)."""
    feature_cols = [
        c for c in labeled_panel.columns
        if not any(c.startswith(p) for p in _EXCLUDE_PREFIXES)
        and c not in _EXCLUDE_EXACT
        and labeled_panel[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    df = labeled_panel.dropna(subset=[label_col])
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Final clean: remove inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    valid = y.notna()
    X, y = X[valid], y[valid]

    return X, y, feature_cols


def _walk_forward_score(
    labeled_panel: pd.DataFrame,
    feature_cols: List[str],
    trained_models: dict,
    trainer: ModelTrainer,
    ensemble: EnsemblePredictor,
) -> pd.Series:
    """Score all (Date, Ticker) pairs with trained models."""
    X = labeled_panel[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model_preds = trainer.predict(trained_models, X)
    if not model_preds:
        log.warning("No model predictions; using random scores as fallback")
        return pd.Series(
            np.random.uniform(0.45, 0.55, len(X)), index=X.index, name="ensemble_score"
        )

    return ensemble.predict(model_preds, index=X.index)


def _get_rebalance_dates(
    all_dates: pd.DatetimeIndex, freq: str
) -> List[pd.Timestamp]:
    s = pd.Series(all_dates, index=all_dates)
    period_map = {
        "monthly": "ME",
        "quarterly": "QE",
        "weekly": "W",
        "daily": None,
    }
    period_key = period_map.get(freq, "ME")
    if period_key is None:
        return list(all_dates)
    return [g.iloc[-1] for _, g in s.groupby(pd.Grouper(freq=period_key))]


def _print_top_stocks(top_df: pd.DataFrame) -> None:
    if top_df.empty:
        log.warning("No top stocks to display")
        return
    print("\n" + "=" * 55)
    print("  TOP RANKED NSE STOCKS (Latest Signal)")
    print("=" * 55)
    cols = [c for c in ["Rank", "Ticker", "Score", "Sector"] if c in top_df.columns]
    print(top_df[cols].to_string(index=False))
    print("=" * 55 + "\n")