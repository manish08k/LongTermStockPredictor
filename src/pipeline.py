"""
pipeline.py – End-to-end orchestration of the quant system.
Steps: data → validate → features → alphas → labels → train →
        ensemble → rank → portfolio → backtest → evaluate.
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
from src.risk_management import RiskManager
from src.utils import (
    ensure_dir, get_logger, load_config, save_parquet,
    set_random_seed, timeit,
)

log = get_logger(__name__)


class QuantPipeline:
    """
    Master pipeline that wires all components together.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = load_config(config_path)
        set_random_seed(self.cfg.get("project", {}).get("random_seed", 42))

        # Initialise components
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
        self.backtest_engine = BacktestEngine(self.cfg)
        self.evaluator = Evaluator(self.cfg)

        # Data store
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.clean_data: Dict[str, pd.DataFrame] = {}
        self.feature_panel: Optional[pd.DataFrame] = None
        self.labeled_panel: Optional[pd.DataFrame] = None

    # ── Public API ─────────────────────────────────────────────────────────

    @timeit
    def run(self) -> BacktestResult:
        """Execute the full pipeline end-to-end."""
        log.info("=" * 60)
        log.info("  QUANT LONGTERM AI PIPELINE STARTING")
        log.info("=" * 60)

        # ── Step 1: Data ingestion ─────────────────────────────────────────
        log.info("[1/12] Data ingestion …")
        self.raw_data = self.ingestion.fetch_all()

        # ── Step 2: Validation & alignment ────────────────────────────────
        log.info("[2/12] Validation …")
        self.clean_data = self.validator.validate(self.raw_data)
        self.aligned_data = self.validator.align_panel(self.clean_data)

        # Get benchmark data
        benchmark_ticker = self.cfg["data"].get("benchmark_ticker", self.cfg["data"].get("benchmark", "^NSEI"))
        benchmark_df = self.clean_data.get(benchmark_ticker)
        stock_data = {k: v for k, v in self.aligned_data.items() if k != benchmark_ticker}

        if not stock_data:
            raise RuntimeError("No stock data available after validation")

        # ── Step 3: Feature engineering ────────────────────────────────────
        log.info("[3/12] Feature engineering …")
        feature_panel = self.feature_eng.build_features(stock_data)
        feature_panel = self.feature_eng.add_cross_sectional_features(feature_panel)
        feature_panel = self.feature_eng.normalize(feature_panel)

        # ── Step 4: Add OHLCV columns to panel (for alpha factors) ─────────
        log.info("[4/12] Merging OHLCV into panel …")
        feature_panel = self._merge_ohlcv(feature_panel, stock_data)

        # ── Step 5: Alpha factors ──────────────────────────────────────────
        log.info("[5/12] Computing alpha factors …")
        feature_panel = self.alpha_comp.compute(feature_panel)

        # ── Step 6: Labeling ───────────────────────────────────────────────
        log.info("[6/12] Labeling …")
        labeled_panel = self.labeler.label(feature_panel, benchmark_df)
        self.labeled_panel = labeled_panel

        # Save features
        feat_path = Path(self.cfg["data"]["features_data_path"]) / "panel.parquet"
        ensure_dir(feat_path.parent)
        save_parquet(labeled_panel, feat_path)
        log.info(f"Feature panel saved → {feat_path} ({labeled_panel.shape})")

        # ── Step 7: Prepare ML dataset ─────────────────────────────────────
        log.info("[7/12] Preparing ML dataset …")
        X, y, feature_cols = self._prepare_ml_data(labeled_panel)
        log.info(f"ML dataset: {X.shape[0]} rows × {X.shape[1]} features | "
                 f"pos_rate={y.mean():.2%}")

        # ── Step 8: Train models ───────────────────────────────────────────
        log.info("[8/12] Training models …")
        model_names = self.cfg["models"].get("use_models", ["xgboost", "lightgbm"])
        trained_models = self.trainer.train(X, y, model_names)

        # ── Step 9: Walk-forward scoring ───────────────────────────────────
        log.info("[9/12] Walk-forward scoring …")
        ensemble_scores = self._walk_forward_score(
            labeled_panel, feature_cols, trained_models
        )

        # ── Step 10: Ranking ───────────────────────────────────────────────
        log.info("[10/12] Ranking stocks …")
        ranked = self.ranker.rank(ensemble_scores)

        # Show latest top stocks
        latest_top = self.ranker.latest_top_stocks(ensemble_scores)
        self._print_top_stocks(latest_top)

        # ── Step 11: Portfolio construction & backtesting ──────────────────
        log.info("[11/12] Portfolio optimization & backtesting …")
        backtest_result = self._run_backtest(
            ranked, ensemble_scores, stock_data, benchmark_df
        )

        # ── Step 12: Evaluation ────────────────────────────────────────────
        log.info("[12/12] Evaluation …")
        metrics = self.evaluator.evaluate(backtest_result)
        self.evaluator.save_metrics_csv(metrics)

        log.info("Pipeline completed successfully ✓")
        return backtest_result

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _merge_ohlcv(
        feature_panel: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge OHLCV columns into the feature panel."""
        ohlcv_frames = []
        for tkr, df in stock_data.items():
            sub = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub["Ticker"] = tkr
            sub.index.name = "Date"
            sub = sub.reset_index().set_index(["Date", "Ticker"])
            ohlcv_frames.append(sub)

        if not ohlcv_frames:
            return feature_panel

        ohlcv = pd.concat(ohlcv_frames)
        merged = feature_panel.join(ohlcv, how="left")
        return merged

    @staticmethod
    def _prepare_ml_data(
        labeled_panel: pd.DataFrame,
        label_col: str = "label_primary",
    ):
        """
        Separate feature matrix X from labels y.
        Excludes label columns, raw OHLCV, and metadata.
        """
        exclude_prefixes = ("fwd_ret_", "label_", "ann_ret_", "meta_label")
        exclude_exact = {"Open", "High", "Low", "Close", "Volume", "Ticker"}

        feature_cols = [
            c for c in labeled_panel.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
            and c not in exclude_exact
            and labeled_panel[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]

        df = labeled_panel.dropna(subset=[label_col])
        X = df[feature_cols].copy()
        y = df[label_col].copy()

        # Align
        valid = y.notna() & X.notna().all(axis=1)
        X, y = X[valid], y[valid]

        return X, y, feature_cols

    def _walk_forward_score(
        self,
        labeled_panel: pd.DataFrame,
        feature_cols: List[str],
        trained_models: dict,
    ) -> pd.Series:
        """
        Score all (Date, Ticker) pairs using trained models.
        Use full-sample models (already trained) for simplicity;
        for production, use time-series walk-forward re-training.
        """
        X = labeled_panel[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        model_preds = self.trainer.predict(trained_models, X)

        if not model_preds:
            log.warning("No model predictions available; using random scores")
            return pd.Series(np.random.uniform(0.4, 0.6, len(X)), index=X.index)

        scores = self.ensemble.predict(model_preds, index=X.index)
        return scores

    def _run_backtest(
        self,
        ranked: pd.DataFrame,
        ensemble_scores: pd.Series,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
    ) -> BacktestResult:
        """Build portfolio weights per rebalance date and run backtest."""
        all_dates = ranked.index.get_level_values("Date").unique().sort_values()
        rebal_freq = self.cfg["backtest"]["rebalance_frequency"]
        rebal_dates = self._get_rebalance_dates(all_dates, rebal_freq)

        portfolio_weights: Dict[pd.Timestamp, Dict[str, float]] = {}

        for rdate in rebal_dates:
            try:
                # Get ranked stocks at this date (or nearest prior)
                available_dates = all_dates[all_dates <= rdate]
                if len(available_dates) == 0:
                    continue
                score_date = available_dates[-1]
                day_scores = ensemble_scores.loc[score_date]

                # Top-N stocks
                top_n = self.cfg["ranking"]["top_n"]
                top_tickers = (
                    day_scores
                    .sort_values(ascending=False)
                    .head(top_n * 2)  # over-select, then filter
                    .index.tolist()
                )

                # Volatility filter
                filtered = self.risk_mgr.filter_universe(
                    top_tickers, stock_data, as_of_date=rdate
                )
                if not filtered:
                    filtered = top_tickers[:top_n]
                filtered = filtered[:top_n]

                # Portfolio optimization
                day_score_series = day_scores.loc[filtered] if filtered else pd.Series()
                weights = self.optimizer.optimize(
                    filtered, stock_data, scores=day_score_series
                )

                if weights:
                    portfolio_weights[rdate] = weights
                    log.debug(f"Rebalance {rdate.date()}: {len(weights)} stocks")

            except Exception as e:
                log.warning(f"Portfolio construction failed for {rdate}: {e}")

        if not portfolio_weights:
            log.error("No portfolio weights generated; check data coverage")
            # Fallback: equal weight all stocks at start
            all_tickers = list(stock_data.keys())[:self.cfg["ranking"]["top_n"]]
            portfolio_weights[all_dates[0]] = {t: 1.0 / len(all_tickers) for t in all_tickers}

        log.info(f"Portfolio weights built for {len(portfolio_weights)} rebalance dates")

        if benchmark_df is None:
            # Create dummy benchmark
            first_tkr = next(iter(stock_data.values()))
            benchmark_df = first_tkr

        result = self.backtest_engine.run(
            portfolio_weights_by_date=portfolio_weights,
            price_data=stock_data,
            benchmark_data=benchmark_df,
        )
        return result

    @staticmethod
    def _get_rebalance_dates(
        all_dates: pd.DatetimeIndex,
        freq: str,
    ) -> List[pd.Timestamp]:
        """Extract rebalance dates from trading calendar."""
        if freq == "monthly":
            return [
                g.iloc[-1]
                for _, g in pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("M"))
            ]
        elif freq == "quarterly":
            return [
                g.iloc[-1]
                for _, g in pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("Q"))
            ]
        elif freq == "weekly":
            return [
                g.iloc[-1]
                for _, g in pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("W"))
            ]
        elif freq == "daily":
            return list(all_dates)
        else:
            return [
                g.iloc[-1]
                for _, g in pd.Series(all_dates).groupby(pd.Series(all_dates).dt.to_period("M"))
            ]

    @staticmethod
    def _print_top_stocks(top_df: pd.DataFrame) -> None:
        print("\n" + "=" * 50)
        print("  TOP RANKED STOCKS (Latest)")
        print("=" * 50)
        print(top_df.to_string(index=False))
        print("=" * 50 + "\n")
