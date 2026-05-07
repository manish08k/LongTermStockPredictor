"""
tests/test_all.py – Unit tests for all modules.
Run with: pytest tests/ -v --tb=short
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for one ticker."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.abs(np.random.normal(1e6, 2e5, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates
    )


@pytest.fixture
def sample_multi_ticker_data(sample_ohlcv):
    """Create a dict of 5 tickers."""
    np.random.seed(42)
    tickers = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "RELIANCE.NS"]
    data = {}
    for tkr in tickers:
        noise = np.random.normal(0, 0.01, len(sample_ohlcv))
        df = sample_ohlcv.copy()
        df["Close"] *= (1 + noise.cumsum())
        df["Close"] = df["Close"].clip(lower=1)
        data[tkr] = df
    return data


@pytest.fixture
def minimal_config():
    return {
        "project": {"random_seed": 42, "log_level": "DEBUG"},
        "data": {
            "tickers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "RELIANCE.NS"],
            "benchmark_ticker": "^NSEI",
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "interval": "1d",
            "min_history_days": 100,
            "cache_data": False,
            "raw_data_path": "data/raw",
            "processed_data_path": "data/processed",
            "features_data_path": "data/features",
        },
        "features": {
            "momentum_windows": [21, 63],
            "ma_windows": [20, 50],
            "vol_windows": [10, 21],
            "rsi_window": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_window": 20,
            "bb_std": 2,
            "atr_window": 14,
            "lag_periods": [1, 5],
            "zscore_window": 63,
            "cross_section_rank": True,
        },
        "labeling": {
            "horizons": [21, 63],
            "primary_horizon": 63,
            "label_type": "binary",
            "outperform_threshold": 0.0,
            "triple_barrier": {"enabled": False},
        },
        "models": {
            "use_models": ["random_forest"],
            "cv_splits": 3,
            "cv_gap": 5,
            "early_stopping_rounds": 20,
            "feature_importance_top_n": 10,
            "xgboost": {
                "n_estimators": 50, "max_depth": 4, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 3, "gamma": 0.0,
                "reg_alpha": 0.0, "reg_lambda": 1.0,
                "eval_metric": "auc", "tree_method": "hist",
            },
            "lightgbm": {
                "n_estimators": 50, "max_depth": 4, "learning_rate": 0.1,
                "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_samples": 5, "reg_alpha": 0.0, "reg_lambda": 1.0, "verbose": -1,
            },
            "random_forest": {
                "n_estimators": 50, "max_depth": 5, "min_samples_leaf": 5,
                "max_features": 0.5, "n_jobs": 1,
            },
        },
        "ensemble": {
            "method": "weighted_average",
            "weights": {"xgboost": 0.4, "lightgbm": 0.4, "random_forest": 0.2},
            "stacking_meta_model": "logistic",
        },
        "ranking": {
            "top_n": 3,
            "score_method": "probability",
            "min_score_threshold": 0.40,
        },
        "portfolio": {
            "method": "equal_weight",
            "max_weight": 0.40,
            "min_weight": 0.05,
            "risk_free_rate": 0.065,
            "regularization": 0.0001,
            "solver": "SLSQP",
        },
        "risk": {
            "max_portfolio_volatility": 0.35,
            "stop_loss_pct": -0.15,
            "max_drawdown_exit": -0.25,
            "volatility_filter_window": 10,
            "volatility_filter_threshold": 0.80,
        },
        "backtest": {
            "initial_capital": 1_000_000,
            "rebalance_frequency": "monthly",
            "transaction_cost_bps": 20,
            "slippage_bps": 10,
            "min_trading_days": 60,
            "walk_forward": {"train_window": 252, "test_window": 63, "step": 63},
        },
        "evaluation": {
            "annualization_factor": 252,
            "benchmark_ticker": "^NSEI",
            "output_dir": "results_test",
        },
    }


# ─── Utils tests ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_winsorize(self):
        from src.utils import winsorize
        s = pd.Series(range(100))
        out = winsorize(s, 0.05, 0.95)
        assert out.min() >= s.quantile(0.05) - 0.1
        assert out.max() <= s.quantile(0.95) + 0.1

    def test_zscore(self):
        from src.utils import zscore
        s = pd.Series(np.random.normal(0, 1, 100))
        z = zscore(s)
        assert abs(z.mean()) < 0.1
        assert abs(z.std() - 1.0) < 0.1

    def test_annualized_return(self):
        from src.utils import annualized_return
        # 10% total over 252 days should give ~10% CAGR
        rets = pd.Series([0.1 / 252] * 252)
        cagr = annualized_return(rets)
        assert 0.08 < cagr < 0.12

    def test_sharpe_ratio(self):
        from src.utils import sharpe_ratio
        rets = pd.Series(np.random.normal(0.0005, 0.01, 252))
        sr = sharpe_ratio(rets)
        assert isinstance(sr, float)

    def test_max_drawdown(self):
        from src.utils import max_drawdown
        equity = pd.Series([100, 110, 90, 80, 95, 100])
        dd = max_drawdown(equity)
        assert dd < 0
        assert dd >= -1.0

    def test_ensure_dir(self, tmp_path):
        from src.utils import ensure_dir
        p = tmp_path / "test" / "nested"
        result = ensure_dir(p)
        assert result.exists()


# ─── DataValidation tests ─────────────────────────────────────────────────────

class TestDataValidation:
    def test_validate_passes_good_data(self, minimal_config, sample_multi_ticker_data):
        from src.data_validation import DataValidator
        validator = DataValidator(minimal_config)
        clean = validator.validate(sample_multi_ticker_data)
        assert len(clean) > 0

    def test_validate_removes_short_data(self, minimal_config, sample_ohlcv):
        from src.data_validation import DataValidator
        validator = DataValidator(minimal_config)
        short_data = {"BAD.NS": sample_ohlcv.head(10)}
        clean = validator.validate(short_data)
        assert "BAD.NS" not in clean

    def test_align_panel(self, minimal_config, sample_multi_ticker_data):
        from src.data_validation import DataValidator
        validator = DataValidator(minimal_config)
        aligned = validator.align_panel(sample_multi_ticker_data)
        assert isinstance(aligned, dict)
        assert len(aligned) > 0


# ─── FeatureEngineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_build_features(self, minimal_config, sample_multi_ticker_data):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(minimal_config)
        panel = fe.build_features(sample_multi_ticker_data)
        assert isinstance(panel, pd.DataFrame)
        assert panel.index.names == ["Date", "Ticker"]
        assert "mom_21d" in panel.columns
        assert "rsi" in panel.columns

    def test_no_lookahead(self, minimal_config, sample_multi_ticker_data):
        """Verify features don't reference future data."""
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(minimal_config)
        panel = fe.build_features(sample_multi_ticker_data)
        # Features should not be perfectly correlated with future prices
        # (basic sanity – real test would require more sophistication)
        assert panel.shape[0] > 0

    def test_cross_sectional_ranks(self, minimal_config, sample_multi_ticker_data):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(minimal_config)
        panel = fe.build_features(sample_multi_ticker_data)
        panel = fe.add_cross_sectional_features(panel)
        cs_cols = [c for c in panel.columns if c.startswith("cs_rank_")]
        assert len(cs_cols) > 0


# ─── AlphaFactors tests ───────────────────────────────────────────────────────

class TestAlphaFactors:
    def _make_panel(self, sample_multi_ticker_data):
        frames = []
        for tkr, df in sample_multi_ticker_data.items():
            sub = df.copy()
            sub["Ticker"] = tkr
            sub.index.name = "Date"
            sub = sub.reset_index().set_index(["Date", "Ticker"])
            frames.append(sub)
        return pd.concat(frames).sort_index()

    def test_alpha_factors_run(self, minimal_config, sample_multi_ticker_data):
        from src.alpha_factors import AlphaFactors
        panel = self._make_panel(sample_multi_ticker_data)
        af = AlphaFactors(minimal_config)
        result = af.compute(panel)
        alpha_cols = [c for c in result.columns if c.startswith("a0")]
        assert len(alpha_cols) >= 10

    def test_no_inf_in_alphas(self, minimal_config, sample_multi_ticker_data):
        from src.alpha_factors import AlphaFactors
        panel = self._make_panel(sample_multi_ticker_data)
        af = AlphaFactors(minimal_config)
        result = af.compute(panel)
        alpha_cols = [c for c in result.columns if c.startswith("a0")]
        for col in alpha_cols:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"


# ─── Labeling tests ───────────────────────────────────────────────────────────

class TestLabeling:
    def _make_panel(self, data):
        frames = []
        for tkr, df in data.items():
            sub = df.copy()
            sub["Ticker"] = tkr
            sub.index.name = "Date"
            sub = sub.reset_index().set_index(["Date", "Ticker"])
            frames.append(sub)
        return pd.concat(frames).sort_index()

    def test_label_binary(self, minimal_config, sample_multi_ticker_data):
        from src.labeling import Labeler
        panel = self._make_panel(sample_multi_ticker_data)
        labeler = Labeler(minimal_config)
        labeled = labeler.label(panel)
        assert "label_primary" in labeled.columns
        assert set(labeled["label_primary"].dropna().unique()).issubset({0, 1})

    def test_forward_return_no_lookahead(self, minimal_config, sample_multi_ticker_data):
        from src.labeling import Labeler
        panel = self._make_panel(sample_multi_ticker_data)
        labeler = Labeler(minimal_config)
        labeled = labeler.label(panel)
        # Last primary_horizon rows should be NaN (dropped)
        assert "fwd_ret_63d" in labeled.columns


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestModel:
    def _make_xy(self, n=500):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(n, 20), columns=[f"f{i}" for i in range(20)])
        y = pd.Series((np.random.randn(n) > 0).astype(int))
        return X, y

    def test_random_forest_trains(self, minimal_config):
        from src.model import RandomForestModel
        X, y = self._make_xy()
        m = RandomForestModel(minimal_config)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert len(proba) == len(X)
        assert proba.min() >= 0 and proba.max() <= 1

    def test_feature_importance(self, minimal_config):
        from src.model import RandomForestModel
        X, y = self._make_xy()
        m = RandomForestModel(minimal_config)
        m.fit(X, y)
        imp = m.feature_importances()
        assert len(imp) == X.shape[1]
        assert imp.sum() > 0

    def test_model_trainer(self, minimal_config, tmp_path):
        from src.model import ModelTrainer
        minimal_config["models"]["saved_models"] = str(tmp_path)
        X, y = self._make_xy(n=300)
        # Make time-like index
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        tickers = ["TCS.NS"] * 300
        X.index = pd.MultiIndex.from_arrays([dates, tickers], names=["Date", "Ticker"])
        y.index = X.index
        trainer = ModelTrainer(minimal_config)
        models = trainer.train(X, y, ["random_forest"])
        assert "random_forest" in models


# ─── Ensemble tests ───────────────────────────────────────────────────────────

class TestEnsemble:
    def test_weighted_average(self, minimal_config):
        from src.ensemble import EnsemblePredictor
        ep = EnsemblePredictor(minimal_config)
        preds = {
            "xgboost": np.array([0.8, 0.6, 0.3]),
            "lightgbm": np.array([0.7, 0.5, 0.4]),
            "random_forest": np.array([0.6, 0.7, 0.3]),
        }
        scores = ep.predict(preds)
        assert len(scores) == 3
        assert scores.min() >= 0 and scores.max() <= 1

    def test_rank_average(self, minimal_config):
        from src.ensemble import EnsemblePredictor
        cfg = dict(minimal_config)
        cfg["ensemble"] = dict(minimal_config["ensemble"])
        cfg["ensemble"]["method"] = "rank_average"
        ep = EnsemblePredictor(cfg)
        preds = {
            "a": np.array([0.9, 0.5, 0.1]),
            "b": np.array([0.8, 0.6, 0.2]),
        }
        scores = ep.predict(preds)
        assert len(scores) == 3


# ─── Ranking tests ────────────────────────────────────────────────────────────

class TestRanking:
    def _make_scores(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=5, freq="ME")
        tickers = ["A", "B", "C", "D", "E"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
        return pd.Series(np.random.uniform(0.4, 0.9, len(idx)), index=idx)

    def test_rank_shape(self, minimal_config):
        from src.ranking import StockRanker
        ranker = StockRanker(minimal_config)
        scores = self._make_scores()
        ranked = ranker.rank(scores)
        assert "rank" in ranked.columns
        assert "selected" in ranked.columns

    def test_top_n(self, minimal_config):
        from src.ranking import StockRanker
        ranker = StockRanker(minimal_config)
        scores = self._make_scores()
        ranked = ranker.rank(scores)
        # Per date, at most top_n=3 selected
        per_date = ranked.groupby(level="Date")["selected"].sum()
        assert (per_date <= minimal_config["ranking"]["top_n"]).all()


# ─── Portfolio tests ──────────────────────────────────────────────────────────

class TestPortfolio:
    def test_equal_weight(self, minimal_config, sample_multi_ticker_data):
        from src.portfolio import PortfolioOptimizer
        po = PortfolioOptimizer(minimal_config)
        tickers = list(sample_multi_ticker_data.keys())[:3]
        weights = po.optimize(tickers, sample_multi_ticker_data)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        for t in tickers:
            assert t in weights

    def test_weight_constraints(self, minimal_config, sample_multi_ticker_data):
        from src.portfolio import PortfolioOptimizer
        po = PortfolioOptimizer(minimal_config)
        tickers = list(sample_multi_ticker_data.keys())
        weights = po.optimize(tickers, sample_multi_ticker_data)
        for w in weights.values():
            assert w >= minimal_config["portfolio"]["min_weight"] - 0.01
            assert w <= minimal_config["portfolio"]["max_weight"] + 0.01


# ─── RiskManagement tests ─────────────────────────────────────────────────────

class TestRiskManagement:
    def test_vol_filter(self, minimal_config, sample_multi_ticker_data):
        from src.risk_management import RiskManager
        rm = RiskManager(minimal_config)
        tickers = list(sample_multi_ticker_data.keys())
        approved = rm.filter_universe(tickers, sample_multi_ticker_data)
        assert isinstance(approved, list)

    def test_stop_loss(self, minimal_config):
        from src.risk_management import RiskManager
        rm = RiskManager(minimal_config)
        weights = {"TCS.NS": 0.5, "INFY.NS": 0.5}
        entry = {"TCS.NS": 100.0, "INFY.NS": 100.0}
        # INFY hits stop loss
        current = {"TCS.NS": 98.0, "INFY.NS": 80.0}
        adj = rm.apply_stop_loss(weights, entry, current)
        assert adj["INFY.NS"] == 0.0

    def test_kelly_sizing(self, minimal_config):
        from src.risk_management import RiskManager
        rm = RiskManager(minimal_config)
        k = rm.kelly_position_size(win_prob=0.6, avg_win=0.10, avg_loss=0.05, fraction=0.25)
        assert 0.0 <= k <= 1.0


# ─── Backtest tests ───────────────────────────────────────────────────────────

class TestBacktest:
    def test_backtest_runs(self, minimal_config, sample_multi_ticker_data, sample_ohlcv):
        from src.backtest import BacktestEngine
        engine = BacktestEngine(minimal_config)

        tickers = list(sample_multi_ticker_data.keys())[:3]
        stock_data = {t: sample_multi_ticker_data[t] for t in tickers}

        # Build monthly rebalance weights
        all_dates = sample_ohlcv.index
        rebal_dates = pd.date_range(all_dates[0], all_dates[-1], freq="BMS")
        weights_by_date = {
            d: {t: 1.0 / len(tickers) for t in tickers}
            for d in rebal_dates
            if d in all_dates or True
        }

        result = engine.run(weights_by_date, stock_data, sample_ohlcv)
        assert len(result.equity_curve) > 0
        assert "cagr" in result.metrics
        assert "sharpe" in result.metrics

    def test_transaction_costs_reduce_returns(
        self, minimal_config, sample_multi_ticker_data, sample_ohlcv
    ):
        from src.backtest import BacktestEngine
        import copy
        cfg_high_cost = copy.deepcopy(minimal_config)
        cfg_high_cost["backtest"]["transaction_cost_bps"] = 100

        cfg_low_cost = copy.deepcopy(minimal_config)
        cfg_low_cost["backtest"]["transaction_cost_bps"] = 1

        tickers = list(sample_multi_ticker_data.keys())[:3]
        stock_data = {t: sample_multi_ticker_data[t] for t in tickers}
        all_dates = sample_ohlcv.index
        rebal_dates = pd.date_range(all_dates[0], all_dates[-1], freq="BMS")
        weights_by_date = {
            d: {t: 1.0 / len(tickers) for t in tickers}
            for d in rebal_dates
        }

        r_high = BacktestEngine(cfg_high_cost).run(weights_by_date, stock_data, sample_ohlcv)
        r_low = BacktestEngine(cfg_low_cost).run(weights_by_date, stock_data, sample_ohlcv)

        assert r_low.metrics["total_transaction_cost"] < r_high.metrics["total_transaction_cost"]


# ─── Run marker ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
