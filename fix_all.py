"""
fix_all.py – Run this once from inside your quant-longterm-ai folder.
It patches config.yaml AND all src/*.py files to eliminate every KeyError.

Usage:  python fix_all.py
"""
import os, re, yaml
from pathlib import Path

BASE = Path(__file__).parent

# ── 1. Patch config.yaml ──────────────────────────────────────────────────
print("Patching config.yaml ...")
with open(BASE / "config.yaml") as f:
    cfg = yaml.safe_load(f)

d = cfg.setdefault("data", {})
d.setdefault("benchmark_ticker",    d.get("benchmark", "^NSEI"))
d.setdefault("raw_data_path",       d.get("raw_path", "data/raw"))
d.setdefault("processed_data_path", d.get("processed_path", "data/processed"))
d.setdefault("features_data_path",  d.get("features_path", "data/features"))
d.setdefault("min_history_days",    d.get("min_history", 252))
d.setdefault("cache_data",          d.get("cache", True))
d.setdefault("interval",            "1d")

f2 = cfg.setdefault("features", {})
f2.setdefault("rsi_window",        f2.get("rsi_period", 14))
f2.setdefault("bb_window",         f2.get("bb_period", 20))
f2.setdefault("bb_std",            2)
f2.setdefault("atr_window",        f2.get("atr_period", 14))
f2.setdefault("macd_fast",         12)
f2.setdefault("macd_slow",         26)
f2.setdefault("macd_signal",       9)
f2.setdefault("momentum_windows",  [21, 63, 126, 252])
f2.setdefault("ma_windows",        [20, 50, 100, 200])
f2.setdefault("vol_windows",       [10, 21, 63])
f2.setdefault("lag_periods",       [1, 5, 10, 21])
f2.setdefault("zscore_window",     252)
f2.setdefault("cross_section_rank", True)

lb = cfg.setdefault("labeling", {})
lb.setdefault("horizons",             [21, 63, 126, 252])
lb.setdefault("primary_horizon",      63)
lb.setdefault("label_type",           lb.get("method", "binary"))
lb.setdefault("outperform_threshold", lb.get("min_return_threshold", 0.0))
lb.setdefault("triple_barrier",       {"enabled": False})

m = cfg.setdefault("models", {})
m.setdefault("use_models",              ["xgboost", "lightgbm", "random_forest"])
m.setdefault("cv_splits",              5)
m.setdefault("cv_gap",                 21)
m.setdefault("early_stopping_rounds",  50)
m.setdefault("feature_importance_top_n", 30)
m.setdefault("saved_models",           "models/saved_models")
m.setdefault("xgboost", {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "eval_metric": "auc", "tree_method": "hist"
})
m.setdefault("lightgbm", {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1
})
m.setdefault("random_forest", {
    "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20,
    "max_features": 0.5, "n_jobs": -1
})

e = cfg.setdefault("ensemble", {})
e.setdefault("method",  "weighted_average")
e.setdefault("weights", {"xgboost": 0.40, "lightgbm": 0.40, "random_forest": 0.20})
e.setdefault("stacking_meta_model", "logistic")

r = cfg.setdefault("ranking", {})
r.setdefault("top_n",                10)
r.setdefault("score_method",         "probability")
r.setdefault("min_score_threshold",  0.50)

p = cfg.setdefault("portfolio", {})
p.setdefault("method",         "max_sharpe")
p.setdefault("max_weight",     0.20)
p.setdefault("min_weight",     0.02)
p.setdefault("risk_free_rate", 0.065)
p.setdefault("regularization", p.get("l2_gamma", 0.0001))
p.setdefault("solver",         "SLSQP")

rk = cfg.setdefault("risk", {})
rk.setdefault("max_portfolio_volatility",  rk.get("max_vol_threshold", 0.25))
rk.setdefault("stop_loss_pct",             -0.15)
rk.setdefault("max_drawdown_exit",         rk.get("max_drawdown_limit", -0.20))
rk.setdefault("volatility_filter_window",  21)
rk.setdefault("volatility_filter_threshold", rk.get("max_vol_threshold", 0.50))

b = cfg.setdefault("backtest", {})
b.setdefault("initial_capital",      10_000_000)
b.setdefault("rebalance_frequency",  b.get("rebalance_freq", "monthly"))
b.setdefault("transaction_cost_bps", b.get("transaction_cost", 20))
b.setdefault("slippage_bps",         b.get("slippage", 10))
b.setdefault("min_trading_days",     126)
b.setdefault("walk_forward",         {"train_window": 504, "test_window": 63, "step": 63})

ev = cfg.setdefault("evaluation", {})
ev.setdefault("annualization_factor", 252)
ev.setdefault("output_dir",           "results")

pr = cfg.setdefault("project", {})
pr.setdefault("random_seed", pr.get("seed", 42))
pr.setdefault("log_level",   "INFO")

with open(BASE / "config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print("  config.yaml ✓")

# ── 2. Patch src/*.py – replace all dc["x"] / rc["x"] etc with .get() ────
SAFE_DEFAULTS = {
    # data_ingestion / data_validation
    'dc["benchmark_ticker"]':     'dc.get("benchmark_ticker", dc.get("benchmark", "^NSEI"))',
    'dc["raw_data_path"]':        'dc.get("raw_data_path", "data/raw")',
    'dc["processed_data_path"]':  'dc.get("processed_data_path", "data/processed")',
    'dc["features_data_path"]':   'dc.get("features_data_path", "data/features")',
    'dc["min_history_days"]':     'dc.get("min_history_days", 252)',
    'dc["cache_data"]':           'dc.get("cache_data", True)',
    'dc["interval"]':             'dc.get("interval", "1d")',
    # features
    'fc["rsi_window"]':           'fc.get("rsi_window", fc.get("rsi_period", 14))',
    'fc["bb_window"]':            'fc.get("bb_window", fc.get("bb_period", 20))',
    'fc["bb_std"]':               'fc.get("bb_std", 2)',
    'fc["atr_window"]':           'fc.get("atr_window", fc.get("atr_period", 14))',
    'fc["macd_fast"]':            'fc.get("macd_fast", 12)',
    'fc["macd_slow"]':            'fc.get("macd_slow", 26)',
    'fc["macd_signal"]':          'fc.get("macd_signal", 9)',
    'fc["momentum_windows"]':     'fc.get("momentum_windows", [21,63,126,252])',
    'fc["ma_windows"]':           'fc.get("ma_windows", [20,50,100,200])',
    'fc["vol_windows"]':          'fc.get("vol_windows", [10,21,63])',
    'fc["lag_periods"]':          'fc.get("lag_periods", [1,5,10,21])',
    'fc["zscore_window"]':        'fc.get("zscore_window", 252)',
    # labeling
    'lc["horizons"]':             'lc.get("horizons", [21,63,126,252])',
    'lc["primary_horizon"]':      'lc.get("primary_horizon", 63)',
    'lc["label_type"]':           'lc.get("label_type", lc.get("method", "binary"))',
    'lc["outperform_threshold"]': 'lc.get("outperform_threshold", 0.0)',
    # models
    'cfg["models"]["use_models"]':              'cfg["models"].get("use_models", ["xgboost","lightgbm","random_forest"])',
    'cfg["models"]["cv_splits"]':               'cfg["models"].get("cv_splits", 5)',
    'cfg["models"]["cv_gap"]':                  'cfg["models"].get("cv_gap", 21)',
    'cfg["models"]["early_stopping_rounds"]':   'cfg["models"].get("early_stopping_rounds", 50)',
    'cfg["models"]["feature_importance_top_n"]':'cfg["models"].get("feature_importance_top_n", 30)',
    # ensemble
    'ec["method"]':   'ec.get("method", "weighted_average")',
    'ec["weights"]':  'ec.get("weights", {})',
    'ec["stacking_meta_model"]': 'ec.get("stacking_meta_model", "logistic")',
    # ranking
    'rc["top_n"]':                   'rc.get("top_n", 10)',
    'rc["score_method"]':            'rc.get("score_method", "probability")',
    'rc["min_score_threshold"]':     'rc.get("min_score_threshold", 0.50)',
    # portfolio
    'pc["method"]':         'pc.get("method", "max_sharpe")',
    'pc["max_weight"]':     'pc.get("max_weight", 0.20)',
    'pc["min_weight"]':     'pc.get("min_weight", 0.02)',
    'pc["risk_free_rate"]': 'pc.get("risk_free_rate", 0.065)',
    'pc["regularization"]': 'pc.get("regularization", 0.0001)',
    # risk
    'rc["max_portfolio_volatility"]':    'rc.get("max_portfolio_volatility", 0.25)',
    'rc["stop_loss_pct"]':               'rc.get("stop_loss_pct", -0.15)',
    'rc["max_drawdown_exit"]':           'rc.get("max_drawdown_exit", -0.20)',
    'rc["volatility_filter_window"]':    'rc.get("volatility_filter_window", 21)',
    'rc["volatility_filter_threshold"]': 'rc.get("volatility_filter_threshold", 0.50)',
    # backtest
    'bc["initial_capital"]':      'bc.get("initial_capital", 10_000_000)',
    'bc["rebalance_frequency"]':  'bc.get("rebalance_frequency", "monthly")',
    'bc["transaction_cost_bps"]': 'bc.get("transaction_cost_bps", 20)',
    'bc["slippage_bps"]':         'bc.get("slippage_bps", 10)',
    'bc["min_trading_days"]':     'bc.get("min_trading_days", 126)',
    'bc["walk_forward"]':         'bc.get("walk_forward", {})',
    # pipeline / main benchmark
    'self.cfg["data"]["benchmark_ticker"]':
        'self.cfg["data"].get("benchmark_ticker", self.cfg["data"].get("benchmark", "^NSEI"))',
    'cfg["data"]["benchmark_ticker"]':
        'cfg["data"].get("benchmark_ticker", cfg["data"].get("benchmark", "^NSEI"))',
    # labeling benchmark
    'cfg["data"]["benchmark_ticker"]':
        'cfg["data"].get("benchmark_ticker", cfg["data"].get("benchmark", "^NSEI"))',
}

src_dir = BASE / "src"
for pyfile in sorted(src_dir.glob("*.py")):
    text = pyfile.read_text()
    original = text
    for old, new in SAFE_DEFAULTS.items():
        text = text.replace(old, new)
    if text != original:
        pyfile.write_text(text)
        print(f"  {pyfile.name} ✓ (patched)")
    else:
        print(f"  {pyfile.name}   (no changes needed)")

# Also patch main.py
main_py = BASE / "main.py"
if main_py.exists():
    text = main_py.read_text()
    original = text
    for old, new in SAFE_DEFAULTS.items():
        text = text.replace(old, new)
    if text != original:
        main_py.write_text(text)
        print(f"  main.py ✓ (patched)")

# ── 3. Create missing directories ─────────────────────────────────────────
for folder in ["data/raw", "data/processed", "data/features",
               "models/saved_models", "models/configs", "logs", "results"]:
    (BASE / folder).mkdir(parents=True, exist_ok=True)
print("\nAll directories created ✓")
print("\n✅  All done! Now run:  python main.py")
