#!/usr/bin/env python3
"""
main.py – Entry point for the Quant LongTerm AI system.

Usage:
    python main.py                          # Full pipeline
    python main.py --config config.yaml     # Custom config
    python main.py --mode rank              # Ranking only (load saved models)
    python main.py --mode backtest          # Backtest only
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path as FilePath

from src.pipeline import QuantPipeline
from src.utils import get_logger, load_config

log = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quant LongTerm AI – Institutional-Grade Stock Prediction System"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "rank", "backtest", "data"],
        help="Pipeline mode (default: full)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log.info(f"Quant LongTerm AI | mode={args.mode} | config={args.config}")

    # Validate config file
    config_path = FilePath(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    try:
        if args.mode == "full":
            pipeline = QuantPipeline(config_path=str(config_path))
            result = pipeline.run()

            # Final output summary
            print("\n✅ Pipeline complete!")
            print(f"   CAGR      : {result.metrics.get('cagr', 0):.2%}")
            print(f"   Sharpe    : {result.metrics.get('sharpe', 0):.3f}")
            print(f"   Max DD    : {result.metrics.get('max_drawdown', 0):.2%}")
            print(f"   Alpha     : {result.metrics.get('alpha', 0):.2%}")
            print(f"\n📊 Charts saved to: results/")
            print(f"📁 Models saved to: models/saved_models/")

        elif args.mode == "data":
            from src.data_ingestion import DataIngestion
            from src.data_validation import DataValidator
            ing = DataIngestion(cfg)
            raw = ing.fetch_all()
            val = DataValidator(cfg)
            clean = val.validate(raw)
            log.info(f"Data mode: {len(clean)} tickers fetched and validated")

        elif args.mode == "rank":
            log.info("Rank mode: loading saved models and scoring latest data …")
            pipeline = QuantPipeline(config_path=str(config_path))
            # Fetch fresh data only
            pipeline.raw_data = pipeline.ingestion.fetch_all()
            pipeline.clean_data = pipeline.validator.validate(pipeline.raw_data)
            pipeline.aligned_data = pipeline.validator.align_panel(pipeline.clean_data)

            benchmark_ticker = cfg["data"].get("benchmark_ticker", cfg["data"].get("benchmark", "^NSEI"))
            stock_data = {k: v for k, v in pipeline.aligned_data.items()
                          if k != benchmark_ticker}

            feat_panel = pipeline.feature_eng.build_features(stock_data)
            feat_panel = pipeline.feature_eng.add_cross_sectional_features(feat_panel)
            feat_panel = pipeline.alpha_comp.compute(feat_panel)
            feat_panel = pipeline._merge_ohlcv(feat_panel, stock_data)

            X, y, feature_cols = pipeline._prepare_ml_data(feat_panel)

            # Load saved models
            import joblib
            model_dir = FilePath("models/saved_models")
            trained = {}
            for model_path in model_dir.glob("*.pkl"):
                try:
                    m = joblib.load(model_path)
                    trained[model_path.stem] = m
                    log.info(f"Loaded model: {model_path.stem}")
                except Exception as e:
                    log.warning(f"Could not load {model_path}: {e}")

            if not trained:
                log.error("No saved models found. Run 'full' mode first.")
                sys.exit(1)

            scores = pipeline._walk_forward_score(feat_panel, feature_cols, trained)
            latest = pipeline.ranker.latest_top_stocks(scores)
            print("\nTop ranked stocks:\n", latest.to_string(index=False))

        elif args.mode == "backtest":
            log.info("Backtest mode requires a pre-built feature panel.")
            log.warning("Please run 'full' mode first to generate features and models.")

    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
