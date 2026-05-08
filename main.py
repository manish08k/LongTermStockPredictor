"""
main.py – Entry point for the quant-longterm-ai NSE equities platform.

Usage:
    python main.py                              # uses configs/config.yaml (default)
    python main.py --rank-only                  # rank latest stocks only
    python main.py --config path/to/config.yaml # custom config path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline import QuantPipeline
from src.utils import get_logger

log = get_logger(__name__)

# Candidate config paths searched in order
_CONFIG_SEARCH_PATHS = [
    "configs/config.yaml",   # preferred: configs/ subfolder
    "config.yaml",           # legacy: project root
]


def resolve_config(user_path: str | None) -> Path:
    """
    Resolve the config file path.

    Priority:
    1. Explicit --config argument (error if not found)
    2. Auto-search _CONFIG_SEARCH_PATHS in order
    """
    # Explicit path supplied
    if user_path and user_path not in ("configs/config.yaml", "config.yaml"):
        p = Path(user_path)
        if not p.exists():
            log.error(f"Config file not found: {p}")
            sys.exit(1)
        return p

    # Try default search paths
    for candidate in _CONFIG_SEARCH_PATHS:
        p = Path(candidate)
        if p.exists():
            log.info(f"Using config: {p}")
            return p

    # Nothing found – give a helpful message
    log.error(
        "No config file found. Searched:\n"
        + "\n".join(f"  • {c}" for c in _CONFIG_SEARCH_PATHS)
        + "\n\nRun:  cp configs/config.yaml . "
        "  (or use --config path/to/config.yaml)"
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quant LongTerm AI – NSE Indian Equities Platform"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config YAML. "
            "Defaults to configs/config.yaml, then config.yaml."
        ),
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Only compute latest stock rankings (no backtest)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)

    pipeline = QuantPipeline(config_path=str(config_path))

    if args.rank_only:
        log.info("Running ranking-only mode …")
        top_stocks = pipeline.rank_stocks_now()
        print(top_stocks.to_string(index=False))
    else:
        log.info("Running full pipeline …")
        result = pipeline.run()
        log.info(f"Final portfolio value: ₹{result.metrics.get('final_value', 0):,.0f}")
        log.info(f"CAGR:   {result.metrics.get('cagr', 0):.2%}")
        log.info(f"Sharpe: {result.metrics.get('sharpe', 0):.3f}")


if __name__ == "__main__":
    main()