"""
ensemble.py – Combine multiple model predictions via weighted average,
rank averaging, or stacking (logistic meta-model).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

log = get_logger(__name__)


class EnsemblePredictor:
    """
    Blends predictions from multiple base models.

    Supported methods:
      - weighted_average : weighted mean of probabilities
      - rank_average     : average of per-model percentile ranks
      - stacking         : logistic regression meta-model
    """

    def __init__(self, cfg: dict) -> None:
        ec = cfg["ensemble"]
        self.method: str = ec.get("method", "weighted_average")
        self.weights: Dict[str, float] = ec.get("weights", {})
        self.meta_model_type: str = ec.get("stacking_meta_model", "logistic")
        self._meta_model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._model_names: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_stacker(
        self,
        oof_preds: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> None:
        """
        Fit meta-model on out-of-fold predictions.
        Only used when method == 'stacking'.
        """
        if self.method != "stacking":
            return
        self._model_names = list(oof_preds.keys())
        X_meta = np.column_stack([oof_preds[n] for n in self._model_names])
        self._scaler = StandardScaler()
        X_meta = self._scaler.fit_transform(X_meta)
        self._meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._meta_model.fit(X_meta, y)
        log.info(f"Stacking meta-model trained on {X_meta.shape[0]} OOF samples")

    def predict(
        self,
        model_preds: Dict[str, np.ndarray],
        index: Optional[pd.Index] = None,
    ) -> pd.Series:
        """
        Blend model predictions into a single score series.

        Parameters
        ----------
        model_preds : {model_name: probability array}
        index       : optional pandas index for the output series

        Returns
        -------
        pd.Series of ensemble scores (higher = more likely to outperform)
        """
        if not model_preds:
            raise ValueError("No model predictions provided to ensemble")

        if self.method == "weighted_average":
            score = self._weighted_average(model_preds)
        elif self.method == "rank_average":
            score = self._rank_average(model_preds)
        elif self.method == "stacking":
            score = self._stacking_predict(model_preds)
        else:
            log.warning(f"Unknown ensemble method '{self.method}', falling back to simple mean")
            score = np.mean(list(model_preds.values()), axis=0)

        result = pd.Series(score, index=index, name="ensemble_score")
        log.debug(f"Ensemble score: mean={result.mean():.4f} std={result.std():.4f}")
        return result

    # ── Private methods ─────────────────────────────────────────────────────

    def _weighted_average(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        total_w = 0.0
        blended = np.zeros(len(next(iter(preds.values()))))
        for name, pred in preds.items():
            w = self.weights.get(name, 1.0 / len(preds))
            blended += w * np.asarray(pred)
            total_w += w
        return blended / (total_w + 1e-10)

    @staticmethod
    def _rank_average(preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert each model's scores to percentile ranks, then average."""
        n = len(next(iter(preds.values())))
        ranks = np.zeros(n)
        for pred in preds.values():
            arr = np.asarray(pred)
            # percentile rank
            from scipy.stats import rankdata
            ranks += rankdata(arr) / n
        return ranks / len(preds)

    def _stacking_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if self._meta_model is None:
            log.warning("Meta-model not fitted; falling back to simple mean")
            return np.mean(list(preds.values()), axis=0)
        names = self._model_names or list(preds.keys())
        X_meta = np.column_stack([preds[n] for n in names if n in preds])
        X_meta = self._scaler.transform(X_meta)
        return self._meta_model.predict_proba(X_meta)[:, 1]

    def combine_scores(
        self,
        ml_score: pd.Series,
        alpha_score: pd.Series,
        ml_weight: float = 0.7,
        alpha_weight: float = 0.3,
    ) -> pd.Series:
        """
        Combine ML-based scores with pure alpha factor scores.
        Both inputs should be normalised (0–1 range preferred).
        """
        # Percentile-normalise
        ml_norm = ml_score.rank(pct=True)
        alpha_norm = alpha_score.rank(pct=True)
        combined = ml_weight * ml_norm + alpha_weight * alpha_norm
        return combined.rename("combined_score")
