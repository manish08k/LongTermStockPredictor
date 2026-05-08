"""
ensemble.py – Blends model predictions via weighted average, rank average,
or stacking. Also combines ML scores with pure alpha factor scores.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from src.utils import get_logger

log = get_logger(__name__)


class EnsemblePredictor:
    """
    Blends predictions from multiple base models into a single score.

    Methods:
      weighted_average : weighted mean of model probabilities
      rank_average     : average of per-model percentile ranks
      stacking         : meta-model (logistic) trained on OOF predictions
    """

    def __init__(self, cfg: dict) -> None:
        ec = cfg.get("ensemble", {})
        self.method: str = ec.get("method", "weighted_average")
        self.weights: Dict[str, float] = ec.get("weights", {})
        self.ml_weight: float = ec.get("ml_weight", 0.70)
        self.alpha_weight: float = ec.get("alpha_weight", 0.30)
        self._meta_model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._model_names: List[str] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(
        self,
        model_preds: Dict[str, np.ndarray],
        index: Optional[pd.Index] = None,
    ) -> pd.Series:
        """
        Blend model predictions into a single ensemble score.

        Parameters
        ----------
        model_preds : {model_name: probability array}
        index       : pandas index for the result series

        Returns
        -------
        pd.Series of ensemble scores (higher = more likely outperform)
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
            log.warning(f"Unknown method '{self.method}'; using simple mean")
            score = np.mean(list(model_preds.values()), axis=0)

        result = pd.Series(score, index=index, name="ensemble_score")
        log.debug(
            f"Ensemble [{self.method}]: "
            f"mean={result.mean():.4f} std={result.std():.4f} "
            f"n={len(result)}"
        )
        return result

    def combine_ml_alpha(
        self,
        ml_score: pd.Series,
        alpha_score: pd.Series,
    ) -> pd.Series:
        """
        Combine ML-based scores with pure alpha factor composite scores.
        Both are percentile-ranked before combining.
        """
        # Percentile-normalise within each date cross-section
        ml_norm = ml_score.groupby(level="Date").rank(pct=True)
        alpha_norm = alpha_score.groupby(level="Date").rank(pct=True)
        combined = self.ml_weight * ml_norm + self.alpha_weight * alpha_norm
        log.info(
            f"Combined score: ML×{self.ml_weight} + Alpha×{self.alpha_weight} | "
            f"mean={combined.mean():.4f} std={combined.std():.4f}"
        )
        return combined.rename("combined_score")

    def fit_stacker(
        self,
        oof_preds: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> None:
        """Fit logistic meta-model on OOF predictions."""
        if self.method != "stacking":
            return
        self._model_names = list(oof_preds.keys())
        X_meta = np.column_stack([oof_preds[n] for n in self._model_names])
        self._scaler = StandardScaler()
        X_meta = self._scaler.fit_transform(X_meta)
        self._meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._meta_model.fit(X_meta, y)
        log.info(f"Stacking meta-model fitted on {len(y)} OOF samples")

    # ── Private methods ────────────────────────────────────────────────────

    def _weighted_average(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        n_samples = len(next(iter(preds.values())))
        blended = np.zeros(n_samples)
        total_w = 0.0
        for name, pred in preds.items():
            w = self.weights.get(name, 1.0 / len(preds))
            blended += w * np.asarray(pred)
            total_w += w
        return blended / (total_w + 1e-10)

    @staticmethod
    def _rank_average(preds: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(preds.values())))
        ranks = np.zeros(n)
        for pred in preds.values():
            ranks += rankdata(np.asarray(pred)) / n
        return ranks / len(preds)

    def _stacking_predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if self._meta_model is None:
            log.warning("Meta-model not fitted; using weighted average")
            return self._weighted_average(preds)
        names = self._model_names or list(preds.keys())
        X_meta = np.column_stack([preds[n] for n in names if n in preds])
        X_meta = self._scaler.transform(X_meta)
        return self._meta_model.predict_proba(X_meta)[:, 1]