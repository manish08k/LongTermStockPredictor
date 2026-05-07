"""
model.py – Unified model interface for XGBoost, LightGBM, RandomForest.
Supports time-series cross-validation, feature importance, and persistence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

from src.utils import get_logger, timeit, set_random_seed

log = get_logger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

class BaseModel:
    name: str = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def feature_importances(self) -> pd.Series:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        log.info(f"Saved {self.name} → {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        return joblib.load(path)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"]["xgboost"]
        self.params = {
            "n_estimators": mc.get("n_estimators", 500),
            "max_depth": mc.get("max_depth", 6),
            "learning_rate": mc.get("learning_rate", 0.05),
            "subsample": mc.get("subsample", 0.8),
            "colsample_bytree": mc.get("colsample_bytree", 0.8),
            "min_child_weight": mc.get("min_child_weight", 5),
            "gamma": mc.get("gamma", 0.1),
            "reg_alpha": mc.get("reg_alpha", 0.1),
            "reg_lambda": mc.get("reg_lambda", 1.0),
            "eval_metric": mc.get("eval_metric", "auc"),
            "tree_method": mc.get("tree_method", "hist"),
            "random_state": 42,
            "use_label_encoder": False,
        }
        self.early_stopping = cfg["models"].get("early_stopping_rounds", 50)
        self.model: Optional[xgb.XGBClassifier] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self._feature_names])[:, 1]

    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"]["lightgbm"]
        self.params = {
            "n_estimators": mc.get("n_estimators", 500),
            "max_depth": mc.get("max_depth", 6),
            "learning_rate": mc.get("learning_rate", 0.05),
            "num_leaves": mc.get("num_leaves", 63),
            "subsample": mc.get("subsample", 0.8),
            "colsample_bytree": mc.get("colsample_bytree", 0.8),
            "min_child_samples": mc.get("min_child_samples", 20),
            "reg_alpha": mc.get("reg_alpha", 0.1),
            "reg_lambda": mc.get("reg_lambda", 1.0),
            "verbose": -1,
            "random_state": 42,
        }
        self.early_stopping = cfg["models"].get("early_stopping_rounds", 50)
        self.model: Optional[lgb.LGBMClassifier] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X, y,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self._feature_names])[:, 1]

    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)


# ── Random Forest ─────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"]["random_forest"]
        self.params = {
            "n_estimators": mc.get("n_estimators", 300),
            "max_depth": mc.get("max_depth", 8),
            "min_samples_leaf": mc.get("min_samples_leaf", 20),
            "max_features": mc.get("max_features", 0.5),
            "n_jobs": mc.get("n_jobs", -1),
            "random_state": 42,
        }
        self.model: Optional[RandomForestClassifier] = None
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self._feature_names])[:, 1]

    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)


# ── Model factory ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
}


def build_model(name: str, cfg: dict) -> BaseModel:
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return cls(cfg)


# ── Trainer ──────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains models with time-series cross-validation.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cv_splits: int = cfg["models"].get("cv_splits", 5)
        self.cv_gap: int = cfg["models"].get("cv_gap", 21)
        self.top_n_features: int = cfg["models"].get("feature_importance_top_n", 30)
        self.save_dir: Path = Path(cfg["models"].get("saved_models", "models/saved_models")) if "saved_models" in cfg.get("models", {}) else Path("models/saved_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        set_random_seed(cfg.get("project", {}).get("random_seed", 42))

    @timeit
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, BaseModel]:
        """
        Train all configured models with time-series CV.

        Parameters
        ----------
        X : feature matrix (float), index must be sortable (Date, Ticker)
        y : binary labels

        Returns
        -------
        dict[model_name -> fitted BaseModel]
        """
        names = model_names or self.cfg["models"].get("use_models", ["xgboost", "lightgbm"])
        log.info(f"Training models: {names} | CV splits: {self.cv_splits}")

        # Ensure sorted by date
        X = X.sort_index()
        y = y.loc[X.index]

        # Remove feature columns with too many NaNs
        X = self._clean_features(X)

        # Time-series cross-validation
        cv_scores = self._cross_validate(X, y, names)
        self._log_cv_scores(cv_scores)

        # Final fit on all data
        trained: Dict[str, BaseModel] = {}
        for name in names:
            log.info(f"Final training: {name} on {len(X)} samples …")
            model = build_model(name, self.cfg)
            model.fit(X, y)
            trained[name] = model

            # Save
            save_path = self.save_dir / f"{name}.pkl"
            model.save(save_path)

            # Feature importance
            try:
                imp = model.feature_importances()
                top = imp.head(self.top_n_features)
                log.info(f"Top features [{name}]:\n{top.to_string()}")
            except Exception as e:
                log.debug(f"Feature importance failed for {name}: {e}")

        return trained

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model_names: List[str]
    ) -> Dict[str, List[float]]:
        """Time-series split CV returning AUC per fold."""
        tscv = TimeSeriesSplit(n_splits=self.cv_splits, gap=self.cv_gap)
        scores: Dict[str, List[float]] = {n: [] for n in model_names}

        # Use flat integer positions for CV (time-ordered)
        X_vals = X.values
        y_vals = y.values

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_vals)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_tr.nunique() < 2 or y_val.nunique() < 2:
                log.debug(f"Fold {fold}: skipping (single class)")
                continue

            for name in model_names:
                try:
                    m = build_model(name, self.cfg)
                    m.fit(X_tr, y_tr)
                    proba = m.predict_proba(X_val)
                    auc = roc_auc_score(y_val, proba)
                    scores[name].append(auc)
                except Exception as e:
                    log.warning(f"CV fold {fold} failed for {name}: {e}")

        return scores

    @staticmethod
    def _log_cv_scores(scores: Dict[str, List[float]]) -> None:
        for name, aucs in scores.items():
            if aucs:
                log.info(
                    f"CV AUC [{name}]: mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
                    f"| folds={aucs}"
                )

    @staticmethod
    def _clean_features(X: pd.DataFrame, max_null_pct: float = 0.30) -> pd.DataFrame:
        """Drop columns with too many nulls, then fill remaining."""
        null_pct = X.isna().mean()
        drop_cols = null_pct[null_pct > max_null_pct].index.tolist()
        if drop_cols:
            log.info(f"Dropping {len(drop_cols)} high-null features")
            X = X.drop(columns=drop_cols)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X

    def predict(
        self,
        models: Dict[str, BaseModel],
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all trained models.
        """
        X = self._clean_features(X)
        preds: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            try:
                preds[name] = model.predict_proba(X)
            except Exception as e:
                log.error(f"Prediction failed for {name}: {e}")
        return preds
