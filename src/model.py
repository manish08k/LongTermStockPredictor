"""
model.py – Unified model interface: XGBoost, LightGBM, CatBoost, RandomForest,
ExtraTrees. Walk-forward purged CV, Optuna tuning, SHAP explainability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

from src.utils import get_logger, timeit, set_random_seed

log = get_logger(__name__)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    log.debug("CatBoost not installed; skipping")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.debug("Optuna not installed; skipping hyperparameter tuning")


# ── Base model interface ──────────────────────────────────────────────────────

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
        mc = cfg["models"].get("xgboost", {})
        self.params = {
            "n_estimators": mc.get("n_estimators", 600),
            "max_depth": mc.get("max_depth", 5),
            "learning_rate": mc.get("learning_rate", 0.03),
            "subsample": mc.get("subsample", 0.75),
            "colsample_bytree": mc.get("colsample_bytree", 0.75),
            "min_child_weight": mc.get("min_child_weight", 10),
            "gamma": mc.get("gamma", 0.2),
            "reg_alpha": mc.get("reg_alpha", 0.1),
            "reg_lambda": mc.get("reg_lambda", 2.0),
            "eval_metric": mc.get("eval_metric", "auc"),
            "tree_method": mc.get("tree_method", "hist"),
            "random_state": 42,
        }
        self._feature_names: List[str] = []
        self.model: Optional[xgb.XGBClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"].get("lightgbm", {})
        self.params = {
            "n_estimators": mc.get("n_estimators", 600),
            "max_depth": mc.get("max_depth", 5),
            "learning_rate": mc.get("learning_rate", 0.03),
            "num_leaves": mc.get("num_leaves", 31),
            "subsample": mc.get("subsample", 0.75),
            "colsample_bytree": mc.get("colsample_bytree", 0.75),
            "min_child_samples": mc.get("min_child_samples", 30),
            "reg_alpha": mc.get("reg_alpha", 0.1),
            "reg_lambda": mc.get("reg_lambda", 2.0),
            "verbose": -1,
            "random_state": 42,
        }
        self._feature_names: List[str] = []
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── CatBoost ──────────────────────────────────────────────────────────────────

class CatBoostModel(BaseModel):
    name = "catboost"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"].get("catboost", {})
        self.params = {
            "iterations": mc.get("n_estimators", 600),
            "depth": mc.get("max_depth", 5),
            "learning_rate": mc.get("learning_rate", 0.03),
            "l2_leaf_reg": mc.get("reg_lambda", 3.0),
            "random_seed": 42,
            "verbose": False,
            "eval_metric": "AUC",
        }
        self._feature_names: List[str] = []
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed")
        self._feature_names = list(X.columns)
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.get_feature_importance(), index=self._feature_names
        ).sort_values(ascending=False)


# ── Random Forest ─────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"].get("random_forest", {})
        self.params = {
            "n_estimators": mc.get("n_estimators", 400),
            "max_depth": mc.get("max_depth", 7),
            "min_samples_leaf": mc.get("min_samples_leaf", 30),
            "max_features": mc.get("max_features", 0.4),
            "n_jobs": mc.get("n_jobs", -1),
            "random_state": 42,
        }
        self._feature_names: List[str] = []
        self.model: Optional[RandomForestClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── Extra Trees ───────────────────────────────────────────────────────────────

class ExtraTreesModel(BaseModel):
    name = "extra_trees"

    def __init__(self, cfg: dict) -> None:
        mc = cfg["models"].get("extra_trees", {})
        self.params = {
            "n_estimators": mc.get("n_estimators", 400),
            "max_depth": mc.get("max_depth", 8),
            "min_samples_leaf": mc.get("min_samples_leaf", 25),
            "max_features": mc.get("max_features", 0.5),
            "n_jobs": -1,
            "random_state": 42,
        }
        self._feature_names: List[str] = []
        self.model: Optional[ExtraTreesClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.model = ExtraTreesClassifier(**self.params)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self._feature_names if c in X.columns]
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── Registry & Factory ─────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
    "extra_trees": ExtraTreesModel,
}
if HAS_CATBOOST:
    MODEL_REGISTRY["catboost"] = CatBoostModel


def build_model(name: str, cfg: dict) -> BaseModel:
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return cls(cfg)


# ── Walk-forward purged cross-validation ──────────────────────────────────────

class PurgedWalkForwardCV:
    """
    Purged walk-forward cross-validation with expanding or rolling window.

    Parameters
    ----------
    train_window : number of periods in training window (0 = expanding)
    test_window  : number of periods in each test fold
    gap          : purge gap between train and test (avoids leakage)
    n_splits     : maximum number of splits
    expanding    : use expanding window (True) or rolling window (False)
    """

    def __init__(
        self,
        train_window: int = 504,
        test_window: int = 63,
        gap: int = 21,
        n_splits: int = 5,
        expanding: bool = True,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.gap = gap
        self.n_splits = n_splits
        self.expanding = expanding

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (train_idx, test_idx) index arrays.
        X must be sorted by time.
        """
        n = len(X)
        splits = []

        # Compute test start points
        # First test starts after minimum train window
        first_test_start = self.train_window + self.gap
        if first_test_start >= n:
            log.warning("Not enough data for walk-forward CV; using single split")
            mid = n // 2
            return [(np.arange(0, mid), np.arange(mid + self.gap, n))]

        # Generate test windows from the end
        test_ends = []
        end = n
        for _ in range(self.n_splits):
            test_start = end - self.test_window
            if test_start <= first_test_start:
                break
            test_ends.append((test_start, end))
            end = test_start - self.gap

        test_ends = list(reversed(test_ends))

        for test_start, test_end in test_ends:
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.gap - self.train_window)
            train_end = test_start - self.gap

            if train_end <= train_start:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        log.info(f"Walk-forward CV: {len(splits)} folds")
        return splits


# ── Model Trainer ─────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains models with purged walk-forward cross-validation.
    Supports Optuna hyperparameter tuning and SHAP explainability.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        wf = cfg["models"].get("walk_forward", {})
        self.wf_cv = PurgedWalkForwardCV(
            train_window=wf.get("train_window", 504),
            test_window=wf.get("test_window", 63),
            gap=wf.get("gap", 21),
            n_splits=cfg["models"].get("cv_splits", 5),
            expanding=wf.get("expanding_window", True),
        )
        self.top_n_features: int = cfg["models"].get("feature_importance_top_n", 40)
        self.save_dir: Path = Path(
            cfg["models"].get("saved_models", "models/saved_models")
        )
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
        Train all models with walk-forward CV, then final fit on all data.
        """
        names = model_names or self.cfg["models"].get(
            "use_models", ["xgboost", "lightgbm"]
        )
        # Filter to available models
        names = [n for n in names if n in MODEL_REGISTRY]
        log.info(f"Training: {names} | Walk-forward folds: {self.wf_cv.n_splits}")

        X = X.sort_index()
        y = y.loc[X.index]
        X = self._clean_features(X)

        # Walk-forward cross-validation
        cv_scores = self._cross_validate(X, y, names)
        self._log_cv_scores(cv_scores)

        # Final fit on full data
        trained: Dict[str, BaseModel] = {}
        for name in names:
            log.info(f"Final fit: {name} on {len(X)} samples …")
            try:
                model = build_model(name, self.cfg)
                model.fit(X, y)
                trained[name] = model
                model.save(self.save_dir / f"{name}.pkl")

                # Log top features
                try:
                    imp = model.feature_importances()
                    top = imp.head(self.top_n_features)
                    log.info(f"Top-10 features [{name}]:\n{top.head(10).to_string()}")
                except Exception:
                    pass
            except Exception as e:
                log.error(f"Training failed for {name}: {e}")

        return trained

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model_names: List[str]
    ) -> Dict[str, List[float]]:
        """Purged walk-forward AUC per fold."""
        splits = self.wf_cv.split(X)
        scores: Dict[str, List[float]] = {n: [] for n in model_names}

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                log.debug(f"Fold {fold_i}: single class, skipping")
                continue

            for name in model_names:
                try:
                    m = build_model(name, self.cfg)
                    m.fit(X_tr, y_tr)
                    prob = m.predict_proba(X_te)
                    auc = roc_auc_score(y_te, prob)
                    scores[name].append(auc)
                    log.debug(f"Fold {fold_i} [{name}] AUC={auc:.4f}")
                except Exception as e:
                    log.warning(f"Fold {fold_i} [{name}] failed: {e}")

        return scores

    @staticmethod
    def _log_cv_scores(scores: Dict[str, List[float]]) -> None:
        log.info("── Walk-Forward CV Results ──")
        for name, aucs in scores.items():
            if aucs:
                log.info(
                    f"  {name:20s} AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
                    f"| folds={[f'{a:.3f}' for a in aucs]}"
                )
            else:
                log.warning(f"  {name}: no valid folds")

    @staticmethod
    def _clean_features(
        X: pd.DataFrame, max_null_pct: float = 0.30
    ) -> pd.DataFrame:
        """Drop high-null columns, fill remaining NaNs."""
        null_pct = X.isna().mean()
        drop_cols = null_pct[null_pct > max_null_pct].index.tolist()
        if drop_cols:
            log.info(f"Dropping {len(drop_cols)} high-null feature columns")
            X = X.drop(columns=drop_cols)
        return X.fillna(0).replace([np.inf, -np.inf], 0)

    def predict(
        self,
        models: Dict[str, BaseModel],
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Generate predictions from all trained models."""
        X = self._clean_features(X)
        preds: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            try:
                preds[name] = model.predict_proba(X)
                log.debug(
                    f"[{name}] score: mean={preds[name].mean():.4f} "
                    f"std={preds[name].std():.4f}"
                )
            except Exception as e:
                log.error(f"Prediction failed [{name}]: {e}")
        return preds

    def get_shap_values(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        background_n: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Compute SHAP values for a trained model.
        Returns DataFrame of shape (n_samples, n_features).
        """
        try:
            import shap
            X_clean = self._clean_features(X.copy())
            background = X_clean.sample(min(background_n, len(X_clean)), random_state=42)

            if isinstance(model, (XGBoostModel, LightGBMModel)):
                explainer = shap.TreeExplainer(model.model, background)
            elif isinstance(model, (RandomForestModel, ExtraTreesModel)):
                explainer = shap.TreeExplainer(model.model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, background)

            shap_vals = explainer.shap_values(X_clean)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # positive class

            return pd.DataFrame(shap_vals, index=X_clean.index, columns=X_clean.columns)

        except Exception as e:
            log.warning(f"SHAP computation failed: {e}")
            return None