"""
Model training and causal probability calibration routines for financial fraud classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def build_logistic_regression(
    numeric_cols: List[str],
    categorical_cols: List[str] = ["type"],
    class_weight: Optional[Union[str, Dict]] = None,
    max_iter: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """Builds a scikit-learn Pipeline with ColumnTransformer for Logistic Regression."""
    if class_weight == "balanced":
        logger.warning(
            "class_weight='balanced' detected: minority class is artificially upweighted. "
            "This shifts the decision boundary intercept and inflates raw predicted probabilities "
            "toward 0.5, severely distorting calibration and Brier score."
        )

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    class_weight: Optional[str] = None,
    max_iter: int = 200,
    random_state: int = 42,
) -> Pipeline:
    """Fits a standardized Logistic Regression baseline on training features."""
    cat_cols = [col for col in X_train.columns if str(X_train[col].dtype) == "category" or col == "type"]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    pipeline = build_logistic_regression(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 25,
    eval_metric: str = "binary_logloss",
) -> lgb.LGBMClassifier:
    """Trains a LightGBM Classifier with early stopping monitored on chronological validation data."""
    lgb_params = DEFAULT_LGBM_PARAMS.copy()
    if params:
        lgb_params.update(params)

    cat_cols = [col for col in X_train.columns if str(X_train[col].dtype) == "category"]
    model = lgb.LGBMClassifier(**lgb_params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=callbacks,
    )
    return model


class CausalCalibrator(BaseEstimator, ClassifierMixin):
    """
    Causal 1D probability calibrator mapping raw base model scores to calibrated posteriors.
    Fitted strictly on a dedicated historical calibration split without K-fold lookahead.
    """

    def __init__(self, base_estimator: Any, method: str = "isotonic"):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator_: Any = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X_cal: Union[pd.DataFrame, np.ndarray],
        y_cal: Union[pd.Series, np.ndarray],
    ) -> CausalCalibrator:
        """Fits a 1D mapping from raw probabilities to empirical class frequencies."""
        y_cal = np.asarray(y_cal)
        p_raw = self.base_estimator.predict_proba(X_cal)[:, 1]
        p_clipped = np.clip(p_raw, 1e-12, 1.0 - 1e-12)

        if self.method == "sigmoid":
            # Platt scaling: LogisticRegression on log-odds
            logits = logit(p_clipped).reshape(-1, 1)
            self.calibrator_ = LogisticRegression(solver="lbfgs", random_state=42)
            self.calibrator_.fit(logits, y_cal)
        elif self.method == "isotonic":
            # Non-parametric isotonic regression
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator_.fit(p_clipped, y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms uncalibrated base probabilities into calibrated posterior probabilities."""
        p_raw = self.base_estimator.predict_proba(X)[:, 1]
        p_clipped = np.clip(p_raw, 1e-12, 1.0 - 1e-12)

        if self.method == "sigmoid":
            logits = logit(p_clipped).reshape(-1, 1)
            p_cal = self.calibrator_.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            p_cal = self.calibrator_.predict(p_clipped)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Generates binary classifications at a specified probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def calibrate_model(
    base_model: Any,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    method: str = "isotonic",
) -> CausalCalibrator:
    """Fits CausalCalibrator on dedicated calibration split."""
    calibrator = CausalCalibrator(base_estimator=base_model, method=method)
    calibrator.fit(X_cal, y_cal)
    return calibrator


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """Serializes a trained model or calibrator to disk using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: Union[str, Path]) -> Any:
    """Loads a serialized model or calibrator from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
