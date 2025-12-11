# model_training.py

import pickle
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    DATA_DIR,
    DEFAULT_DATA_VERSION,
    BURNOUT_MODEL_PATH,
    FEATURES_METADATA_PATH,
    GLOBAL_SEED,
)
from data_generation import EmployeeDataConfig, generate_and_save
from features import FeatureEngineer, FeatureConfig


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = GLOBAL_SEED
    target_col: str = "HighBurnoutRisk"


class BurnoutPredictor:
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config
        self.model: XGBClassifier | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.metrics_: Dict[str, Any] = {}

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.training_config
        fe_cfg = FeatureConfig(target_col=cfg.target_col, drop_cols=["Persona"])
        self.feature_engineer = FeatureEngineer(fe_cfg)

        X = self.feature_engineer.fit_transform(df)
        y = df[cfg.target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.metrics_ = {
            "auc": float(auc),
            "classification_report": report,
        }
        return self.metrics_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_engineer is None:
            raise RuntimeError("Model not trained or loaded.")
        X = self.feature_engineer.transform(df)
        return self.model.predict_proba(X)[:, 1]

    def save(self) -> None:
        if self.model is None or self.feature_engineer is None:
            raise RuntimeError("Nothing to save.")

        with open(BURNOUT_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

        metadata = {
            "feature_engineer": self.feature_engineer,
            "metrics": self.metrics_,
        }
        with open(FEATURES_METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load() -> "BurnoutPredictor":
        with open(BURNOUT_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        predictor = BurnoutPredictor(TrainingConfig())
        predictor.model = model
        predictor.feature_engineer = metadata["feature_engineer"]
        predictor.metrics_ = metadata.get("metrics", {})
        return predictor


def train_pipeline(
    data_version: str = DEFAULT_DATA_VERSION,
    n_employees: int = 5000,
) -> Dict[str, Any]:
    path = DATA_DIR / f"employees_{data_version}.parquet"
    if not path.exists():
        generate_and_save(EmployeeDataConfig(n_employees=n_employees, version=data_version), path)

    df = pd.read_parquet(path)
    trainer = BurnoutPredictor(TrainingConfig())
    metrics = trainer.train(df)
    trainer.save()
    return metrics
