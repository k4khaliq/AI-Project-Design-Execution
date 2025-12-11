# features.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class FeatureConfig:
    target_col: str = "HighBurnoutRisk"
    drop_cols: Optional[List[str]] = None


class FeatureEngineer:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[List[str]] = None

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "WorkHoursPerWeek" in df.columns and "RemoteDaysPerWeek" in df.columns:
            df["WorkloadIntensity"] = df["WorkHoursPerWeek"] - 2 * df["RemoteDaysPerWeek"]

        if "TenureYears" in df.columns and "RecentPromotion" in df.columns:
            df["CareerStagnation"] = np.where(
                (df["TenureYears"] > 5) & (df["RecentPromotion"] == 0), 1, 0
            )

        if "StressLevelSelfReport" in df.columns and "SleepHours" in df.columns:
            df["StressSleepGap"] = df["StressLevelSelfReport"] - (10 - df["SleepHours"])

        if "ManagerSupportScore" in df.columns and "RecognitionFrequency" in df.columns:
            df["SupportRecognitionIndex"] = (
                df["ManagerSupportScore"] + df["RecognitionFrequency"]
            )

        return df

    def fit(self, df: pd.DataFrame) -> None:
        df = self._create_derived_features(df)

        target_col = self.config.target_col
        drop_cols = set(self.config.drop_cols or []) | {"EmployeeID", target_col, "BurnoutScore", "BurnoutProbability"}

        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        self.pipeline.fit(feature_df)

        # store final feature names
        num_features = numeric_cols
        cat_features = list(
            self.pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
        )
        self.feature_names_ = num_features + cat_features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("FeatureEngineer not fitted yet.")

        df = self._create_derived_features(df)

        target_col = self.config.target_col
        drop_cols = set(self.config.drop_cols or []) | {"EmployeeID", target_col, "BurnoutScore", "BurnoutProbability"}
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        X = self.pipeline.transform(feature_df)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "feature_names": self.feature_names_,
        }
