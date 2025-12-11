# data_generation.py

import numpy as np
import pandas as pd
from typing import Literal
from dataclasses import dataclass
from config import DATA_DIR, GLOBAL_SEED

RNG = np.random.default_rng(GLOBAL_SEED)


@dataclass
class EmployeeDataConfig:
    n_employees: int = 5000
    version: Literal["v1", "v2"] = "v2"


class UnifiedEmployeeDataGenerator:
    """
    Synthetic employee wellbeing / burnout data generator.

    v2 is persona-based and now explicitly balances the distribution of
    risk bands that the app later maps to:
      - Low   (green)   → BurnoutProb < 0.40
      - Medium(orange)  → 0.40–0.70
      - High  (red)     → > 0.70

    We do NOT hard-code the model output. Instead we:
      1. Build an internal rule-based "risk intensity" (rule_prob).
      2. Use it ONLY to:
         - define HighBurnoutRisk label for training;
         - shape WorkLifeBalanceScore and JobSatisfactionScore;
         - balance the dataset so Low / Medium / High buckets are ~equal.
    """

    def __init__(self, config: EmployeeDataConfig):
        self.config = config

    # ---------------- Public API ----------------

    def generate(self) -> pd.DataFrame:
        if self.config.version == "v1":
            return self._generate_v1()
        elif self.config.version == "v2":
            return self._generate_v2()
        else:
            raise ValueError(f"Unknown data version: {self.config.version}")

    # ---------------- v1: simple baseline (kept mostly as-is) ----------------

    def _generate_v1(self) -> pd.DataFrame:
        n = self.config.n_employees
        df = pd.DataFrame({
            "EmployeeID": np.arange(1, n + 1),
            "Age": RNG.integers(22, 60, size=n),
            "TenureYears": RNG.uniform(0.2, 15, size=n).round(1),
            "WorkHoursPerWeek": RNG.normal(42, 6, size=n).clip(30, 65).round(1),
            "RemoteDaysPerWeek": RNG.integers(0, 5, size=n),
            "JobRole": RNG.choice(
                ["Engineer", "Manager", "Analyst", "Support", "Sales"], size=n
            ),
            "TeamSize": RNG.integers(3, 25, size=n),
            "ManagerSupportScore": RNG.integers(1, 6, size=n),
            "RecognitionFrequency": RNG.integers(0, 6, size=n),
            "RecentPromotion": RNG.integers(0, 2, size=n),
            "TrainingHoursLast6M": RNG.integers(0, 60, size=n),
            "StressLevelSelfReport": RNG.integers(1, 11, size=n),
            "SleepHours": RNG.normal(6.5, 1.0, size=n).clip(3, 9).round(1),
            "FinancialStressScore": RNG.integers(1, 6, size=n),
        })

        # simple burnout score just to define labels (not used directly in the app)
        burnout_score = (
            0.15 * (df["WorkHoursPerWeek"] - 40)
            + 0.30 * (df["StressLevelSelfReport"] - 5)
            - 0.20 * (df["ManagerSupportScore"] - 3)
            - 0.15 * (df["RecognitionFrequency"] - 2)
            + 0.10 * (df["FinancialStressScore"] - 3)
            - 0.10 * (df["RemoteDaysPerWeek"] - 2)
        )
        burnout_score = burnout_score + RNG.normal(0, 1.0, size=len(df))
        df["BurnoutScore"] = burnout_score

        # binary label for training
        df["HighBurnoutRisk"] = (
            df["BurnoutScore"] > df["BurnoutScore"].quantile(0.7)
        ).astype(int)

        # Work-Life Balance & Job Satisfaction (1–4)
        wlb_raw = (
            4.2
            - 0.06 * (df["WorkHoursPerWeek"] - 40)
            - 0.10 * (df["StressLevelSelfReport"] - 5)
            + 0.08 * (df["RemoteDaysPerWeek"] - 2)
        ) + RNG.normal(0, 0.3, size=len(df))

        js_raw = (
            2.5
            + 0.18 * (df["ManagerSupportScore"] - 3)
            + 0.12 * (df["RecognitionFrequency"] - 2)
            - 0.08 * (df["StressLevelSelfReport"] - 5)
        ) + RNG.normal(0, 0.3, size=len(df))

        df["WorkLifeBalanceScore"] = np.clip(wlb_raw, 1, 4).round(0)
        df["JobSatisfactionScore"] = np.clip(js_raw, 1, 4).round(0)

        return df

    # ---------------- v2: persona-based, now risk-balanced ----------------

    def _generate_v2(self) -> pd.DataFrame:
        # We generate a larger pool, then sample to get balanced Low/Med/High
        n_final = self.config.n_employees
        pool_n = int(n_final * 3)  # oversample so each band has enough cases

        personas = RNG.choice(
            ["OverloadedHighPerformer", "QuietQuitter", "EngagedBalanced", "EarlyCareer"],
            size=pool_n,
            p=[0.25, 0.20, 0.40, 0.15],
        )

        df = pd.DataFrame(
            {
                "EmployeeID": np.arange(1, pool_n + 1),
                "Persona": personas,
            }
        )

        def sample_from_persona(persona: str) -> dict:
            if persona == "OverloadedHighPerformer":
                return {
                    "Age": RNG.integers(28, 45),
                    "TenureYears": RNG.uniform(2, 10),
                    "WorkHoursPerWeek": RNG.normal(52, 4),
                    "RemoteDaysPerWeek": RNG.integers(0, 3),
                    "JobRole": RNG.choice(["Engineer", "Manager", "Sales"]),
                    "TeamSize": RNG.integers(5, 15),
                    "ManagerSupportScore": RNG.integers(2, 5),
                    "RecognitionFrequency": RNG.integers(1, 4),
                    "RecentPromotion": RNG.choice([0, 1], p=[0.5, 0.5]),
                    "TrainingHoursLast6M": RNG.integers(10, 50),
                    "StressLevelSelfReport": RNG.integers(7, 11),
                    "SleepHours": RNG.normal(5.8, 0.7),
                    "FinancialStressScore": RNG.integers(2, 6),
                }
            elif persona == "QuietQuitter":
                return {
                    "Age": RNG.integers(26, 50),
                    "TenureYears": RNG.uniform(3, 12),
                    "WorkHoursPerWeek": RNG.normal(40, 3),
                    "RemoteDaysPerWeek": RNG.integers(2, 5),
                    "JobRole": RNG.choice(["Analyst", "Support"]),
                    "TeamSize": RNG.integers(5, 25),
                    "ManagerSupportScore": RNG.integers(1, 3),
                    "RecognitionFrequency": RNG.integers(0, 2),
                    "RecentPromotion": 0,
                    "TrainingHoursLast6M": RNG.integers(0, 20),
                    "StressLevelSelfReport": RNG.integers(5, 9),
                    "SleepHours": RNG.normal(6.5, 0.8),
                    "FinancialStressScore": RNG.integers(2, 5),
                }
            elif persona == "EngagedBalanced":
                return {
                    "Age": RNG.integers(25, 55),
                    "TenureYears": RNG.uniform(1, 15),
                    "WorkHoursPerWeek": RNG.normal(40, 3),
                    "RemoteDaysPerWeek": RNG.integers(1, 4),
                    "JobRole": RNG.choice(["Engineer", "Analyst", "Support"]),
                    "TeamSize": RNG.integers(3, 20),
                    "ManagerSupportScore": RNG.integers(3, 6),
                    "RecognitionFrequency": RNG.integers(2, 6),
                    "RecentPromotion": RNG.choice([0, 1], p=[0.7, 0.3]),
                    "TrainingHoursLast6M": RNG.integers(20, 60),
                    "StressLevelSelfReport": RNG.integers(2, 7),
                    "SleepHours": RNG.normal(7.2, 0.6),
                    "FinancialStressScore": RNG.integers(1, 4),
                }
            else:  # EarlyCareer
                return {
                    "Age": RNG.integers(22, 30),
                    "TenureYears": RNG.uniform(0.2, 3),
                    "WorkHoursPerWeek": RNG.normal(44, 5),
                    "RemoteDaysPerWeek": RNG.integers(1, 4),
                    "JobRole": RNG.choice(["Support", "Analyst", "Engineer"]),
                    "TeamSize": RNG.integers(3, 12),
                    "ManagerSupportScore": RNG.integers(2, 5),
                    "RecognitionFrequency": RNG.integers(1, 4),
                    "RecentPromotion": 0,
                    "TrainingHoursLast6M": RNG.integers(10, 50),
                    "StressLevelSelfReport": RNG.integers(4, 9),
                    "SleepHours": RNG.normal(6.3, 0.9),
                    "FinancialStressScore": RNG.integers(2, 6),
                }

        rows = [sample_from_persona(p) for p in personas]
        features_df = pd.DataFrame(rows)
        df = pd.concat([df, features_df], axis=1)

        # clean ranges
        df["TenureYears"] = df["TenureYears"].clip(0.1, 20).round(1)
        df["WorkHoursPerWeek"] = df["WorkHoursPerWeek"].clip(30, 65).round(1)
        df["SleepHours"] = df["SleepHours"].clip(3, 9).round(1)

        # ----- RULE-BASED risk (only for labels + WLB/JS shaping) -----
        rule_raw = (
            0.20 * (df["WorkHoursPerWeek"] - 40)
            + 0.35 * (df["StressLevelSelfReport"] - 5)
            - 0.25 * (df["ManagerSupportScore"] - 3)
            - 0.20 * (df["RecognitionFrequency"] - 2)
            + 0.15 * (df["FinancialStressScore"] - 3)
            - 0.10 * (df["RemoteDaysPerWeek"] - 2)
        )
        rule_raw = rule_raw + RNG.normal(0, 1.2, size=len(df))

        rule_prob = 1 / (1 + np.exp(-0.4 * rule_raw))  # 0–1 intensity
        df["BurnoutScore"] = rule_raw
        df["BurnoutProbability"] = rule_prob

        # internal risk band for balancing
        df["RuleRiskBand"] = pd.cut(
            rule_prob,
            bins=[0.0, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        # binary label the model will learn
        df["HighBurnoutRisk"] = (rule_prob > 0.6).astype(int)

        # ---- Work-Life Balance & Job Satisfaction (1–4), linked to rule_prob ----
        risk_intensity = rule_prob

        wlb_raw = (
            4.3
            - 1.8 * risk_intensity
            - 0.06 * (df["WorkHoursPerWeek"] - 40)
            - 0.08 * (df["StressLevelSelfReport"] - 5)
            + 0.06 * (df["RemoteDaysPerWeek"] - 2)
            + 0.06 * (df["ManagerSupportScore"] - 3)
        ) + RNG.normal(0, 0.25, size=len(df))

        js_raw = (
            3.0
            - 1.6 * risk_intensity
            + 0.20 * (df["ManagerSupportScore"] - 3)
            + 0.14 * (df["RecognitionFrequency"] - 2)
            - 0.10 * (df["StressLevelSelfReport"] - 5)
            - 0.05 * (df["FinancialStressScore"] - 3)
        ) + RNG.normal(0, 0.25, size=len(df))

        df["WorkLifeBalanceScore"] = np.clip(wlb_raw, 1, 4).round(0)
        df["JobSatisfactionScore"] = np.clip(js_raw, 1, 4).round(0)

        # ----- BALANCE Low / Medium / High based on RuleRiskBand -----
        target_per_band = n_final // 3
        bands = ["Low", "Medium", "High"]
        balanced_parts = []

        for i, band in enumerate(bands):
            band_df = df[df["RuleRiskBand"] == band]
            if band_df.empty:
                # extremely unlikely, but guard anyway: fallback to sampling from all
                band_df = df

            if i == len(bands) - 1:
                # last band takes the remainder
                needed = n_final - target_per_band * (len(bands) - 1)
            else:
                needed = target_per_band

            if len(band_df) >= needed:
                sampled = band_df.sample(n=needed, replace=False, random_state=GLOBAL_SEED + i)
            else:
                sampled = band_df.sample(n=needed, replace=True, random_state=GLOBAL_SEED + i)

            balanced_parts.append(sampled)

        df_balanced = pd.concat(balanced_parts, axis=0).reset_index(drop=True)

        # Reassign EmployeeID to be 1..n_final for a clean demo
        df_balanced["EmployeeID"] = np.arange(1, len(df_balanced) + 1)

        # Drop helper column
        df_balanced = df_balanced.drop(columns=["RuleRiskBand"])

        # ========== FIX PERSONA LABELS BASED ON ACTUAL RISK ==========
        # Re-label personas to match predicted risk levels for consistency
        # This prevents showing "EngagedBalanced" for high-risk employees
        
        # Calculate actual risk probabilities to determine true persona
        from model_training import BurnoutPredictor
        try:
            predictor = BurnoutPredictor.load()
            actual_probs = predictor.predict_proba(df_balanced)
            
            # Re-assign personas based on actual model predictions
            df_balanced.loc[actual_probs >= 0.70, "Persona"] = "OverloadedHighPerformer"
            df_balanced.loc[(actual_probs >= 0.40) & (actual_probs < 0.70), "Persona"] = "QuietQuitter"
            df_balanced.loc[actual_probs < 0.40, "Persona"] = "EngagedBalanced"
            
        except Exception as e:
            # If model not available yet, use rule-based probability
            print(f"Model not available for persona re-labeling: {e}")
            print("Using rule-based probability for persona assignment")
            rule_probs = df_balanced["BurnoutProbability"]
            df_balanced.loc[rule_probs >= 0.70, "Persona"] = "OverloadedHighPerformer"
            df_balanced.loc[(rule_probs >= 0.40) & (rule_probs < 0.70), "Persona"] = "QuietQuitter"  
            df_balanced.loc[rule_probs < 0.40, "Persona"] = "EngagedBalanced"


        return df_balanced


def generate_and_save(config: EmployeeDataConfig, output_path=None) -> str:
    if output_path is None:
        output_path = DATA_DIR / f"employees_{config.version}.parquet"
    gen = UnifiedEmployeeDataGenerator(config)
    df = gen.generate()
    df.to_parquet(output_path, index=False)
    return str(output_path)
