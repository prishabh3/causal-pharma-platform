import logging
import numpy as np
import pandas as pd
import os
import sys

DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic")

from causal.estimation import CausalEstimator
from causal.discovery import notears_linear, get_dag_from_w
from causal.bandits import LinUCB
from data.ingestion import generate_mimic_synthetic

if DATA_SOURCE == "mimic":
    from data.mimic_pipeline import process_mimic_data

logger = logging.getLogger(__name__)

# Domain-knowledge reference structure (for comparison with learned DAG)
REFERENCE_DAG_EDGES = [
    {"source": "age", "target": "treatment"},
    {"source": "blood_pressure", "target": "treatment"},
    {"source": "comorbidities", "target": "treatment"},
    {"source": "age", "target": "outcome"},
    {"source": "blood_pressure", "target": "outcome"},
    {"source": "comorbidities", "target": "outcome"},
    {"source": "treatment", "target": "outcome"},
]

FEATURE_LABELS = {
    "age": "Age",
    "blood_pressure": "Blood Pressure",
    "comorbidities": "Comorbidities",
    "GENDER_M": "Male",
    "ADMISSION_TYPE_EMERGENCY": "Emergency Admit",
    "ADMISSION_TYPE_URGENT": "Urgent Admit",
    "icd_1": "ICD 1", "icd_2": "ICD 2", "icd_3": "ICD 3", 
    "icd_4": "ICD 4", "icd_5": "ICD 5", "icd_6": "ICD 6",
    "creatinine": "Creatinine", "lactate": "Lactate", "wbc": "WBC", 
    "glucose": "Glucose", "bilirubin": "Bilirubin"
}

class CausalService:
    """
    Service layer that wraps the causal ML engine.
    Trains initial models at startup on synthetic or MIMIC data.
    """

    def __init__(self):
        self.estimator = CausalEstimator(method="doubly_robust")
        self.is_fitted = False
        self.dag = None
        self._ui_cols = ["age", "blood_pressure", "comorbidities"]
        
        if DATA_SOURCE == "mimic":
            self._X_cols = [
                'age', 'GENDER_M', 'ADMISSION_TYPE_EMERGENCY', 'ADMISSION_TYPE_URGENT',
                'icd_1', 'icd_2', 'icd_3', 'icd_4', 'icd_5', 'icd_6',
                'creatinine', 'lactate', 'wbc', 'glucose', 'bilirubin'
            ]
        else:
            self._X_cols = ["age", "blood_pressure", "comorbidities"]
            
        self._X_medians = None
        self._Y_mean = 0.0
        self._cate_std = 1.0
        self._background_X = None
        self._explainer = None
        self.grf_estimator = None
        self.bandit = LinUCB(n_arms=2, d=len(self._X_cols), alpha=1.0)
        self._fit_initial_model()

    def _magnitude_band(self, cate: float) -> str:
        abs_c = abs(cate)
        if abs_c < 2.0:
            return "small"
        if abs_c < 5.0:
            return "moderate"
        return "large"

    def _fit_initial_model(self):
        """Fit models on data at startup."""
        logger.info(f"Fitting initial causal model on {DATA_SOURCE} data...")
        try:
            if DATA_SOURCE == "mimic":
                raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mimic_raw")
                out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cohort.parquet")
                df = process_mimic_data(raw_dir, out_path)
            else:
                df = generate_mimic_synthetic(n_samples=2000)

            X = df[self._X_cols].values
            self._X_medians = df[self._X_cols].median().values
            T = df["treatment"].values.astype(int)
            Y = df["outcome"].values

            self.estimator.fit(Y, T, X=X)
            self.is_fitted = True
            self._Y_mean = float(np.mean(Y))

            train_cate = self.estimator.estimate_cate(X)
            self._cate_std = float(np.std(train_cate)) or 1.0

            # SHAP background sample
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=min(150, len(X)), replace=False)
            self._background_X = X[idx]

            # Warm-start contextual bandit on synthetic trajectories
            for row, t, y in zip(X[:500], T[:500], Y[:500]):
                ctx = row.reshape(-1)
                arm = self.bandit.select_arm(ctx)
                reward = y / 100.0 if arm == int(t) else (y / 100.0) * 0.5
                self.bandit.update(arm, ctx, reward)

            # Only run NOTEARS on a subset if many features, or just full for synthetic
            X_dag = df[self._X_cols + ["treatment", "outcome"]].values.astype(float)
            X_dag_norm = (X_dag - X_dag.mean(axis=0)) / (X_dag.std(axis=0) + 1e-8)
            W_est = notears_linear(X_dag_norm, lambda1=0.1, loss_type="l2")
            self.dag = get_dag_from_w(W_est)
            logger.info("Model fitting complete.")
            
            try:
                from causal.grf_estimator import GRFEstimator
                self.grf_estimator = GRFEstimator()
                self.grf_estimator.fit(Y, T, X=X)
                logger.info("GRF model fitting complete.")
            except Exception as e:
                logger.warning(f"Skipping GRF model fitting: {type(e).__name__}: {e}")
                self.grf_estimator = None
                
        except Exception as e:
            logger.error(f"Model fitting failed: {e}. Service will return zeros.")
            self.is_fitted = False

    def _prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros((len(features_df), len(self._X_cols)))
        # Start with medians
        X = np.tile(self._X_medians, (len(features_df), 1))
        # Override with any features provided by UI
        for i, col in enumerate(self._X_cols):
            if col in features_df.columns:
                X[:, i] = features_df[col].values
        return X

    def predict_cate(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(features_df))
        X = self._prepare_features(features_df)
        return self.estimator.estimate_cate(X)

    def predict_cate_with_meta(self, features_df: pd.DataFrame) -> dict:
        cate = np.array(self.predict_cate(features_df)).flatten()
        margin = 1.96 * self._cate_std / (2000 ** 0.5)
        margin = max(margin, 0.25)
        return {
            "cate_estimates": cate.tolist(),
            "ate_estimate": float(np.mean(cate)),
            "ci_lower": (cate - margin).tolist(),
            "ci_upper": (cate + margin).tolist(),
            "magnitude": [self._magnitude_band(c) for c in cate],
            "confidence_pct": [
                min(95, max(55, int(100 - 100 * margin / (abs(c) + 1e-6))))
                for c in cate
            ],
        }

    def predict_cate_grf_with_meta(self, features_df: pd.DataFrame) -> dict:
        if self.grf_estimator is None:
            raise Exception("GRFEstimator is not available. Please install R and rpy2.")
        if not self.is_fitted:
            cate = np.zeros(len(features_df)).flatten()
        else:
            X = self._prepare_features(features_df)
            cate = self.grf_estimator.estimate_cate(X).flatten()
            
        margin = max(1.96 * self._cate_std / (2000 ** 0.5), 0.25)
        return {
            "cate_estimates": cate.tolist(),
            "ate_estimate": float(np.mean(cate)),
            "ci_lower": (cate - margin).tolist(),
            "ci_upper": (cate + margin).tolist(),
            "magnitude": [self._magnitude_band(c) for c in cate],
            "confidence_pct": [
                min(95, max(55, int(100 - 100 * margin / (abs(c) + 1e-6))))
                for c in cate
            ],
        }

    def simulate_intervention(
        self, features_df: pd.DataFrame, treatment_value: int
    ) -> list:
        if not self.is_fitted:
            return np.zeros(len(features_df)).tolist()
        X = self._prepare_features(features_df)
        cate = self.estimator.estimate_cate(X)
        cate = np.array(cate).flatten()
        expected = self._Y_mean + treatment_value * cate
        return expected.tolist()

    def simulate_both_outcomes(self, features_df: pd.DataFrame) -> dict:
        if not self.is_fitted:
            z = np.zeros(len(features_df)).tolist()
            return {"outcome_if_withhold": z, "outcome_if_treat": z, "treatment_effect": z}
        X = self._prepare_features(features_df)
        cate = np.array(self.estimator.estimate_cate(X)).flatten()
        y0 = (self._Y_mean + 0 * cate).tolist()
        y1 = (self._Y_mean + 1 * cate).tolist()
        return {
            "outcome_if_withhold": y0,
            "outcome_if_treat": y1,
            "treatment_effect": cate.tolist(),
        }

    def policy_decision(self, features_df: pd.DataFrame) -> dict:
        cate = np.array(self.predict_cate(features_df)).flatten()
        recommended = [1 if c > 0 else 0 for c in cate]
        rationales = []
        bandit_recs = []
        
        X_eval = self._prepare_features(features_df)
        for i, row in enumerate(X_eval):
            ctx = row.reshape(-1)
            bandit_arm = int(self.bandit.select_arm(ctx))
            bandit_recs.append(bandit_arm)
            c = cate[i]
            if recommended[i] == 1:
                rationales.append(
                    f"Estimated benefit is positive ({c:+.2f} outcome units). "
                    "Treatment is expected to help this patient."
                )
            else:
                rationales.append(
                    f"Estimated benefit is not positive ({c:+.2f} outcome units). "
                    "Withholding treatment is expected to be better or equivalent."
                )
        return {
            "recommended_treatment": recommended,
            "rationale": rationales,
            "bandit_recommendation": bandit_recs,
        }

    def explain_prediction(self, features_df: pd.DataFrame, nsamples: int = 80) -> dict:
        if not self.is_fitted or self._background_X is None:
            n = len(features_df)
            return {
                "feature_names": self._X_cols,
                "feature_labels": [FEATURE_LABELS.get(c, c) for c in self._X_cols],
                "shap_values": [[0.0] * len(self._X_cols) for _ in range(n)],
                "base_value": 0.0,
            }
        try:
            if self._explainer is None:
                from explainability.shap_explainer import CausalExplainer
                self._explainer = CausalExplainer(
                    self.estimator.model, self._background_X
                )
            X = self._prepare_features(features_df)
            shap_vals = self._explainer.get_shap_values(X, nsamples=nsamples)
            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(1, -1)
            base = float(np.atleast_1d(self._explainer.expected_value)[0])
            return {
                "feature_names": self._X_cols,
                "feature_labels": [FEATURE_LABELS.get(c, c) for c in self._X_cols],
                "shap_values": shap_arr.tolist(),
                "base_value": base,
            }
        except Exception as e:
            logger.exception("SHAP explain failed: %s", e)
            n = len(features_df)
            return {
                "feature_names": self._X_cols,
                "feature_labels": [FEATURE_LABELS.get(c, c) for c in self._X_cols],
                "shap_values": [[0.0] * len(self._X_cols) for _ in range(n)],
                "base_value": 0.0,
            }

    def get_dag_edges(self) -> tuple:
        node_names = self._X_cols + ["treatment", "outcome"]
        edges = []
        if self.dag is None:
            return node_names, edges
        for u, v in self.dag.edges():
            try:
                u_idx, v_idx = int(u), int(v)
                if u_idx >= len(node_names) or v_idx >= len(node_names):
                    logger.warning("DAG edge (%d, %d) exceeds node_names length %d; skipping.", u_idx, v_idx, len(node_names))
                    continue
                edges.append({"source": node_names[u_idx], "target": node_names[v_idx]})
            except (ValueError, TypeError):
                pass
        return node_names, edges

    def get_reference_dag(self) -> dict:
        if DATA_SOURCE == "mimic":
            # Just construct a plausible reference DAG for MIMIC
            edges = [
                {"source": "age", "target": "treatment"},
                {"source": "GENDER_M", "target": "treatment"},
                {"source": "creatinine", "target": "treatment"},
                {"source": "lactate", "target": "outcome"},
                {"source": "age", "target": "outcome"},
                {"source": "treatment", "target": "outcome"},
            ]
            node_names = self._X_cols + ["treatment", "outcome"]
            return {"nodes": node_names, "edges": edges}
        
        node_names = self._X_cols + ["treatment", "outcome"]
        return {"nodes": node_names, "edges": REFERENCE_DAG_EDGES}


causal_service = CausalService()
