import logging
import numpy as np
import pandas as pd

from causal.estimation import CausalEstimator
from causal.discovery import notears_linear, get_dag_from_w
from causal.bandits import LinUCB
from data.ingestion import generate_mimic_synthetic

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
}


class CausalService:
    """
    Service layer that wraps the causal ML engine.
    Trains initial models at startup on synthetic MIMIC data.
    """

    def __init__(self):
        self.estimator = CausalEstimator(method="doubly_robust")
        self.is_fitted = False
        self.dag = None
        self._X_cols = ["age", "blood_pressure", "comorbidities"]
        self._Y_mean = 0.0
        self._cate_std = 1.0
        self._background_X = None
        self._explainer = None
        self.bandit = LinUCB(n_arms=2, d=3, alpha=1.0)
        self._fit_initial_model()

    def _magnitude_band(self, cate: float) -> str:
        abs_c = abs(cate)
        if abs_c < 2.0:
            return "small"
        if abs_c < 5.0:
            return "moderate"
        return "large"

    def _fit_initial_model(self):
        """Fit models on synthetic MIMIC data at startup."""
        logger.info("Fitting initial causal model on synthetic MIMIC data...")
        try:
            df = generate_mimic_synthetic(n_samples=2000)
            X = df[self._X_cols].values
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

            X_dag = df[self._X_cols + ["treatment", "outcome"]].values.astype(float)
            X_dag_norm = (X_dag - X_dag.mean(axis=0)) / (X_dag.std(axis=0) + 1e-8)
            W_est = notears_linear(X_dag_norm, lambda1=0.1, loss_type="l2")
            self.dag = get_dag_from_w(W_est)
            logger.info("Model fitting complete.")
        except Exception as e:
            logger.error(f"Model fitting failed: {e}. Service will return zeros.")
            self.is_fitted = False

    def predict_cate(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(features_df))
        X = features_df[self._X_cols].values
        return self.estimator.estimate_cate(X)

    def predict_cate_with_meta(self, features_df: pd.DataFrame) -> dict:
        cate = np.array(self.predict_cate(features_df)).flatten()
        margin = 1.96 * self._cate_std / (2000 ** 0.5)  # approximate SE from training
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

    def simulate_intervention(
        self, features_df: pd.DataFrame, treatment_value: int
    ) -> list:
        X = features_df[self._X_cols].values
        cate = self.estimator.estimate_cate(X) if self.is_fitted else np.zeros(len(features_df))
        cate = np.array(cate).flatten()
        expected = self._Y_mean + treatment_value * cate
        return expected.tolist()

    def simulate_both_outcomes(self, features_df: pd.DataFrame) -> dict:
        """Return expected outcomes under T=0 and T=1 for each patient."""
        X = features_df[self._X_cols].values
        cate = (
            np.array(self.estimator.estimate_cate(X)).flatten()
            if self.is_fitted
            else np.zeros(len(features_df))
        )
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
        for i, row in enumerate(features_df[self._X_cols].values):
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
                "feature_labels": [FEATURE_LABELS[c] for c in self._X_cols],
                "shap_values": [[0.0] * len(self._X_cols) for _ in range(n)],
                "base_value": 0.0,
            }
        try:
            if self._explainer is None:
                from explainability.shap_explainer import CausalExplainer

                self._explainer = CausalExplainer(
                    self.estimator.model, self._background_X
                )
            X = features_df[self._X_cols].values
            shap_vals = self._explainer.get_shap_values(X, nsamples=nsamples)
            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(1, -1)
            base = float(np.atleast_1d(self._explainer.expected_value)[0])
            return {
                "feature_names": self._X_cols,
                "feature_labels": [FEATURE_LABELS[c] for c in self._X_cols],
                "shap_values": shap_arr.tolist(),
                "base_value": base,
            }
        except Exception as e:
            logger.exception("SHAP explain failed: %s", e)
            n = len(features_df)
            return {
                "feature_names": self._X_cols,
                "feature_labels": [FEATURE_LABELS[c] for c in self._X_cols],
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
                u_name = node_names[int(u)]
                v_name = node_names[int(v)]
                edges.append({"source": u_name, "target": v_name})
            except (IndexError, ValueError):
                pass
        return node_names, edges

    def get_reference_dag(self) -> dict:
        node_names = self._X_cols + ["treatment", "outcome"]
        return {"nodes": node_names, "edges": REFERENCE_DAG_EDGES}


causal_service = CausalService()
