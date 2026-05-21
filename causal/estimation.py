"""
Causal estimation module.

Provides a unified interface over multiple EconML / custom estimators:
  - doubly_robust  → LinearDRLearner
  - r_learner      → LinearDML (the R-Learner DML variant)
  - causal_forest  → CausalForestDML (correct module in econml >= 0.14)
  - t_learner      → TLearner
  - x_learner      → XLearner

Bug fixes applied:
  1. CausalForest import path corrected (econml.dml.CausalForestDML, not econml.grf).
  2. TLearner / XLearner do NOT accept a W kwarg in .fit(); removed it.
  3. T must be passed as a 1-D integer array; ensured .astype(int).
  4. estimate_cate always returns a 1-D numpy array (some models return 2-D).
"""

import numpy as np
import logging

from econml.dml import LinearDML, CausalForestDML
from econml.dr import LinearDRLearner
from econml.metalearners import TLearner, XLearner
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression

class CustomDRLearner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.prop_model = LogisticRegression(max_iter=1000, solver="lbfgs")

    def fit(self, Y, T, X, W=None):
        n = len(Y)
        e = cross_val_predict(self.prop_model, X, T, cv=5, method='predict_proba')[:, 1]
        
        clipped_mask = (e < 0.05) | (e > 0.95)
        num_clipped = clipped_mask.sum()
        if num_clipped > 0:
            print(f"Warning: {num_clipped} units had extreme propensity scores and were clipped.")
            
        e = np.clip(e, 0.05, 0.95)
        
        mu1 = np.zeros(n)
        mu0 = np.zeros(n)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            X_train, Y_train, T_train = X[train_idx], Y[train_idx], T[train_idx]
            X_test = X[test_idx]
            
            mask1 = T_train == 1
            if mask1.sum() > 0:
                mod1 = LinearRegression()
                mod1.fit(X_train[mask1], Y_train[mask1])
                mu1[test_idx] = mod1.predict(X_test)
            else:
                mu1[test_idx] = 0
                
            mask0 = T_train == 0
            if mask0.sum() > 0:
                mod0 = LinearRegression()
                mod0.fit(X_train[mask0], Y_train[mask0])
                mu0[test_idx] = mod0.predict(X_test)
            else:
                mu0[test_idx] = 0
                
        dr_i = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        
        self.dr_ate_ = np.mean(dr_i)
        self.cate_model = LinearRegression()
        self.cate_model.fit(X, dr_i)
        return self

    def effect(self, X):
        return self.cate_model.predict(X)

class CausalEstimator:
    """
    Unified wrapper for causal treatment effect estimators.

    Parameters
    ----------
    method : str
        One of 'doubly_robust', 'r_learner', 'causal_forest',
        't_learner', 'x_learner'.
    """

    def __init__(self, method: str = "doubly_robust"):
        self.method = method
        self.model = None
        self._supports_W = True  # Whether the underlying model accepts W (confounders)

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W=None):
        """
        Fit the causal model.

        Parameters
        ----------
        Y : 1-D array, outcome
        T : 1-D integer array, binary treatment indicator
        X : 2-D array, features for heterogeneity / effect modifiers
        W : 2-D array or None, additional confounders passed to the nuisance models
        """
        # Bug Fix: coerce T to int to avoid XGBoost categorical warnings
        T = np.array(T).astype(int)
        Y = np.array(Y).astype(float)

        if self.method == "doubly_robust":
            self.model = CustomDRLearner(random_state=42)
            self._supports_W = True

        elif self.method == "r_learner":
            # LinearDML implements the R-Learner via cross-fitting
            self.model = LinearDML(
                model_y=XGBRegressor(
                    n_estimators=100, max_depth=3, eval_metric="rmse", verbosity=0
                ),
                model_t=LogisticRegression(max_iter=1000, solver="lbfgs"),
                discrete_treatment=True,
                random_state=42,
            )
            self._supports_W = True

        elif self.method == "causal_forest":
            # Bug Fix: CausalForest in econml >= 0.14 lives in econml.dml, not econml.grf
            self.model = CausalForestDML(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=42,
                discrete_treatment=True,
            )
            self._supports_W = True

        elif self.method == "t_learner":
            self.model = TLearner(
                models=XGBRegressor(
                    n_estimators=100, max_depth=3, eval_metric="rmse", verbosity=0
                )
            )
            # Bug Fix: TLearner.fit() signature is fit(Y, T, X) — no W parameter
            self._supports_W = False

        elif self.method == "x_learner":
            self.model = XLearner(
                models=XGBRegressor(
                    n_estimators=100, max_depth=3, eval_metric="rmse", verbosity=0
                ),
                propensity_model=LogisticRegression(max_iter=1000, solver="lbfgs"),
            )
            # Bug Fix: XLearner.fit() also does not accept W
            self._supports_W = False

        else:
            raise ValueError(f"Unknown estimation method: '{self.method}'")

        logger.info(f"Fitting {self.method} estimator...")
        if self._supports_W:
            self.model.fit(Y, T, X=X, W=W)
        else:
            # Bug Fix: pass only (Y, T, X) for meta-learners
            self.model.fit(Y, T, X=X)

        logger.info(f"{self.method} estimator fitted successfully.")
        return self

    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE for new observations.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before estimating CATE.")
        cate = self.model.effect(X)
        # Bug Fix: some models return 2-D arrays; always flatten to 1-D
        return np.array(cate).flatten()

    def estimate_ate(self, X: np.ndarray) -> float:
        """Compute the population ATE as the mean CATE."""
        return float(np.mean(self.estimate_cate(X)))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    X = np.random.normal(size=(n, 2))
    T = np.random.binomial(1, p=1 / (1 + np.exp(-X[:, 0])))
    Y = 2.0 * T + 1.5 * X[:, 1] + T * X[:, 0] + np.random.normal(size=n)

    for method in ["doubly_robust", "r_learner", "causal_forest", "t_learner", "x_learner"]:
        est = CausalEstimator(method=method)
        est.fit(Y, T, X=X)
        ate = est.estimate_ate(X)
        print(f"[{method}] Estimated ATE: {ate:.4f}")
