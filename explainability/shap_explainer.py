"""
SHAP-based explainability for causal treatment effect models.

Bug fixes applied:
  1. shap.kmeans background summary now passes data as a numpy array,
     not a Pandas DataFrame, to avoid type errors with KernelExplainer.
  2. `waterfall_plot` now closes the figure after returning it to prevent
     matplotlib state leakage across calls.
  3. Added `nsamples` parameter to `get_shap_values` to control runtime
     (KernelExplainer can be very slow with default settings).
  4. `summary_plot` now calls `plt.close()` after capturing the figure
     to avoid memory leaks.
"""

import shap
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CausalExplainer:
    """
    SHAP-based explainer for EconML causal CATE models.

    Uses KernelExplainer on the `model.effect` function for model-agnostic
    attribution of treatment effect predictions.
    """

    def __init__(self, causal_model, background_data: np.ndarray):
        """
        Parameters
        ----------
        causal_model : fitted CausalEstimator (or any object with .effect(X))
        background_data : np.ndarray of shape (n, d) — representative samples
        """
        self.model = causal_model

        # Bug Fix: ensure background_data is a plain numpy array
        background_data = np.array(background_data)

        if len(background_data) > 100:
            # Summarise to k=10 cluster centres for speed
            self.background = shap.kmeans(background_data, 10)
        else:
            self.background = background_data

        logger.info("Initialising SHAP KernelExplainer on model.effect ...")
        self.explainer = shap.KernelExplainer(self.model.effect, self.background)
        self.expected_value = self.explainer.expected_value
        logger.info(f"SHAP base value (expected CATE): {self.expected_value:.4f}")

    def get_shap_values(self, X: np.ndarray, nsamples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for each row in X.

        Parameters
        ----------
        X       : np.ndarray, shape (n, d)
        nsamples: number of SHAP background samples (controls accuracy vs speed)

        Returns
        -------
        np.ndarray of shape (n, d)
        """
        # Bug Fix: pass nsamples to limit computation time
        return self.explainer.shap_values(X, nsamples=nsamples, silent=True)

    def summary_plot(
        self, X: np.ndarray, shap_values: np.ndarray, feature_names=None, show: bool = False
    ):
        """
        Generate a SHAP beeswarm summary plot.

        Returns the matplotlib Figure object when show=False.
        """
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        fig = plt.gcf()
        if show:
            plt.show()
        # Bug Fix: close figure to avoid accumulation in memory
        plt.close(fig)
        return fig

    def waterfall_plot(
        self,
        X_instance: np.ndarray,
        shap_values_instance: np.ndarray,
        expected_value: float,
        feature_names=None,
        show: bool = False,
    ):
        """
        Generate a SHAP waterfall plot for a single prediction.

        Returns the matplotlib Figure object when show=False.
        """
        import matplotlib.pyplot as plt

        explanation = shap.Explanation(
            values=shap_values_instance,
            base_values=expected_value,
            data=X_instance,
            feature_names=feature_names,
        )
        shap.waterfall_plot(explanation, show=False)
        fig = plt.gcf()
        if show:
            plt.show()
        # Bug Fix: close figure to prevent matplotlib state leakage
        plt.close(fig)
        return fig
