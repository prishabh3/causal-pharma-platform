import logging
import numpy as np

logger = logging.getLogger(__name__)

class GRFEstimator:
    """
    Python wrapper for the R 'grf' package (Generalized Random Forests).
    Specifically wraps the causal_forest() function.
    
    Requires R and rpy2 installed on the system, plus the 'grf' R package.
    """
    def __init__(self):
        self.model = None
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            import rpy2.robjects.numpy2ri
            
            # Activate numpy <-> R conversion
            rpy2.robjects.numpy2ri.activate()
            
            self.robjects = robjects
            self.grf = importr('grf')
            self.base = importr('base')
        except ImportError:
            raise ImportError(
                "rpy2 is not installed or R is not configured. "
                "To use GRFEstimator, you must install R and then run: "
                "`pip install rpy2`. Additionally, install the 'grf' package "
                "in R via `install.packages('grf')`."
            )
        except Exception as e:
            raise ImportError(
                f"Failed to load R environment or 'grf' package. "
                f"Ensure R is installed and `install.packages('grf')` has been run. "
                f"Inner error: {e}"
            )

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
        """
        Fit a causal forest model using grf::causal_forest.
        """
        logger.info("Fitting GRF causal_forest in R...")
        
        # Ensure correct types
        X_r = self.robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        Y_r = self.robjects.FloatVector(Y)
        T_r = self.robjects.FloatVector(T)
        
        # Train causal_forest
        # Default tuning parameters are robust enough for demonstration
        self.model = self.grf.causal_forest(X_r, Y_r, T_r)
        
        logger.info("GRF causal_forest fitted successfully.")
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE for new observations.
        """
        if self.model is None:
            raise ValueError("GRFEstimator must be fitted before calling predict_cate.")
            
        X_r = self.robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        
        # Predict
        preds_r = self.robjects.r.predict(self.model, X_r)
        
        # Convert predictions back to numpy
        # R returns a list/dataframe-like object where 'predictions' contains the values
        preds_array = np.array(preds_r.rx2('predictions'))
        
        return preds_array.flatten()

    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """Alias for compatibility with CausalEstimator interface."""
        return self.predict_cate(X)
        
    def estimate_ate(self, X: np.ndarray) -> float:
        """Compute the population ATE as the mean CATE."""
        return float(np.mean(self.estimate_cate(X)))
