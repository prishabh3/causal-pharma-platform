import pytest
import pandas as pd
import numpy as np
import urllib.error
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from causal.estimation import CausalEstimator

def test_estimators_lalonde():
    URL = "http://www.nber.org/~rdehejia/data/nsw_dw.dta"
    try:
        df = pd.read_stata(URL)
    except Exception:
        if os.path.exists("data/lalonde.dta"):
            df = pd.read_stata("data/lalonde.dta")
        else:
            raise RuntimeError("Could not download LaLonde dataset and local copy at data/lalonde.dta not found.")
    
    Y = df['re78'].values
    T = df['treat'].values
    X = df.drop(columns=['data_id', 'treat', 're78']).values
    
    rct_ate = 1794.0
    tol = rct_ate * 0.40 # 40% tolerance
    
    results = []

    # 1. Naive OLS
    X_ols = np.column_stack([X, T])
    ols = LinearRegression().fit(X_ols, Y)
    ate_ols = ols.coef_[-1]
    bias_ols = abs(ate_ols - rct_ate) / rct_ate
    
    n = len(X)
    Y_pred = ols.predict(X_ols)
    residuals = Y - Y_pred
    mse = np.sum(residuals**2) / (n - X_ols.shape[1])
    var_b = mse * np.linalg.inv(X_ols.T @ X_ols)[-1, -1]
    se_ols = np.sqrt(var_b)
    ci_ols = (ate_ols - 1.96*se_ols, ate_ols + 1.96*se_ols)
    
    results.append(("Naive OLS", ate_ols, ci_ols, bias_ols))
    
    # 2. IPW
    prop_model = LogisticRegression(max_iter=1000).fit(X, T)
    p = prop_model.predict_proba(X)[:, 1]
    p = np.clip(p, 0.05, 0.95)
    ipw_y1 = T * Y / p
    ipw_y0 = (1 - T) * Y / (1 - p)
    ate_ipw_samples = ipw_y1 - ipw_y0
    ate_ipw = np.mean(ate_ipw_samples)
    bias_ipw = abs(ate_ipw - rct_ate) / rct_ate
    se_ipw = np.std(ate_ipw_samples) / np.sqrt(n)
    ci_ipw = (ate_ipw - 1.96*se_ipw, ate_ipw + 1.96*se_ipw)
    results.append(("IPW", ate_ipw, ci_ipw, bias_ipw))
    
    # 3. Doubly Robust
    dr = CausalEstimator(method="doubly_robust")
    dr.fit(Y, T, X)
    ate_dr = dr.estimate_ate(X)
    bias_dr = abs(ate_dr - rct_ate) / rct_ate
    
    cates_dr = dr.estimate_cate(X)
    se_dr = np.std(cates_dr) / np.sqrt(n)
    ci_dr = (ate_dr - 1.96*se_dr, ate_dr + 1.96*se_dr)
    
    results.append(("Doubly Robust", ate_dr, ci_dr, bias_dr))
    
    print("\nBenchmark Results")
    print(f"{'Estimator':<15} | {'ATE Estimate':<12} | {'95% CI':<25} | {'Bias vs RCT':<12}")
    for name, ate, ci, bias in results:
        print(f"{name:<15} | ${ate:<11.2f} | (${ci[0]:.2f}, ${ci[1]:.2f}) | {bias*100:.1f}%")
        
if __name__ == "__main__":
    test_estimators_lalonde()
