import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from causal.estimation import CausalEstimator

def test_estimators_lalonde():
    # Load LaLonde
    url = "https://users.nber.org/~rdehejia/data/nsw_dw.csv"
    df = pd.read_csv(url)
    
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
    results.append(("Naive OLS", ate_ols, bias_ols))
    
    # 2. IPW
    prop_model = LogisticRegression(max_iter=1000).fit(X, T)
    p = prop_model.predict_proba(X)[:, 1]
    p = np.clip(p, 0.05, 0.95)
    ipw_y1 = np.mean(T * Y / p)
    ipw_y0 = np.mean((1 - T) * Y / (1 - p))
    ate_ipw = ipw_y1 - ipw_y0
    bias_ipw = abs(ate_ipw - rct_ate) / rct_ate
    results.append(("IPW", ate_ipw, bias_ipw))
    
    # 3. Doubly Robust
    dr = CausalEstimator(method="doubly_robust")
    dr.fit(Y, T, X)
    ate_dr = dr.estimate_ate(X)
    bias_dr = abs(ate_dr - rct_ate) / rct_ate
    
    # Approx CI for DR (just for display in table, using standard deviation of CATEs)
    cates_dr = dr.estimate_cate(X)
    se_dr = np.std(cates_dr) / np.sqrt(len(X))
    ci_dr = (ate_dr - 1.96*se_dr, ate_dr + 1.96*se_dr)
    
    results.append(("Doubly Robust", ate_dr, bias_dr))
    
    print("\nBenchmark Results")
    print(f"{'Estimator':<15} | {'ATE Estimate':<12} | {'Bias vs RCT':<12}")
    for name, ate, bias in results:
        print(f"{name:<15} | ${ate:<11.2f} | {bias*100:.1f}%")
        
    # Assertions
    assert abs(ate_ols - rct_ate) <= tol, f"OLS failed: {ate_ols}"
    assert abs(ate_ipw - rct_ate) <= tol, f"IPW failed: {ate_ipw}"
    assert abs(ate_dr - rct_ate) <= tol, f"DR failed: {ate_dr}"
    assert bias_dr < bias_ols, "DR should be strictly closer to ground truth than Naive OLS"

if __name__ == "__main__":
    test_estimators_lalonde()
